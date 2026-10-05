"""``marinedata labels check`` — crosswalk coverage for every source id the project knows.

Scope (spec 3.3 point 5): the union of ``registry/sources/*.yaml`` and
``registry/ingest-specs/*.yaml`` (118 ingested ids exist only as specs and the older
``taxonomy check`` never saw them), plus ids named in a release file, an optional audit
cache (``audit.tsv`` style) and, behind ``--bucket``, the ids staged in ``rs-storage-open``.

Per id the native label vocabulary is a local vocab TSV (annotation-weighted when it carries
counts) or, failing that, the crosswalk's own edges. Status:

* ``pass``  — mapped >= ``MIN_MAPPED`` (0.95), no contract violation.
* ``warn``  — mapped in [0.90, 0.95) on an ungated source, or a floor waived by
  ``coverage_exceptions``.
* ``fail``  — mapped < 0.90, or < 0.95 on a gated source, or a silent drop / dead target /
  ``unmapped`` edge without a note (the crosswalk contract), or an unknown release id.
* ``n/a``   — no class labels, an open vocabulary, or labels not declared.
* ``missing-crosswalk`` — class labels (staged or declared) and no crosswalk to measure.

``narrower`` counts as mapped but abstains in projection. Gated = named in
``--release-sources``, or every id with ``--strict``; exit 1 when a gated id is ``fail`` or
``missing-crosswalk``. Without either flag the check is a report and exits 0.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from .licence_class import collapse_aliases, staged_id
from .registry import Registry, _default_root
from .taxonomy import (
    MIN_MAPPED,
    OPEN_BUCKET,
    OPEN_VOCABULARY,
    WARN_MAPPED,
    VocabAudit,
    audit_all,
    audit_crosswalk,
    crosswalk_for,
    labelled_sources,
    load_meta,
)

PASS, WARN, FAIL, NA, MISSING = "pass", "warn", "fail", "n/a", "missing-crosswalk"
STATUSES = (PASS, WARN, FAIL, NA, MISSING)
CLASS_KINDS = frozenset({"boxes", "masks", "image-class", "points", "tracks"})
TSV_COLUMNS = (
    "source_id",
    "origin",
    "ingested",
    "status",
    "mapped_pct",
    "basis",
    "n_labels",
    "n_unmapped",
    "crosswalk_id",
    "crosswalk_missing",
    "gated",
    "reasons",
    "unmapped",
)
TSV_UNMAPPED_CAP = 25


class LabelsCheckError(Exception):
    """The check could not run (unreadable input, bucket listing failure)."""


@dataclass(frozen=True)
class LabelRow:
    source_id: str
    origin: str  # registry | spec | both | cache | bucket | release
    ingested: bool
    status: str
    mapped_pct: float | None = None
    basis: str = ""  # annotations | label-types | crosswalk-edges
    n_labels: int = 0
    n_unmapped: int = 0
    crosswalk_id: str = ""
    crosswalk_missing: bool = False
    gated: bool = False
    reasons: tuple[str, ...] = ()
    unmapped: tuple[str, ...] = field(default=(), repr=False)

    @property
    def blocks(self) -> bool:
        return self.gated and self.status in (FAIL, MISSING)


# ── grading ──────────────────────────────────────────────────────────────────────


def grade(
    audit: VocabAudit,
    *,
    gated: bool = False,
    exception: bool = False,
    min_mapped: float = MIN_MAPPED,
    warn_mapped: float = WARN_MAPPED,
) -> tuple[str, tuple[str, ...]]:
    """``(status, reasons)`` for one measured vocabulary."""
    contract = []
    if audit.silent_drops:
        contract.append(f"{len(audit.silent_drops)} silent drop(s): label has no crosswalk edge")
    if audit.dead_targets:
        contract.append(f"{len(audit.dead_targets)} dead target(s)")
    if missing_note := [k for k, v in audit.unmappable.items() if not v.strip()]:
        contract.append(f"{len(missing_note)} unmapped edge(s) without a note")
    if not audit.counts:
        contract.append("empty vocabulary")
    if contract:
        return FAIL, tuple(contract)
    cov = audit.coverage
    if cov >= min_mapped:
        return PASS, ()
    why = f"{cov:.1%} mapped < {min_mapped:.0%}"
    if exception:
        return WARN, (f"{why}; waived by coverage_exceptions",)
    if gated or cov < warn_mapped:
        return FAIL, (why,)
    return WARN, (f"{why} (>= {warn_mapped:.0%}; fails if a release includes it)",)


def _unmapped_labels(audit: VocabAudit) -> tuple[str, ...]:
    """Silent drops first, then explicit ``unmapped`` edges; heaviest first when weighted."""
    explicit = sorted(audit.unmappable, key=lambda k: -(audit.counts.get(k) or 0))
    return (*audit.silent_drops, *explicit)


# ── inputs ───────────────────────────────────────────────────────────────────────


def spec_ids(spec_dir: Path) -> set[str]:
    out: set[str] = set()
    for path in sorted(spec_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and isinstance(data.get("id"), str):
            out.add(data["id"])
    return out


def read_release_sources(path: str | Path) -> set[str]:
    """Ids from a ``--release-sources`` file: one per line (``#`` comments) or RELEASE.json."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".json":
        return {s["id"] for s in json.loads(text).get("sources", [])}
    return {line.split("#")[0].strip() for line in text.splitlines() if line.split("#")[0].strip()}


def read_audit_cache(path: str | Path) -> dict[str, dict[str, str]]:
    """Offline stand-in for the bucket: ``audit.tsv`` rows keyed by id."""
    with Path(path).open(newline="") as fh:
        return {r["id"]: r for r in csv.DictReader(fh, delimiter="\t")}


def bucket_label_counts(client=None, bucket: str = OPEN_BUCKET) -> dict[str, int]:
    """Label-object count per source id, from ONE flat read-only listing of ``sources/``.

    Credentials come from ``~/.config/rclone/rclone.conf`` ``[rs-hel1]`` through configparser
    and are never printed. Any listing error aborts: an unverified bucket is not a pass.
    """
    if client is None:
        import boto3

        cfg = configparser.ConfigParser()
        cfg.read(Path.home() / ".config" / "rclone" / "rclone.conf")
        sec = cfg["rs-hel1"]
        client = boto3.client(
            "s3",
            endpoint_url=sec["endpoint"],
            aws_access_key_id=sec["access_key_id"],
            aws_secret_access_key=sec["secret_access_key"],
        )
    counts: dict[str, int] = {}
    try:
        for page in client.get_paginator("list_objects_v2").paginate(
            Bucket=bucket, Prefix="sources/"
        ):
            for obj in page.get("Contents", []):
                parts = obj["Key"].split("/")
                if len(parts) > 4 and parts[3] == "labels":
                    counts[parts[1]] = counts.get(parts[1], 0) + 1
    except Exception as exc:
        raise LabelsCheckError(f"bucket listing failed ({type(exc).__name__}); unverified") from exc
    return counts


# ── evaluation ───────────────────────────────────────────────────────────────────


def _class_bearing(
    sid: str, registry_labelled: set[str], cache: dict[str, str] | None, staged: int
) -> tuple[bool, str]:
    """``(has class labels, why not)``."""
    if sid in registry_labelled:
        return True, ""
    kinds: set[str] = set()
    if cache:
        for col in ("label_kinds", "label_kinds_declared"):
            kinds |= {k for k in (cache.get(col) or "").split(",") if k}
    if kinds & CLASS_KINDS or staged:
        return True, ""
    if kinds - {"none", "unknown"}:
        return False, "no class labels (" + ",".join(sorted(kinds - {"none", "unknown"})) + ")"
    if "unknown" in kinds:
        return False, "labels declared unknown"
    return False, "no label information"


def evaluate(
    registry: Registry,
    registry_root: str | Path,
    *,
    spec_dir: Path | None = None,
    audit_cache: dict[str, dict[str, str]] | None = None,
    bucket_labels: dict[str, int] | None = None,
    release: Iterable[str] = (),
    strict: bool = False,
    only: Iterable[str] = (),
) -> list[LabelRow]:
    root = Path(registry_root)
    meta = load_meta(root)
    exceptions = set(meta.get("coverage_exceptions") or {})
    declared = meta.get("source_crosswalks") or {}
    reg_ids = {s.id for s in registry.sources}
    specs = spec_ids(spec_dir if spec_dir is not None else root / "ingest-specs")
    cache = audit_cache or {}
    staged_labels = bucket_labels or {}
    release_ids = set(release)
    labelled = set(labelled_sources(registry))
    vocab = {a.source_id: a for a in audit_all(registry, root)}
    crosswalk_ids = {c.id for c in registry.crosswalks}
    unalias = {staged_id(r): r for r in reg_ids if staged_id(r) != r}
    release_ids = {unalias.get(r, r) for r in release_ids}
    ids = collapse_aliases(reg_ids | specs | set(cache) | set(staged_labels) | release_ids, reg_ids)
    if only:
        ids = [i for i in ids if i in set(only)]

    rows = []
    for sid in ids:
        origin = (
            "both"
            if sid in reg_ids and (sid in specs or staged_id(sid) in specs)
            else "registry"
            if sid in reg_ids
            else "spec"
            if sid in specs
            else "cache"
            if sid in cache
            else "bucket"
            if sid in staged_labels
            else "release"
        )
        staged = staged_id(
            sid
        )  # WP-R4: the bucket/audit key (atlantis-synthetic-depth -> atlantis)
        ingested = {sid, staged} & (set(cache) | set(staged_labels)) != set()
        gated = strict or sid in release_ids
        base = {"source_id": sid, "origin": origin, "ingested": bool(ingested), "gated": gated}
        if origin == "release":
            rows.append(
                LabelRow(
                    **base,
                    status=FAIL,
                    reasons=("release source id is in no registry source, ingest spec or cache",),
                )
            )
            continue
        rows.append(
            _row_for(
                sid,
                base,
                registry=registry,
                root=root,
                labelled=labelled,
                vocab=vocab.get(sid),
                cache=cache.get(sid) or cache.get(staged),
                staged=staged_labels.get(sid, 0) or staged_labels.get(staged, 0),
                declared=declared.get(sid),
                exception=sid in exceptions,
                crosswalk_ids=crosswalk_ids,
                in_registry=sid in reg_ids,
            )
        )
    return rows


def _row_for(
    sid: str,
    base: dict,
    *,
    registry: Registry,
    root: Path,
    labelled: set[str],
    vocab: VocabAudit | None,
    cache: dict[str, str] | None,
    staged: int,
    declared: dict | None,
    exception: bool,
    crosswalk_ids: set[str],
    in_registry: bool,
) -> LabelRow:
    if (declared or {}).get("crosswalk") == OPEN_VOCABULARY or (cache or {}).get(
        "crosswalk_file"
    ) == OPEN_VOCABULARY:
        return LabelRow(**base, status=NA, reasons=("open vocabulary",))
    bearing, why_not = _class_bearing(sid, labelled, cache, staged)
    audit = vocab
    basis = ""
    crosswalk_id = vocab.crosswalk_id if vocab else ""
    if audit is None:
        crosswalk_id = _crosswalk_id(registry, sid, cache, crosswalk_ids, in_registry)
        if crosswalk_id:
            audit = _edge_audit(registry, root, sid, crosswalk_id)
            basis = "crosswalk-edges"
    if audit is None:
        if not bearing:
            return LabelRow(**base, status=NA, reasons=(why_not,))
        return LabelRow(
            **base,
            status=MISSING,
            crosswalk_missing=True,
            reasons=("class labels and no crosswalk, vocab or open_vocabulary entry",),
        )
    status, reasons = grade(audit, gated=base["gated"], exception=exception)
    if basis:
        reasons = (*reasons, "vocabulary unobserved: measured on crosswalk edges")
    unmapped = _unmapped_labels(audit)
    return LabelRow(
        **base,
        status=status,
        mapped_pct=round(100 * audit.coverage, 2),
        basis=basis or ("annotations" if audit.weighted else "label-types"),
        n_labels=len(audit.counts),
        n_unmapped=len(unmapped),
        crosswalk_id=crosswalk_id or audit.crosswalk_id,
        reasons=reasons,
        unmapped=unmapped,
    )


def _crosswalk_id(
    registry: Registry,
    sid: str,
    cache: dict[str, str] | None,
    crosswalk_ids: set[str],
    in_registry: bool,
) -> str:
    loader = registry.source(sid).loader if in_registry else None
    if loader and loader.crosswalk_id:
        return loader.crosswalk_id
    stem = Path((cache or {}).get("crosswalk_file") or "").stem
    return stem if stem in crosswalk_ids else ""


def _edge_audit(registry: Registry, root: Path, sid: str, crosswalk_id: str) -> VocabAudit | None:
    crosswalk = crosswalk_for(registry, root, crosswalk_id)
    edges = getattr(crosswalk, "edges", None)
    if not edges:
        return None
    target = registry.label_schema(crosswalk.target_schema)
    return audit_crosswalk(
        sid,
        crosswalk,
        {n.id for n in target.nodes},
        {e.source_label: None for e in edges},
        weighted=False,
        label_ids={e.source_label: e.source_label_id for e in edges if e.source_label_id},
    )


# ── output ───────────────────────────────────────────────────────────────────────


def summarise(rows: list[LabelRow]) -> dict:
    counts = {s: sum(r.status == s for r in rows) for s in STATUSES}
    return {
        "ids": len(rows),
        **counts,
        "spec_only": sum(r.origin == "spec" for r in rows),
        "spec_only_ingested": sum(r.origin == "spec" and r.ingested for r in rows),
        "gated": sum(r.gated for r in rows),
        "blocking": [r.source_id for r in rows if r.blocks],
        "min_mapped": MIN_MAPPED,
        "warn_mapped": WARN_MAPPED,
    }


def write_json(rows: list[LabelRow], path: str | Path) -> None:
    doc = {"summary": summarise(rows), "rows": [asdict(r) for r in rows]}
    Path(path).write_text(json.dumps(doc, indent=1) + "\n")


def write_tsv(rows: list[LabelRow], path: str | Path) -> None:
    with Path(path).open("w", newline="") as fh:
        out = csv.writer(fh, delimiter="\t", lineterminator="\n")
        out.writerow(TSV_COLUMNS)
        for r in rows:
            more = len(r.unmapped) - TSV_UNMAPPED_CAP
            shown = "|".join(r.unmapped[:TSV_UNMAPPED_CAP]) + (
                f"|(+{more} more)" if more > 0 else ""
            )
            out.writerow(
                [
                    r.source_id,
                    r.origin,
                    str(r.ingested).lower(),
                    r.status,
                    "" if r.mapped_pct is None else f"{r.mapped_pct:.2f}",
                    r.basis,
                    r.n_labels,
                    r.n_unmapped,
                    r.crosswalk_id,
                    str(r.crosswalk_missing).lower(),
                    str(r.gated).lower(),
                    "; ".join(r.reasons),
                    shown.replace("\t", " "),
                ]
            )


def run(args: argparse.Namespace, *, lister: Callable[[], dict[str, int]] | None = None) -> int:
    """Entry for ``marinedata labels check``; ``args.source_id[1:]`` narrows the ids."""
    root = _default_root()
    registry = Registry.load(root)
    try:
        cache = read_audit_cache(args.audit_cache) if args.audit_cache else None
        release = read_release_sources(args.release_sources) if args.release_sources else set()
        bucket = (lister or bucket_label_counts)() if args.bucket else None
    except (OSError, KeyError, LabelsCheckError) as exc:
        print(f"labels check: {exc}")
        return 2
    rows = evaluate(
        registry,
        root,
        audit_cache=cache,
        bucket_labels=bucket,
        release=release,
        strict=args.strict,
        only=args.source_id[1:],
    )
    if args.json:
        write_json(rows, args.json)
    if args.tsv:
        write_tsv(rows, args.tsv)
    shown = [r for r in rows if r.status in (WARN, FAIL, MISSING)]
    for r in shown if args.verbose else shown[:40]:
        pct = "  n/a " if r.mapped_pct is None else f"{r.mapped_pct:5.1f}%"
        print(f"{r.status:<17} {r.source_id:<46} {pct}  {'; '.join(r.reasons)[:90]}")
    if len(shown) > 40 and not args.verbose:
        print(f"... {len(shown) - 40} more (use -v, --json or --tsv)")
    s = summarise(rows)
    print(
        f"labels check: {s['ids']} ids  pass {s[PASS]}  warn {s[WARN]}  fail {s[FAIL]}  "
        f"n/a {s[NA]}  missing-crosswalk {s[MISSING]}  "
        f"(spec-only {s['spec_only']}, gated {s['gated']}, blocking {len(s['blocking'])})"
    )
    return 1 if s["blocking"] else 0
