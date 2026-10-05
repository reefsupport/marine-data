"""The unified taxonomy: pinned WoRMS snapshot, semver, per-release diff, label gate.

Runtime reads ONLY committed files — ``registry/taxonomy/taxonomy.yaml`` (the version
and the snapshot it pins), the snapshot parquet, the schemas, the crosswalks and the
observed vocabularies in ``registry/taxonomy/vocab/``. Network access lives in
:mod:`marinedata.worms_snapshot` (codegen) and nowhere else.

Three checks, all offline:

* **nodes** — every taxon-axis node of a canonical schema carries an accepted AphiaID
  that the snapshot confirms (name, rank, status), or ``non_taxon: true`` with a reason.
* **vocabularies** — every observed source label has a crosswalk edge. A label with no
  edge is a *silent drop* and fails; an ``unmappable`` edge is a listed, reasoned drop
  and counts against the mapped share (instance-weighted where counts exist).
* **coverage** — every labelled image source has a crosswalk or a documented exception.

The CI gate is scoped (D-Q, :func:`scoped_gate`): it fails only for sources staged in
rs-storage-open ``sources/`` or named in a release manifest; the rest are listed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import coralnet_labels
from .registry import Registry
from .schema import Axis, Fidelity, NodeFacts

MIN_MAPPED = 0.95
WARN_MAPPED = 0.90
"""Soft floor reported as WARN by ``labels check`` (:mod:`marinedata.labels_check`)."""
L2_TASK = "benthic-l2"
LABELLED_KINDS = frozenset(
    {"point-label", "dense-mask", "instance-mask", "bbox", "image-label", "track"}
)


class TaxonomyGateError(Exception):
    """The release label gate failed."""


def taxonomy_dir(registry_root: str | Path) -> Path:
    return Path(registry_root) / "taxonomy"


def load_meta(registry_root: str | Path) -> dict:
    return yaml.safe_load((taxonomy_dir(registry_root) / "taxonomy.yaml").read_text())


def load_snapshot(registry_root: str | Path) -> dict[int, dict]:
    """AphiaID → snapshot row, from the pinned parquet named in taxonomy.yaml."""
    import pyarrow.parquet as pq

    meta = load_meta(registry_root)
    table = pq.read_table(taxonomy_dir(registry_root) / meta["snapshot"])
    return {int(r["aphia_id"]): r for r in table.to_pylist()}


# ── nodes ────────────────────────────────────────────────────────────────────────


def check_nodes(registry: Registry, snapshot: dict[int, dict]) -> list[str]:
    """Problems with canonical taxon nodes; empty means 100% anchored or declared.

    Strict: the snapshot must say ``accepted`` and resolve the id to itself. Informal
    WoRMS names (``Pisces``, ``... incertae sedis``) never anchor a canonical node.
    """
    out: list[str] = []
    for schema in registry.schemas:
        if not schema.canonical:
            continue
        for node in schema.nodes:
            if node.axis is not Axis.TAXON:
                continue
            where = f"{schema.id}:{node.id}"
            if node.worms_aphia_id is None:
                if not node.non_taxon:
                    out.append(f"{where}: no AphiaID and not declared non_taxon")
                continue
            row = snapshot.get(node.worms_aphia_id)
            if row is None:
                out.append(f"{where}: AphiaID {node.worms_aphia_id} absent from the snapshot")
            elif row["status"] != "accepted" or row["accepted_aphia_id"] != node.worms_aphia_id:
                out.append(
                    f"{where}: AphiaID {node.worms_aphia_id} is {row['status']} "
                    f"(accepted {row['accepted_aphia_id']} {row['accepted_name']})"
                )
            elif (row["scientific_name"], row["rank"]) != (
                node.worms_scientificname,
                node.worms_rank,
            ):
                out.append(
                    f"{where}: frozen {node.worms_scientificname}/{node.worms_rank} but "
                    f"snapshot says {row['scientific_name']}/{row['rank']}"
                )
    return out


def node_facts(
    registry: Registry, node_id: str | None, schema_id: str = "rs-benthic-v1"
) -> NodeFacts:
    """``(taxon_rank, worms_aphia_id, rs_benthic_code)`` of one canonical node.

    The L2 vocabulary is the ``benthic-l2`` task's class list (MariMap ``CoralLabelCode``),
    so the code follows the registry rather than a second hard-coded list.
    """
    l2 = set(registry.task(L2_TASK).classes) if _has_task(registry, L2_TASK) else set()
    return registry.label_schema(schema_id).node_facts(node_id, l2)


def _has_task(registry: Registry, task_id: str) -> bool:
    return any(t.id == task_id for t in registry.tasks)


def node_counts(registry: Registry) -> dict[str, int]:
    nodes = [n for s in registry.schemas if s.canonical for n in s.nodes if n.axis is Axis.TAXON]
    return {
        "nodes": len(nodes),
        "aphia": sum(n.worms_aphia_id is not None for n in nodes),
        "non_taxon": sum(n.non_taxon for n in nodes),
    }


# ── vocabularies ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class VocabAudit:
    source_id: str
    crosswalk_id: str
    weighted: bool
    counts: dict[str, int | None]
    mapped: tuple[str, ...]
    unmappable: dict[str, str] = field(default_factory=dict)
    silent_drops: tuple[str, ...] = ()
    dead_targets: tuple[str, ...] = ()

    @property
    def coverage(self) -> float:
        """Share of annotations (or label types when no counts exist) with a target."""
        if self.weighted:
            total = sum(v or 0 for v in self.counts.values())
            hit = sum(self.counts[k] or 0 for k in self.mapped)
        else:
            total, hit = len(self.counts), len(self.mapped)
        return hit / total if total else 0.0

    def line(self) -> str:
        flag = "✗" if self.silent_drops or self.dead_targets else "✓"
        kind = "annotations" if self.weighted else "types"
        return (
            f"{flag} {self.source_id:<22} {self.crosswalk_id:<22} {self.coverage:>7.2%} of "
            f"{kind}  {len(self.counts):>3} labels  {len(self.unmappable)} unmappable  "
            f"{len(self.silent_drops)} silent"
        )


def read_vocab_rows(path: str | Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """Header comments and one dict per label row (``label``, ``count``, ``label_native_id``,
    ``description``; absent columns are ``""``). Columns are read by header name, so the
    optional ``label_native_id`` column can sit anywhere after ``label``."""
    head: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    columns = ["label", "count"]
    for line in Path(path).read_text().splitlines():
        if line.startswith("# "):
            key, _, value = line[2:].partition(": ")
            head[key] = value
        elif line.startswith("label\t"):
            columns = line.split("\t")
        elif line:
            cells = line.split("\t")
            row = dict.fromkeys(("label", "count", "label_native_id", "description"), "")
            row.update(zip(columns, cells, strict=False))
            rows.append(row)
    return head, rows


def read_vocab(path: str | Path) -> tuple[dict[str, str], dict[str, int | None]]:
    head, rows = read_vocab_rows(path)
    return head, {r["label"]: int(r["count"]) if r["count"] else None for r in rows}


def crosswalk_for(registry: Registry, registry_root: str | Path, crosswalk_id: str):
    """The crosswalk a vocab TSV names. ``coralnet-label-id`` is a resolver: curated edges
    plus a functional-group fallback for any public CoralNet label id (D-S2)."""
    if crosswalk_id == coralnet_labels.CROSSWALK_ID:
        return coralnet_labels.resolver(registry, registry_root)
    return registry.crosswalk(crosswalk_id)


def audit_crosswalk(
    source_id: str,
    crosswalk,
    target_ids: set[str],
    counts: dict[str, int | None],
    *,
    weighted: bool,
    label_ids: dict[str, str] | None = None,
) -> VocabAudit:
    """Audit observed ``counts`` against ``crosswalk``. ``label_ids`` (label -> native id)
    makes edges match by id first, name second. ``narrower`` counts as mapped."""
    ids = label_ids or {}
    mapped, unmappable, silent, dead = [], {}, [], []
    for label in counts:
        edge = crosswalk.edge(label, ids.get(label) or None)
        if edge is None:
            silent.append(label)
        elif edge.fidelity is Fidelity.UNMAPPABLE:
            unmappable[label] = edge.note or ""
        else:
            mapped.append(label)
            dead += [f"{label}->{t}" for t in edge.targets.values() if t not in target_ids]
    return VocabAudit(
        source_id=source_id,
        crosswalk_id=crosswalk.id,
        weighted=weighted,
        counts=counts,
        mapped=tuple(mapped),
        unmappable=unmappable,
        silent_drops=tuple(silent),
        dead_targets=tuple(dead),
    )


def audit_vocab(
    registry: Registry, path: str | Path, registry_root: str | Path | None = None
) -> VocabAudit:
    head, rows = read_vocab_rows(path)
    counts = {r["label"]: int(r["count"]) if r["count"] else None for r in rows}
    root = registry_root if registry_root is not None else Path(path).resolve().parents[2]
    crosswalk = crosswalk_for(registry, root, head["crosswalk"])
    target = registry.label_schema(crosswalk.target_schema)
    return audit_crosswalk(
        head["source"],
        crosswalk,
        {n.id for n in target.nodes},
        counts,
        weighted=head.get("counts") == "annotations",
        label_ids={r["label"]: r["label_native_id"] for r in rows if r["label_native_id"]},
    )


def audit_all(registry: Registry, registry_root: str | Path) -> list[VocabAudit]:
    vocab = taxonomy_dir(registry_root) / "vocab"
    return [audit_vocab(registry, p, registry_root) for p in sorted(vocab.glob("*.tsv"))]


# ── gate ─────────────────────────────────────────────────────────────────────────


def labelled_sources(registry: Registry) -> list[str]:
    """Image/video sources whose annotations carry classes."""
    out = []
    for s in registry.sources:
        mods = {getattr(m, "value", str(m)) for m in s.modalities}
        if not any("image" in m or "video" in m for m in mods):
            continue
        if any(a.kind.value in LABELLED_KINDS for a in s.annotations):
            out.append(s.id)
    return out


OPEN_BUCKET = "rs-storage-open"
OPEN_VOCABULARY = "open_vocabulary"


def staged_sources(registry: Registry) -> list[str]:
    """Sources whose registry ``access`` points at ``sources/`` in rs-storage-open (D-D)."""
    out = []
    for s in registry.sources:
        uri, params = s.access.uri or "", s.access.params or {}
        if uri.startswith(f"s3://{OPEN_BUCKET}/sources/") or (
            params.get("bucket") == OPEN_BUCKET
            and str(params.get("prefix", "")).startswith("sources/")
        ):
            out.append(s.id)
    return out


def release_sources(manifests: list[str | Path]) -> set[str]:
    """Source ids named in release manifests (``RELEASE.json`` ``sources[].id``)."""
    out: set[str] = set()
    for path in manifests:
        out |= {s["id"] for s in json.loads(Path(path).read_text()).get("sources", [])}
    return out


def _gate_items(
    registry: Registry, registry_root: str | Path, min_mapped: float
) -> list[tuple[str | None, str]]:
    """Every failure as ``(source_id, message)``; ``None`` = schema-wide (never scoped out)."""
    meta = load_meta(registry_root)
    exceptions: dict[str, str] = meta.get("coverage_exceptions") or {}
    no_crosswalk: dict[str, str] = meta.get("no_crosswalk_yet") or {}
    declared: dict[str, dict] = meta.get("source_crosswalks") or {}
    items: list[tuple[str | None, str]] = [
        (None, f) for f in check_nodes(registry, load_snapshot(registry_root))
    ]
    audits = audit_all(registry, registry_root)
    for a in audits:
        sid = a.source_id
        items += [(sid, f"{sid}: SILENT DROP {label!r}") for label in a.silent_drops]
        items += [(sid, f"{sid}: dead target {t}") for t in a.dead_targets]
        items += [
            (sid, f"{sid}: unmappable {k!r} has no reason")
            for k, v in a.unmappable.items()
            if not v
        ]
        if a.coverage < min_mapped and sid not in exceptions:
            items.append(
                (sid, f"{sid}: {a.coverage:.2%} mapped < {min_mapped:.0%} and no exception")
            )
    covered = {a.source_id for a in audits}
    for sid in labelled_sources(registry):
        spec = registry.source(sid).loader
        if sid in covered:
            continue
        if spec and spec.crosswalk_id:
            # D-S1: a crosswalk with no observed vocabulary is unmeasured, not a pass.
            items.append(
                (
                    sid,
                    f"{sid}: crosswalk {spec.crosswalk_id!r} but no vocab TSV "
                    "(coverage unmeasured; add registry/taxonomy/vocab/<source>.tsv)",
                )
            )
            continue
        kind = (declared.get(sid) or {}).get("crosswalk")
        if kind == OPEN_VOCABULARY and (declared[sid].get("reason") or "").strip():
            continue
        if kind is not None:
            items.append(
                (sid, f"{sid}: declared crosswalk {kind!r} needs {OPEN_VOCABULARY!r} + a reason")
            )
        elif sid in no_crosswalk:
            items.append(
                (
                    sid,
                    f"{sid}: no_crosswalk_yet is not a release-gate exemption for 1.0+ "
                    f"({no_crosswalk[sid]})",
                )
            )
        else:
            items.append(
                (sid, f"{sid}: labelled source with no crosswalk and no documented exception")
            )
    return items


def gate(
    registry: Registry,
    registry_root: str | Path,
    *,
    source_ids: list[str] | None = None,
    min_mapped: float = MIN_MAPPED,
) -> list[str]:
    """Every failure the release label gate sees; an empty list passes.

    ``source_ids`` narrows the vocabulary and coverage checks to one release's sources.
    """
    scope = set(source_ids) if source_ids is not None else None
    return [
        msg
        for sid, msg in _gate_items(registry, registry_root, min_mapped)
        if sid is None or scope is None or sid in scope
    ]


def scoped_gate(
    registry: Registry,
    registry_root: str | Path,
    *,
    releases: list[str | Path] = (),
    min_mapped: float = MIN_MAPPED,
) -> tuple[list[str], list[str]]:
    """D-Q: ``(fails, listed)``. Failures count only for staged sources, sources named in a
    release manifest, and schema-wide node problems; registered-but-not-staged sources
    are returned in ``listed`` (``taxonomy check --strict`` prints them, never fails)."""
    scope = set(staged_sources(registry)) | release_sources(list(releases))
    fails, listed = [], []
    for sid, msg in _gate_items(registry, registry_root, min_mapped):
        (fails if sid is None or sid in scope else listed).append(msg)
    return fails, listed


def assert_release_gate(
    registry: Registry, registry_root: str | Path, source_ids: list[str]
) -> dict[str, str]:
    """Raise :class:`TaxonomyGateError` on any failure; return the release's taxonomy stamp."""
    fails = gate(registry, registry_root, source_ids=source_ids)
    if fails:
        raise TaxonomyGateError("label gate failed:\n  " + "\n  ".join(fails[:40]))
    meta = load_meta(registry_root)
    return {"taxonomy_version": str(meta["version"]), "taxonomy_snapshot": meta["snapshot"]}


# ── semver + diff ────────────────────────────────────────────────────────────────


def manifest(registry: Registry, registry_root: str | Path) -> dict:
    """Machine-readable taxonomy state: canonical nodes + crosswalk edges."""
    meta = load_meta(registry_root)
    nodes = {
        f"{s.id}:{n.id}": {
            "parent": n.parent,
            "axis": n.axis.value,
            "aphia": n.worms_aphia_id,
            "name": n.worms_scientificname or n.name,
            "rank": n.worms_rank,
            "non_taxon": n.non_taxon,
        }
        for s in registry.schemas
        if s.canonical
        for n in s.nodes
    }
    edges = {
        f"{c.id}:{e.source_label}": {
            "targets": {k.value: v for k, v in e.targets.items()},
            "fidelity": e.fidelity.value,
        }
        for c in registry.crosswalks
        for e in c.edges
    }
    return {
        "taxonomy_version": str(meta["version"]),
        "snapshot": meta["snapshot"],
        "nodes": nodes,
        "edges": edges,
    }


def diff(a: dict, b: dict) -> dict:
    """Added / removed / changed nodes and edges between two manifests."""
    out: dict = {"from": a.get("taxonomy_version"), "to": b.get("taxonomy_version")}
    for key in ("nodes", "edges"):
        old, new = a.get(key, {}), b.get(key, {})
        out[key] = {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "changed": {
                k: {"from": old[k], "to": new[k]}
                for k in sorted(set(old) & set(new))
                if old[k] != new[k]
            },
        }
    return out


def load_manifest(ref: str, registry_root: str | Path) -> dict:
    """A manifest by version (``registry/taxonomy/releases/<v>.json``) or by path."""
    path = Path(ref)
    if not path.exists():
        path = taxonomy_dir(registry_root) / "releases" / f"{ref}.json"
    return json.loads(path.read_text())
