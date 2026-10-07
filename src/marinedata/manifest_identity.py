"""D-X v1 byte-identity, proved at MANIFEST level with no image bytes (INT-core2b).

The original D-X check (``docs/integration-v2.md``, INT-core round) ran a real
``marinedata release build`` — resolving every admitted source's root via
:func:`marinedata.cli_release._resolve_roots`, which calls ``fetch_sample`` and
downloads the *whole* pinned staged tree (images included) for anything not passed as
``--local``. That is fine on a machine that already has the corpus staged, but on this
one it re-downloaded mermaid-aws and coralscapes (+14 GB) and drove free disk to
critical (see ``$T/briefs/2026-09-25-5star-INT-core2b.md``).

This module proves the same thing — that the row set, split assignment and per-row
metadata a v1 rebuild would produce still match what was published — from two things
that are never image bytes:

1. Each admitted source's already-staged ``metadata.parquet`` (``stem``, ``partition``,
   ``split_group``, ``width``, ``height`` — the staging pipeline's sample index, read
   the same way :func:`marinedata.release.enumerate_release_rows` does).
2. That source's ``CHECKSUMS.sha256`` (a single small text manifest, GET by known key —
   see :mod:`marinedata.fetchers_remote`), which maps every staged path to its sha256
   without ever fetching the path's *contents*.

Row identity (``sha256``) comes from (2); split comes from the frozen
``registry/SPLIT_MAP.json`` (``SplitMap.assignments``, keyed by ``split_group``); width/
height come from (1). None of the three requires a single image byte on disk.

Scope: this reconstructs the *row-level* manifest (every admitted source's rows, their
split and their sha256) — it does not replicate each task's own label-based inclusion
rule, so it is not a substitute for rebuilding the 6 task TSVs byte-for-byte. That full
task-level rebuild without image bytes needs the ``staged-tree`` loader itself to resolve
a sample's path from ``CHECKSUMS.sha256`` instead of a directory walk — out of scope
here, tracked for WP-8e (which already owns "key sha256 from S3 metadata.parquet,
CHECKSUMS.sha256 or the v1 manifest with no image download").
"""

from __future__ import annotations

import hashlib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .checksums import parse_checksums
from .fetch import _get, cache_root
from .fetchers_remote import _s3_endpoint
from .gate import evaluate
from .models import Source
from .registry import Registry
from .splitmap import SplitMap
from .tables import _require_pyarrow


class ChecksumsFetcher(Protocol):
    def __call__(self, source: Source) -> dict[str, str]: ...


def fetch_source_checksums(
    source: Source, *, timeout: int = 30, retries: int = 8
) -> dict[str, str]:
    """GET and verify ``<prefix>CHECKSUMS.sha256`` for one source over anonymous HTTPS.

    The one network call this module ever makes: one small text file (a path->sha256
    listing), never a file it names. Raises if the source has no S3 access, or if the
    downloaded manifest's own sha256 does not match ``source.checksums.root_digest``
    (the same check :func:`marinedata.fetchers_remote._fetch_s3_manifest` does before
    trusting anything it lists).
    """
    _, host, _ = _s3_endpoint(source)
    prefix = str(source.access.params.get("prefix", ""))
    key = f"{prefix}CHECKSUMS.sha256"
    payload = _get(f"https://{host}/{urllib.parse.quote(key)}", timeout=timeout, retries=retries)
    digest = hashlib.sha256(payload).hexdigest()
    if source.checksums is not None and digest != source.checksums.root_digest:
        raise ValueError(
            f"{source.id}: CHECKSUMS.sha256 digest {digest} does not match "
            f"checksums.root_digest {source.checksums.root_digest} — the pinned "
            "manifest is stale or has been tampered with"
        )
    return parse_checksums(payload.decode("utf-8"))


def _checksum_entry_for_row(
    checksums: dict[str, str], partition: str, stem: str
) -> tuple[str, str] | None:
    """The one ``(images/<partition>/<stem>.<ext>, sha256)`` entry for this row, or
    ``None`` if zero or more than one candidate matches (ambiguous — the caller treats it
    as a miss rather than guessing an extension)."""
    prefix = f"images/{partition}/{stem}." if partition else f"images/{stem}."
    matches = [(k, v) for k, v in checksums.items() if k.startswith(prefix)]
    return matches[0] if len(matches) == 1 else None


@dataclass(frozen=True)
class ManifestRow:
    sha256: str
    split: str
    source_id: str
    width: int
    height: int
    path: str = ""
    """The staged relative path (``images/<partition>/<stem>.<ext>``) — a CHECKSUMS.sha256
    key, never opened; only its parent directory name and stem are used downstream."""


def build_v1_row_manifest(
    registry: Registry,
    split_map: SplitMap,
    *,
    cache_dir: str | Path | None = None,
    profile: str = "research",
    fetch_checksums: ChecksumsFetcher = fetch_source_checksums,
    skipped: dict[str, str] | None = None,
) -> list[ManifestRow]:
    """Every admitted, staged-tree, already-locally-staged source's rows — sha256, split
    and (width, height) — with no image bytes read or fetched.

    A source without a local ``metadata.parquet`` under ``cache_dir`` (default
    :func:`marinedata.fetch.cache_root`) is skipped and recorded in ``skipped`` rather
    than fetched — fetching the full tree defeats the point of this check.
    """
    _require_pyarrow()
    import pyarrow.parquet as pq

    root = Path(cache_dir) if cache_dir is not None else cache_root()
    prof = registry.profile(profile)
    rows: list[ManifestRow] = []
    for source in registry:
        if source.loader is None or source.loader.layout != "staged-tree":
            continue
        if not evaluate(source, prof).allowed:
            continue
        if "needs-attribution" in source.tags:
            continue
        metadata_path = root / source.id / "metadata.parquet"
        if not metadata_path.is_file():
            if skipped is not None:
                skipped[source.id] = f"no local metadata.parquet under {metadata_path}"
            continue
        checksums = fetch_checksums(source)
        for record in pq.read_table(metadata_path).to_pylist():
            group = record.get("split_group")
            if not group:
                continue
            split = split_map.assignments.get(group)
            if split is None:
                if skipped is not None:
                    skipped[f"{source.id}:{group}"] = "split_group not in frozen SPLIT_MAP.json"
                continue
            entry = _checksum_entry_for_row(
                checksums, record.get("partition") or "", record["stem"]
            )
            if entry is None:
                raise ValueError(
                    f"{source.id}: no unique CHECKSUMS.sha256 entry for "
                    f"partition={record.get('partition')!r} stem={record['stem']!r}"
                )
            path, sha256 = entry
            rows.append(
                ManifestRow(
                    sha256=sha256,
                    split=split,
                    source_id=source.id,
                    width=record["width"],
                    height=record["height"],
                    path=path,
                )
            )
    return rows


def load_published_v1_metadata(hf_v1_root: str | Path) -> dict[str, tuple[str, int]]:
    """``{image_sha256: (split, min_side)}`` from the published v1 HF tree's own
    ``metadata`` config (``data/metadata/{train,validation,test}-*.parquet``) — the
    "non-image columns" the brief points at, never the ``data/images/*`` shards."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    root = Path(hf_v1_root)
    out: dict[str, tuple[str, int]] = {}
    for split_name, filename in (
        ("train", "train-00000-of-00001.parquet"),
        ("validation", "validation-00000-of-00001.parquet"),
        ("test", "test-00000-of-00001.parquet"),
    ):
        path = root / "data" / "metadata" / filename
        if not path.is_file():
            continue
        table = pq.read_table(path, columns=["image_sha256", "min_side"])
        for record in table.to_pylist():
            out[record["image_sha256"]] = (split_name, record["min_side"])
    return out


@dataclass(frozen=True)
class ManifestIdentityReport:
    """The D-X comparison result: row ids (sha256), splits and the one metadata column
    both sides carry (``min_side`` vs. ``min(width, height)``)."""

    rebuilt_rows: int
    published_rows: int
    missing_from_rebuild: tuple[str, ...]
    """sha256 present in the published v1 manifest but not reconstructed here — every
    one of these is a real regression unless it belongs to a source this run skipped."""
    extra_in_rebuild: tuple[str, ...]
    """sha256 reconstructed here but absent from the published manifest — expected for
    any v2-only source that has landed in the registry since v1 was published."""
    split_mismatches: tuple[str, ...]
    min_side_mismatches: tuple[str, ...]

    @property
    def identical(self) -> bool:
        return not (self.missing_from_rebuild or self.split_mismatches or self.min_side_mismatches)


def compare_v1_manifest(
    rows: list[ManifestRow], published: dict[str, tuple[str, int]]
) -> ManifestIdentityReport:
    from .hf_export import HF_SPLITS  # SPLIT_MAP `val` is published as HF `validation`

    rebuilt = {row.sha256: row for row in rows}
    missing = tuple(sorted(set(published) - set(rebuilt)))
    extra = tuple(sorted(set(rebuilt) - set(published)))
    split_mismatches = []
    min_side_mismatches = []
    for sha256, row in rebuilt.items():
        pub = published.get(sha256)
        if pub is None:
            continue
        pub_split, pub_min_side = pub
        if pub_split != HF_SPLITS.get(row.split, row.split):
            split_mismatches.append(sha256)
        if min(row.width, row.height) != pub_min_side:
            min_side_mismatches.append(sha256)
    return ManifestIdentityReport(
        rebuilt_rows=len(rebuilt),
        published_rows=len(published),
        missing_from_rebuild=missing,
        extra_in_rebuild=extra,
        split_mismatches=tuple(sorted(split_mismatches)),
        min_side_mismatches=tuple(sorted(min_side_mismatches)),
    )


def rebuild_v1_metadata_rows(
    rows: list[ManifestRow],
    registry: Registry,
    *,
    stage_root: str | Path,
    quality_by_sha: dict[str, dict],
    meow_polygons: tuple = (),
    backfill_root: str | Path | None = None,
) -> list[dict]:
    """Every column of the published v1 ``metadata`` config, rebuilt for ``rows`` by the
    SAME code the v1 build uses (:func:`marinedata.metadata_release.build_rows`) — with no
    image bytes. ``build_rows`` only reads ``ImageRef.file.parent.name`` and ``.stem``, so
    the CHECKSUMS.sha256 key stands in for the cached file path; licence/provenance come
    from the registry, ``upstream_id`` from the staged ``metadata.parquet``, and the quality
    columns from WP-1's already-computed ``quality.parquet``."""
    from .geo_backfill import BACKFILL_ROOT
    from .metadata_release import ImageRef, build_rows

    refs = [
        ImageRef(sha256=r.sha256, source_id=r.source_id, file=Path(r.path), split=r.split)
        for r in rows
    ]
    return build_rows(
        refs,
        registry,
        Path(stage_root),
        quality_by_sha,
        meow_polygons,
        backfill_root=Path(backfill_root) if backfill_root is not None else BACKFILL_ROOT,
    )


def load_published_v1_metadata_table(hf_v1_root: str | Path) -> dict[str, dict]:
    """``{image_sha256: row}`` with EVERY column of the published v1 ``metadata`` config
    plus a ``split`` key (the parquet file it came from) — never the image shards."""
    _require_pyarrow()
    import pyarrow.parquet as pq

    root = Path(hf_v1_root) / "data" / "metadata"
    out: dict[str, dict] = {}
    for split_name in ("train", "validation", "test"):
        path = root / f"{split_name}-00000-of-00001.parquet"
        if not path.is_file():
            continue
        for record in pq.read_table(path).to_pylist():
            out[record["image_sha256"]] = {**record, "split": split_name}
    return out


def _same(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True  # NaN == NaN for identity purposes
    return a == b


def compare_v1_metadata_columns(
    rebuilt: list[dict], published: dict[str, dict]
) -> dict[str, tuple[str, ...]]:
    """``{column: sha256s whose value differs}`` over the rows both sides carry, for every
    column the published table has (``split`` excluded — :func:`compare_v1_manifest`
    owns that). An empty tuple for every column means the metadata is identical."""
    columns = sorted({c for row in published.values() for c in row} - {"split"})
    mismatches: dict[str, list[str]] = {c: [] for c in columns}
    for row in rebuilt:
        pub = published.get(row["image_sha256"])
        if pub is None:
            continue
        for column in columns:
            if not _same(row.get(column), pub.get(column)):
                mismatches[column].append(row["image_sha256"])
    return {c: tuple(sorted(v)) for c, v in mismatches.items()}


# --- D-X2 (INT-core3): task-file identity at manifest level --------------------------


class ChecksumsDigest:
    """A drop-in for :func:`marinedata.checksums.file_digest` that never reads a byte of
    the file: the sha256 comes from the owning source's pinned ``CHECKSUMS.sha256``,
    keyed by the path relative to that source's root. Used by ``release build
    --manifest-only`` so the v1 task TSVs can be rebuilt byte-for-byte from the staged
    trees' listings alone. A path under no root, or a key the manifest does not list,
    raises (fail closed — never fall back to hashing, never guess)."""

    def __init__(self, roots: dict[str, Path], checksums: dict[str, dict[str, str]]) -> None:
        self._roots = sorted(
            ((Path(root).resolve(), source_id) for source_id, root in roots.items()),
            key=lambda item: len(str(item[0])),
            reverse=True,
        )
        self._checksums = checksums

    def __call__(self, path: Path) -> str:
        resolved = Path(path).resolve()
        for root, source_id in self._roots:
            try:
                key = resolved.relative_to(root).as_posix()
            except ValueError:
                continue
            listing = self._checksums.get(source_id, {})
            if key not in listing:
                raise KeyError(f"{source_id}: {key!r} is not listed in CHECKSUMS.sha256")
            return listing[key]
        raise KeyError(f"{path} is under no resolved source root")


def checksums_digest(
    registry: Registry,
    roots: dict[str, Path],
    fetch_checksums: ChecksumsFetcher = fetch_source_checksums,
) -> ChecksumsDigest:
    return ChecksumsDigest(
        roots, {source_id: fetch_checksums(registry.source(source_id)) for source_id in roots}
    )


def compare_task_dirs(expected: str | Path, actual: str | Path) -> dict[str, str]:
    """``{task file name: status}`` for every ``*.tsv`` on either side — ``identical``
    (byte-for-byte), ``differs``, ``missing`` (expected only) or ``extra`` (actual only)."""
    exp = {p.name: p for p in Path(expected).glob("*.tsv")}
    act = {p.name: p for p in Path(actual).glob("*.tsv")}
    out: dict[str, str] = {}
    for name in sorted(exp.keys() | act.keys()):
        if name not in act:
            out[name] = "missing"
        elif name not in exp:
            out[name] = "extra"
        else:
            same = exp[name].read_bytes() == act[name].read_bytes()
            out[name] = "identical" if same else "differs"
    return out


def v1_primary_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    """One row per sha256, chosen exactly as v1's ``hf_export.build_layout`` chose the
    ``images`` config's primary member: ``min(members, key=(source_id, sample_key))``,
    where the staged-tree loader's ``sample_key`` is the image path relative to the
    source root — i.e. :attr:`ManifestRow.path`. Without this, a sha staged under more
    than one stem (byte-identical duplicates) is compared once per stem and every
    non-primary stem reads as an ``upstream_id`` change."""
    best: dict[str, ManifestRow] = {}
    for row in rows:
        cur = best.get(row.sha256)
        if cur is None or (row.source_id, row.path) < (cur.source_id, cur.path):
            best[row.sha256] = row
    return [best[sha] for sha in sorted(best)]


class _NaN:
    """Marker for "a float NaN" in :data:`NULL_SENTINELS` (NaN never equals itself, so
    it cannot be matched by ``in``/``==``)."""

    def __repr__(self) -> str:
        return "NaN"


NAN = _NaN()

_ABSENT_STR: frozenset[object] = frozenset({""})
_ABSENT_FLOAT: frozenset[object] = frozenset({NAN})

NULL_SENTINELS: dict[str, frozenset[object]] = {
    # geo: `none` is the geo_precision enum member for "this row has no lat/lon" — a v1
    # `none` becoming `site` is the same enrichment as lat/lon going null -> value.
    "geo_precision": frozenset({"none", ""}),
    "geo_source": _ABSENT_STR,
    "lat": _ABSENT_FLOAT,
    "lon": _ABSENT_FLOAT,
    "gps_precision_m": _ABSENT_FLOAT,
    # capture context
    "capture_datetime": _ABSENT_STR,
    "depth_m": _ABSENT_FLOAT,
    "depth_source": _ABSENT_STR,
    "depth_zone": _ABSENT_STR,
    "platform": _ABSENT_STR,
    "camera": _ABSENT_STR,
    "habitat": _ABSENT_STR,
    # biogeography
    "meow_realm": _ABSENT_STR,
    "meow_province": _ABSENT_STR,
    "meow_ecoregion": _ABSENT_STR,
    # provenance fields v1 left unset
    "upstream_url": _ABSENT_STR,
    "lineage_root_digest": _ABSENT_STR,
}
"""D-X2a: per column, the v1 values that MEAN "absent". Such a value becoming a real
value is enrichment, exactly like ``None`` -> value (``None`` is absent in every
column). Listed explicitly per column: a column not listed has NO sentinel, so e.g.
``""`` -> value in ``license`` or NaN -> value in ``q_blur`` (a byte-derived metric over
byte-identical images) is a changed value and fails. Identity/provenance keys
(``image_sha256``, ``source_*``, ``license``, ``attribution``, ``upstream_id``,
``fetch_date``) and the byte-derived quality columns are deliberately absent.
Published v1 (69,600 rows, 2026-09-25) carries 0 ``""`` and 0 NaN in every column;
its only sentinel in use is ``geo_precision = none`` (68,350 rows)."""


def _is_null(value: object, column: str = "") -> bool:
    if value is None:
        return True
    for sentinel in NULL_SENTINELS.get(column, ()):
        if sentinel is NAN:
            if isinstance(value, float) and value != value:
                return True
        elif type(value) is type(sentinel) and value == sentinel:
            return True
    return False


@dataclass(frozen=True)
class ColumnDiff:
    enriched: tuple[str, ...]
    """sha256 whose v1 value was absent (``None`` or a :data:`NULL_SENTINELS` value for
    that column) and whose rebuilt value is not — allowed (D-X2 / D-X2a)."""
    changed: tuple[str, ...]
    """sha256 whose value differs any other way (value->other value, value->null) —
    a D-X2 violation."""


def classify_v1_column_diffs(
    rebuilt: list[dict], published: dict[str, dict]
) -> dict[str, ColumnDiff]:
    """D-X2: per published column, split every difference into allowed enrichment
    (absent -> value, D-X2a sentinels included) and forbidden change (anything else)."""
    columns = sorted({c for row in published.values() for c in row} - {"split"})
    enriched: dict[str, list[str]] = {c: [] for c in columns}
    changed: dict[str, list[str]] = {c: [] for c in columns}
    for row in rebuilt:
        sha = row["image_sha256"]
        pub = published.get(sha)
        if pub is None:
            continue
        for column in columns:
            new, old = row.get(column), pub.get(column)
            if _same(new, old):
                continue
            is_enrichment = _is_null(old, column) and not _is_null(new, column)
            (enriched if is_enrichment else changed)[column].append(sha)
    return {c: ColumnDiff(tuple(sorted(enriched[c])), tuple(sorted(changed[c]))) for c in columns}
