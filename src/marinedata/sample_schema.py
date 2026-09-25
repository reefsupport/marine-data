"""The canonical per-sample row — one schema for every staged source (D-K, WP-2 fields).

Every adapter-driven ingest (:mod:`marinedata.ingest_source`) writes its
``metadata.parquet`` in exactly this shape, so release/HF code (WP-2) can read any
source without per-source branches. Do not fork this schema; extend it here.

Null semantics (the contract — a null always means one specific thing):

* ``license`` / ``attribution`` are NEVER null or empty. A source that states no licence
  gets ``NOASSERTION`` (the SPDX term), never a guess (D-C: record, don't filter).
* ``upstream_id`` null = the upstream has no per-item identifier (rare; e.g. a raw tar
  member list with no stable names). ``upstream_url`` null = no per-item URL resolves
  (e.g. the image is a member inside an archive — the archive URL is in ``INGEST.json``).
* ``lineage_root_digest`` null = a first-hop ingest. It is the ``root_digest`` (sha256 of
  ``CHECKSUMS.sha256``) of the *parent staged version* for a derived source. A row can
  never carry its own version's root digest (the manifest covers this file), so that
  lives in ``INGEST.json`` and the registry stub. ``upstream_digest`` is the sha256 of the
  upstream container (file/archive/parquet shard) the bytes were decoded from.
* ``capture_datetime`` null = unknown. Always UTC; a naive upstream time is stored as UTC
  only when the source documents it as UTC, else null (no local-time guessing).
* ``lat``/``lon`` are both null or both set (WGS84 decimal degrees). ``gps_precision_m``
  null with a position set = precision unknown (e.g. site-level coordinates).
* ``depth_m`` is positive-down metres; null = unknown. ``depth_source`` is required when
  ``depth_m`` is set and must be one of :data:`DEPTH_SOURCES`.
* ``depth_zone`` is derived from ``depth_m`` by :func:`depth_zone_for` when a depth is
  known; it may be set from source-level facts (e.g. a deep-sea-only survey) with
  ``depth_m`` null. When both are set they must agree.
* ``meow_realm`` / ``meow_province`` / ``meow_ecoregion`` / ``habitat`` / ``platform`` /
  ``camera`` null = not stated upstream and not derived yet (derivation is WP-2's
  enrichment step, not the ingest's). The three MEOW fields are always all-null or
  all-set together (one point-in-polygon lookup fills all three or none).
* ``location_generalized`` is ``true`` when ``lat``/``lon`` were rounded to 0.1° and the
  precise position withheld (WP-2 location policy: a source tagged ``location-sensitive``
  in the registry, or an IUCN CR/EN taxon). ``false`` — never null — otherwise, including
  when there is no position to generalize.
* ``width``/``height`` null = the bytes did not decode with Pillow at ingest time (the
  image is still staged; ``image_format`` then comes from the file suffix).
* ``image_member`` is null in per-object layout; in shard layout ``image_path`` is the
  shard tar and ``image_member`` the tar member name.
* ``split_hint`` null = upstream declared no split. It is a hint: releases re-split.
* ``label_refs`` is an empty list (never null) when the sample carries no labels.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = "1"

DEPTH_SOURCES = frozenset({"sensor", "exif", "metadata", "site-nominal", "estimated"})
PLATFORMS = frozenset(
    {
        "diver",
        "snorkel",
        "rov",
        "auv",
        "towed",
        "drop-camera",
        "bruv",
        "lander",
        "uav",
        "vessel",
        "lab",
        "unknown",
    }
)
SPLIT_HINTS = frozenset({"train", "val", "test"})
_SPLIT_ALIASES = {
    "training": "train",
    "validation": "val",
    "valid": "val",
    "dev": "val",
    "testing": "test",
    "eval": "test",
}

# Depth zones, positive-down metres, [lower, upper). Shallow/mesophotic/rariphotic follow
# the reef literature (Baldwin et al. 2018); bathyal/abyssal/hadal the oceanographic ones.
DEPTH_ZONES: tuple[tuple[str, float, float], ...] = (
    ("shallow", 0.0, 30.0),
    ("mesophotic", 30.0, 150.0),
    ("rariphotic", 150.0, 300.0),
    ("bathyal", 300.0, 4000.0),
    ("abyssal", 4000.0, 6000.0),
    ("hadal", 6000.0, math.inf),
)
_ZONE_NAMES = frozenset(z for z, _, _ in DEPTH_ZONES)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SampleSchemaError(ValueError):
    """A row or table violates the canonical sample schema."""


def depth_zone_for(depth_m: float | None) -> str | None:
    """The zone a positive-down depth falls in; ``None`` for an unknown depth.
    Depths in (-2, 0) (tide/rounding noise at the surface) count as shallow."""
    if depth_m is None:
        return None
    d = max(depth_m, 0.0) if depth_m > -2.0 else depth_m
    for name, lo, hi in DEPTH_ZONES:
        if lo <= d < hi:
            return name
    return None


def normalise_split(value: str | None) -> str | None:
    """Map an upstream split name onto :data:`SPLIT_HINTS`; unknown names -> ``None``."""
    if not value:
        return None
    v = value.strip().lower()
    v = _SPLIT_ALIASES.get(v, v)
    return v if v in SPLIT_HINTS else None


@dataclass(frozen=True)
class SampleRow:
    """One staged image. Field order == column order in ``metadata.parquet``."""

    sample_id: str
    source_id: str
    source_version: str
    stem: str
    image_path: str
    image_sha256: str
    image_bytes: int
    image_format: str
    license: str
    attribution: str
    fetch_date: dt.date
    image_member: str | None = None
    width: int | None = None
    height: int | None = None
    upstream_id: str | None = None
    upstream_url: str | None = None
    upstream_digest: str | None = None
    lineage_root_digest: str | None = None
    capture_datetime: dt.datetime | None = None
    lat: float | None = None
    lon: float | None = None
    gps_precision_m: float | None = None
    depth_m: float | None = None
    depth_source: str | None = None
    platform: str | None = None
    camera: str | None = None
    meow_realm: str | None = None
    meow_province: str | None = None
    meow_ecoregion: str | None = None
    depth_zone: str | None = None
    habitat: str | None = None
    location_generalized: bool = False
    split_hint: str | None = None
    label_refs: tuple[str, ...] = field(default_factory=tuple)


FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(SampleRow))
REQUIRED: frozenset[str] = frozenset(
    {
        "sample_id",
        "source_id",
        "source_version",
        "stem",
        "image_path",
        "image_sha256",
        "image_bytes",
        "image_format",
        "license",
        "attribution",
        "fetch_date",
    }
)


def arrow_schema():
    """The pyarrow schema (imported lazily: pyarrow is the ``ingest`` extra)."""
    import pyarrow as pa

    s, f64, f32, i32 = pa.string(), pa.float64(), pa.float32(), pa.int32()
    types = {
        "sample_id": s,
        "source_id": s,
        "source_version": s,
        "stem": s,
        "image_path": s,
        "image_sha256": s,
        "image_bytes": pa.int64(),
        "image_format": s,
        "license": s,
        "attribution": s,
        "fetch_date": pa.date32(),
        "image_member": s,
        "width": i32,
        "height": i32,
        "upstream_id": s,
        "upstream_url": s,
        "upstream_digest": s,
        "lineage_root_digest": s,
        "capture_datetime": pa.timestamp("us", tz="UTC"),
        "lat": f64,
        "lon": f64,
        "gps_precision_m": f32,
        "depth_m": f32,
        "depth_source": s,
        "platform": s,
        "camera": s,
        "meow_realm": s,
        "meow_province": s,
        "meow_ecoregion": s,
        "depth_zone": s,
        "habitat": s,
        "location_generalized": pa.bool_(),
        "split_hint": s,
        "label_refs": pa.list_(s),
    }
    non_nullable = REQUIRED | {"label_refs", "location_generalized"}
    return pa.schema(
        [pa.field(n, types[n], nullable=n not in non_nullable) for n in FIELD_NAMES],
        metadata={b"marinedata.sample_schema": SCHEMA_VERSION.encode()},
    )


def validate_row(row: SampleRow) -> list[str]:
    """Every violation in ``row`` (empty list = valid). Never raises."""
    errs: list[str] = []
    d = asdict(row)
    for name in REQUIRED:
        v = d[name]
        if v is None or (isinstance(v, str) and not v.strip()):
            errs.append(f"{name} is required")
    if row.image_sha256 and not _HEX64.match(row.image_sha256):
        errs.append("image_sha256 is not 64 lowercase hex")
    if row.upstream_digest is not None and not _HEX64.match(row.upstream_digest):
        errs.append("upstream_digest is not 64 lowercase hex")
    if row.lineage_root_digest is not None and not _HEX64.match(row.lineage_root_digest):
        errs.append("lineage_root_digest is not 64 lowercase hex")
    if isinstance(row.image_bytes, int) and row.image_bytes <= 0:
        errs.append("image_bytes must be > 0")
    if row.sample_id and row.source_id and row.sample_id != f"{row.source_id}/{row.stem}":
        errs.append("sample_id must be '<source_id>/<stem>'")
    if row.stem and ("/" in row.stem or "." in row.stem):
        errs.append("stem must not contain '/' or '.' (WebDataset key rule)")
    for dim in ("width", "height"):
        v = getattr(row, dim)
        if v is not None and v <= 0:
            errs.append(f"{dim} must be > 0 when set")
    if (row.lat is None) != (row.lon is None):
        errs.append("lat and lon must be both set or both null")
    if row.lat is not None and not -90.0 <= row.lat <= 90.0:
        errs.append("lat out of range")
    if row.lon is not None and not -180.0 <= row.lon <= 180.0:
        errs.append("lon out of range")
    if row.gps_precision_m is not None and (row.lat is None or row.gps_precision_m < 0):
        errs.append("gps_precision_m needs a position and must be >= 0")
    if row.depth_m is not None:
        if row.depth_source not in DEPTH_SOURCES:
            errs.append(f"depth_source must be one of {sorted(DEPTH_SOURCES)} when depth_m set")
        derived = depth_zone_for(row.depth_m)
        if row.depth_zone is not None and row.depth_zone != derived:
            errs.append(f"depth_zone {row.depth_zone!r} disagrees with depth_m ({derived!r})")
    elif row.depth_source is not None:
        errs.append("depth_source set without depth_m")
    if row.depth_zone is not None and row.depth_zone not in _ZONE_NAMES:
        errs.append(f"depth_zone must be one of {sorted(_ZONE_NAMES)}")
    if row.platform is not None and row.platform not in PLATFORMS:
        errs.append(f"platform must be one of {sorted(PLATFORMS)}")
    if row.split_hint is not None and row.split_hint not in SPLIT_HINTS:
        errs.append(f"split_hint must be one of {sorted(SPLIT_HINTS)}")
    if row.capture_datetime is not None and row.capture_datetime.tzinfo is None:
        errs.append("capture_datetime must be timezone-aware (UTC)")
    if (row.image_member is None) != (not (row.image_path or "").endswith(".tar")):
        errs.append("image_member must be set exactly when image_path is a shard .tar")
    if any(not r.startswith("labels/") for r in row.label_refs):
        errs.append("label_refs must point under labels/")
    meow = (row.meow_realm, row.meow_province, row.meow_ecoregion)
    if any(v is not None for v in meow) and any(v is None for v in meow):
        errs.append("meow_realm/meow_province/meow_ecoregion must be all-set or all-null")
    if row.location_generalized and row.lat is None:
        errs.append("location_generalized requires lat/lon to be set")
    return errs


def validate_rows(rows: Iterable[SampleRow]) -> None:
    """Raise :class:`SampleSchemaError` listing the first violations (and duplicates)."""
    problems: list[str] = []
    seen: set[str] = set()
    for row in rows:
        problems += [f"{row.sample_id or '?'}: {e}" for e in validate_row(row)]
        if row.sample_id in seen:
            problems.append(f"{row.sample_id}: duplicate sample_id")
        seen.add(row.sample_id)
        if len(problems) > 20:
            break
    if problems:
        raise SampleSchemaError("; ".join(problems[:20]))


def _column(rows: Sequence[SampleRow], name: str) -> list[object]:
    if name == "label_refs":
        return [list(r.label_refs) for r in rows]
    if name == "capture_datetime":
        return [
            r.capture_datetime.astimezone(dt.timezone.utc) if r.capture_datetime else None
            for r in rows
        ]
    return [getattr(r, name) for r in rows]


def to_table(rows: Sequence[SampleRow]):
    """Validated, stem-sorted pyarrow table in the canonical schema."""
    import pyarrow as pa

    ordered = sorted(rows, key=lambda r: r.sample_id)
    validate_rows(ordered)
    schema = arrow_schema()
    return pa.table({n: _column(ordered, n) for n in FIELD_NAMES}, schema=schema)


def write_samples(path: Path, rows: Sequence[SampleRow]) -> None:
    """Write ``metadata.parquet`` (zstd, deterministic: sorted rows, no statistics)."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(rows), path, compression="zstd", write_statistics=False)


def validate_table(table) -> None:
    """Check a read-back table: exact column names/types, then every row."""
    expected = arrow_schema()
    if [f.name for f in table.schema] != list(FIELD_NAMES):
        raise SampleSchemaError(f"columns differ from schema v{SCHEMA_VERSION}")
    for got, want in zip(table.schema, expected, strict=True):
        if not got.type.equals(want.type):
            raise SampleSchemaError(f"{got.name}: type {got.type} != {want.type}")
    validate_rows(row_from_mapping(r) for r in table.to_pylist())


def row_from_mapping(values: Mapping[str, object]) -> SampleRow:
    """Build a row from a mapping (e.g. ``table.to_pylist()`` output)."""
    kw = {n: values.get(n) for n in FIELD_NAMES}
    kw["label_refs"] = tuple(kw.get("label_refs") or ())
    return SampleRow(**kw)  # type: ignore[arg-type]


def licence_filter(table, allowed: Iterable[str]):
    """One-call licence filter: keep rows whose ``license`` is in ``allowed``."""
    import pyarrow as pa
    import pyarrow.compute as pc

    return table.filter(pc.is_in(table["license"], value_set=pa.array(sorted(set(allowed)))))
