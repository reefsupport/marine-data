"""Join resolved per-row metadata onto the exported ``metadata`` config (PA-P2.2/P2.4).

``hf_export --metadata-dir <dir>`` reads ``<dir>/<source>/<version>/samples.parquet`` (the resolver
output) and joins it to the ``metadata`` config, and 15 of its columns to the ``points`` config
(PA-P2.6), by ``image_sha256``. The join adds no row: the set of
``image_sha256`` in every config, and every image and mask byte, are identical with and without it
(the P2.3 contract). It is FILL-ONLY: a column the config already has keeps every non-null value and
only gains a value where it was null (the staged geo is null where the resolver found a position);
columns the config lacks are appended. Position columns are never filled for a
``location-sensitive``
source or a row already marked ``location_generalized`` (the build generalised those on purpose).
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from ...hf_parquet import SCALAR_TYPES
from ...metadata_release import METADATA, METADATA_COLUMNS
from .schema import RESOLVED_COLUMNS

_TYPES = dict(METADATA_COLUMNS)
_STR = _TYPES.get("image_sha256", "string")
_DBL = _TYPES.get("lat") or next(t for t in SCALAR_TYPES if "float" in t or "double" in t)
_INT = next((t for t in ("int64", "int32", "int16") if t in SCALAR_TYPES), _DBL)

EXTRA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("lat", _DBL),
    ("lon", _DBL),
    ("depth_m", _DBL),
    ("platform", _STR),
    ("license", _STR),
    ("attribution", _STR),
    ("upstream_id", _STR),
    ("date", _STR),
    ("date_precision", _STR),
    ("year", _INT),
    ("site_id", _STR),
    ("quadrat_id", _STR),
    ("survey_id", _STR),
    ("region", _STR),
    ("coord_precision_m", _DBL),
    ("geo_source", _STR),
    ("date_source", _STR),
    ("license_url", _STR),
    ("rights_holder", _STR),
    ("license_verified_by", _STR),
    ("license_verified_on", _STR),
    ("provenance_url", _STR),
)
"""Candidate columns; those the config already has are only filled, the rest are appended."""
GEO_COLUMNS = frozenset({"lat", "lon", "coord_precision_m", "geo_source"})
POINTS = "points"
POINTS_COLUMNS: tuple[tuple[str, str], ...] = tuple(
    (name, dict(EXTRA_COLUMNS)[name])
    for name in (
        "lat", "lon", "date", "date_precision", "coord_precision_m", "depth_m", "platform",
        "site_id", "quadrat_id", "survey_id", "region",
        "license", "license_url", "attribution", "rights_holder",
    )
)  # fmt: skip
"""The image-level geo, date and licence columns of the HF ``points`` config (PA-P2.6): SampleRow v3
names typed as in :data:`EXTRA_COLUMNS`. ``license`` is the image's; the annotation's own terms stay
in ``ann_license``/``ann_attribution``."""


def _plain(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


def load_resolved(
    metadata_dir: str | Path, source_id: str | None = None
) -> dict[str, dict[str, Any]]:
    """``{image_sha256: {column: value}}`` over every ``*/*/samples.parquet`` (first wins), or
    only those of ``source_id``."""
    import pyarrow.parquet as pq

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(metadata_dir).glob(f"{source_id or '*'}/*/samples.parquet")):
        for rec in pq.read_table(path).to_pylist():
            sha = rec.get("image_sha256")
            if sha and sha not in out:
                out[sha] = {c: _plain(rec.get(c)) for c in RESOLVED_COLUMNS}
    return out


def location_sensitive_sources(registry) -> frozenset[str]:
    return frozenset(
        s.id for s in registry if s.location_sensitive or "location-sensitive" in s.tags
    )


def fill_values(
    values: Mapping[str, Any], found: Mapping[str, Any], keep_geo: bool = False
) -> dict[str, Any]:
    """``values`` with each null or absent column filled from ``found``; no value is replaced."""
    out = dict(values)
    for key, val in found.items():
        if val is None or out.get(key) is not None or (keep_geo and key in GEO_COLUMNS):
            continue
        out[key] = val
    return out


def _resolver_wins(
    values: dict[str, Any], found: Mapping[str, Any], keep_geo: bool
) -> dict[str, Any]:
    """``values`` with the position and the date of ``found`` replacing the row's own staged ones
    (with their precision and source), so a ``metadata`` row never disagrees with the ``points`` of
    the same image (PA-P1.9b). A position needs lat and lon and is left alone when withheld."""
    out = dict(values)
    if not keep_geo and found.get("lat") is not None and found.get("lon") is not None:
        for key in ("lat", "lon", "coord_precision_m", "geo_source"):
            out[key] = found.get(key)
    if found.get("date") is not None:
        for key in ("date", "date_precision", "date_source"):
            out[key] = found.get(key)
    return out


def _widen(
    layout: Mapping[str, Any],
    key: str,
    candidates: tuple[tuple[str, str], ...],
    resolved_by_sha: Mapping[str, Mapping[str, Any]],
    no_geo_sources: frozenset[str],
    *,
    always: bool = False,
    only: frozenset[str] | None = None,
    generalized: frozenset[str] = frozenset(),
    resolver_wins: bool = False,
) -> dict[str, Any]:
    """``layout`` with config ``key`` widened by the ``candidates`` it lacks and each row filled
    from its resolved record by ``image_sha256`` with :func:`fill_values` (a new layout). Without
    resolved rows nothing is done unless ``always`` (the columns are then appended, null); ``only``
    narrows the columns taken from a record; a position is withheld for ``no_geo_sources``, a row
    marked ``location_generalized`` and the ``generalized`` image hashes."""
    entry = layout.get(key)
    if entry is None or not (resolved_by_sha or always):
        return dict(layout)
    spec, splits = entry
    have = {name for name, _ in spec.columns}
    extra = tuple(c for c in candidates if c[0] not in have)
    new_spec = replace(spec, columns=(*spec.columns, *extra))
    blank = {name: None for name, _ in extra}
    new_splits: dict[str, list] = {}
    for split, rows in splits.items():
        merged = []
        for row in rows:
            values = {**blank, **row.values}
            sha = values.get("image_sha256") or values.get("sha256")
            found = resolved_by_sha.get(sha, {})
            if only is not None:
                found = {k: v for k, v in found.items() if k in only}
            keep_geo = (
                values.get("source_id") in no_geo_sources
                or values.get("location_generalized")
                or sha in generalized
            )
            filled = fill_values(values, found, bool(keep_geo))
            if resolver_wins:
                filled = _resolver_wins(filled, found, bool(keep_geo))
            merged.append(replace(row, values=filled))
        new_splits[split] = merged
    return {**layout, key: (new_spec, new_splits)}


def join_resolved(
    layout: Mapping[str, Any],
    resolved_by_sha: Mapping[str, Mapping[str, Any]],
    no_geo_sources: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """``layout`` with the ``metadata`` config widened by the resolved columns (a new layout):
    fill-only, except position and date, which the resolver output sets (it is what ``points``
    carries, see :func:`_resolver_wins`)."""
    return _widen(
        layout, METADATA, EXTRA_COLUMNS, resolved_by_sha, no_geo_sources, resolver_wins=True
    )


def join_points_config(
    layout: Mapping[str, Any],
    resolved_by_sha: Mapping[str, Mapping[str, Any]],
    no_geo_sources: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """``layout`` with the HF ``points`` config widened by :data:`POINTS_COLUMNS` (a new layout).
    The columns exist (null) even with no resolved rows, so the schema does not depend on
    ``--metadata-dir``; with rows each is filled by ``image_sha256`` by :func:`fill_values`, the
    one fill-only rule. A position is also withheld from the points of an image whose ``metadata``
    row is ``location_generalized``. No row is added and no other config is touched."""
    meta = layout.get(METADATA)
    generalized = frozenset(
        r.values["image_sha256"]
        for rows in (meta[1].values() if meta else ())
        for r in rows
        if r.values.get("location_generalized")
    )
    return _widen(
        layout, POINTS, POINTS_COLUMNS, resolved_by_sha, no_geo_sources,
        always=True, only=frozenset(n for n, _ in POINTS_COLUMNS), generalized=generalized,
    )  # fmt: skip


def join_layout(
    layout: Mapping[str, Any],
    resolved_by_sha: Mapping[str, Mapping[str, Any]] | None = None,
    no_geo_sources: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """The ``hf_export`` join step: ``metadata`` widened only with resolved rows (no
    ``--metadata-dir``: unchanged), ``points`` always (null columns without)."""
    resolved = resolved_by_sha or {}
    joined = join_resolved(layout, resolved, no_geo_sources)
    return join_points_config(joined, resolved, no_geo_sources)


def join_points(
    points: str | Path,
    out: str | Path,
    resolved_by_sha: Mapping[str, Mapping[str, Any]],
    no_geo_sources: frozenset[str] = frozenset(),
    batch_size: int = 100_000,
) -> dict[str, int]:
    """A points parquet widened by the resolved columns (fill-only, by ``image_sha256``), streamed
    to ``out``. The canonical ``points`` file is schema-checked and stays as built. Returns the
    counts ``points``, ``lat_lon``, ``date`` and ``lat_lon_date`` (points carrying each)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    src = pq.ParquetFile(points)
    kind = dict.fromkeys(("lat", "lon", "depth_m", "coord_precision_m"), pa.float64())
    fields = [
        pa.field(c, kind.get(c, pa.int16() if c == "year" else pa.string()))
        for c in RESOLVED_COLUMNS
    ]
    schema = pa.schema([*src.schema_arrow, *fields])
    blank = dict.fromkeys(RESOLVED_COLUMNS)
    n = Counter(points=0, lat_lon=0, date=0, lat_lon_date=0)
    with pq.ParquetWriter(out, schema, compression="zstd") as writer:
        for batch in src.iter_batches(batch_size=batch_size):
            rows = [
                fill_values(
                    {**blank, **r},
                    resolved_by_sha.get(r["image_sha256"], {}),
                    r["source_id"] in no_geo_sources,
                )
                for r in batch.to_pylist()
            ]
            for r in rows:
                geo, day = r["lat"] is not None and r["lon"] is not None, r["date"] is not None
                n.update(points=1, lat_lon=geo, date=day, lat_lon_date=geo and day)
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    return dict(n)
