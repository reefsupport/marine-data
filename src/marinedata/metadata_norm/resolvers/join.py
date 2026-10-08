"""Join resolved per-row metadata onto the exported ``metadata`` config (PA-P2.2/P2.4).

``hf_export --metadata-dir <dir>`` reads ``<dir>/<source>/<version>/samples.parquet`` (the resolver
output) and joins it to the ``metadata`` config by ``image_sha256``. The join adds no row: the set
of
``image_sha256`` in every config, and every image and mask byte, are identical with and without it
(the P2.3 contract). It is FILL-ONLY: a column the config already has keeps every non-null value and
only gains a value where it was null (the staged geo is null where the resolver found a position);
columns the config lacks are appended. Position columns are never filled for a
``location-sensitive``
source or a row already marked ``location_generalized`` (the build generalised those on purpose).
"""

from __future__ import annotations

import datetime as dt
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


def _plain(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


def load_resolved(metadata_dir: str | Path) -> dict[str, dict[str, Any]]:
    """``{image_sha256: {column: value}}`` over every ``*/*/samples.parquet`` (first wins)."""
    import pyarrow.parquet as pq

    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(metadata_dir).glob("*/*/samples.parquet")):
        for rec in pq.read_table(path).to_pylist():
            sha = rec.get("image_sha256")
            if sha and sha not in out:
                out[sha] = {c: _plain(rec.get(c)) for c in RESOLVED_COLUMNS}
    return out


def location_sensitive_sources(registry) -> frozenset[str]:
    return frozenset(
        s.id for s in registry if s.location_sensitive or "location-sensitive" in s.tags
    )


def join_resolved(
    layout: Mapping[str, Any],
    resolved_by_sha: Mapping[str, Mapping[str, Any]],
    no_geo_sources: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """``layout`` with the ``metadata`` config widened by the resolved columns (a new layout)."""
    entry = layout.get(METADATA)
    if entry is None or not resolved_by_sha:
        return dict(layout)
    spec, splits = entry
    have = {name for name, _ in spec.columns}
    extra = tuple(c for c in EXTRA_COLUMNS if c[0] not in have)
    new_spec = replace(spec, columns=(*spec.columns, *extra))
    blank = {name: None for name, _ in extra}
    new_splits: dict[str, list] = {}
    for split, rows in splits.items():
        merged = []
        for row in rows:
            values = {**blank, **row.values}
            found = resolved_by_sha.get(values.get("image_sha256"), {})
            keep_geo = values.get("source_id") in no_geo_sources or values.get(
                "location_generalized"
            )
            for key, val in found.items():
                if val is None or values.get(key) is not None:
                    continue
                if keep_geo and key in GEO_COLUMNS:
                    continue
                values[key] = val
            merged.append(replace(row, values=values))
        new_splits[split] = merged
    return {**layout, METADATA: (new_spec, new_splits)}
