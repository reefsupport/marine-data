"""Output contract of the per-row metadata resolvers (PA-P2.1).

A resolver turns what a staged source already carries (file names, upstream tables, EXIF) into
one :class:`ResolvedRow` per image. The rows are written as NEW local files
(``<out>/<source>/<version>/samples.parquet`` + ``coverage.json`` + ``unresolved.tsv``); nothing is
written into a staged tree or the bucket. The columns are the ones ``SampleRow`` (schema v3,
PA-P2.2) carries; staged trees are not rewritten, the exporter joins these files by sha256.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

IDENTITY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_id",
    "source_version",
    "partition",
    "stem",
    "image_path",
    "image_sha256",
)
RESOLVED_COLUMNS: tuple[str, ...] = (
    "lat",
    "lon",
    "date",
    "date_precision",
    "year",
    "depth_m",
    "platform",
    "site_id",
    "quadrat_id",
    "survey_id",
    "region",
    "coord_precision_m",
    "geo_source",
    "date_source",
    "license",
    "license_url",
    "attribution",
    "rights_holder",
    "license_verified_by",
    "license_verified_on",
    "provenance_url",
    "upstream_id",
)
OUTPUT_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + RESOLVED_COLUMNS


@dataclass(frozen=True)
class ResolvedRow:
    """One staged image. ``date`` is a calendar day; a source that only carries a year fills
    ``year`` and leaves ``date`` null. ``coord_precision_m`` is the radius (m) the coordinate is
    good to; ``geo_source`` / ``date_source`` name the table or field the value came from."""

    sample_id: str
    source_id: str
    source_version: str
    partition: str
    stem: str
    image_path: str
    image_sha256: str
    lat: float | None = None
    lon: float | None = None
    date: dt.date | None = None
    date_precision: str | None = None
    year: int | None = None
    depth_m: float | None = None
    platform: str | None = None
    site_id: str | None = None
    quadrat_id: str | None = None
    survey_id: str | None = None
    region: str | None = None
    coord_precision_m: float | None = None
    geo_source: str | None = None
    date_source: str | None = None
    license: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    rights_holder: str | None = None
    license_verified_by: str | None = None
    license_verified_on: dt.date | None = None
    provenance_url: str | None = None
    upstream_id: str | None = None


@dataclass(frozen=True)
class Unresolved:
    """An image a resolver could not fully resolve; ``field`` is ``geo``/``date``/``site``."""

    partition: str
    stem: str
    image_sha256: str
    field: str
    reason: str


assert tuple(f.name for f in fields(ResolvedRow)) == OUTPUT_COLUMNS


def arrow_schema():
    import pyarrow as pa

    s = pa.string()
    types = {"lat": pa.float64(), "lon": pa.float64(), "date": pa.date32(), "year": pa.int16()}
    types |= {"depth_m": pa.float64(), "coord_precision_m": pa.float64()}
    types["license_verified_on"] = pa.date32()
    return pa.schema([pa.field(n, types.get(n, s)) for n in OUTPUT_COLUMNS])


def to_table(rows: Sequence[ResolvedRow]):
    """The rows as a pyarrow table in :data:`OUTPUT_COLUMNS` order (all columns nullable)."""
    import pyarrow as pa

    schema = arrow_schema()
    cols = {
        n: pa.array([getattr(r, n) for r in rows], type=schema.field(n).type)
        for n in OUTPUT_COLUMNS
    }
    return pa.table(cols, schema=schema)


def dedupe_by_sha(rows: Sequence[ResolvedRow]) -> tuple[list[ResolvedRow], int]:
    """One row per ``image_sha256`` (first wins) and the number of rows dropped."""
    seen: set[str] = set()
    kept: list[ResolvedRow] = []
    for r in rows:
        if r.image_sha256 in seen:
            continue
        seen.add(r.image_sha256)
        kept.append(r)
    return kept, len(rows) - len(kept)


def coverage(rows: Sequence[ResolvedRow]) -> dict[str, dict[str, float | int]]:
    """Non-null count and rate per resolved column."""
    n = len(rows)
    out: dict[str, dict[str, float | int]] = {}
    for c in RESOLVED_COLUMNS:
        k = sum(getattr(r, c) is not None for r in rows)
        out[c] = {"non_null": k, "rate": round(k / n, 6) if n else 0.0}
    return out


def unresolved_tsv(source_id: str, version: str, items: Sequence[Unresolved]) -> str:
    head = "source_id\tversion\tpartition\tstem\timage_sha256\tfield\treason\n"
    body = "".join(
        f"{source_id}\t{version}\t{u.partition}\t{u.stem}\t{u.image_sha256}\t{u.field}\t{u.reason}\n"
        for u in items
    )
    return head + body


def write_outputs(
    out_dir: str | Path,
    source_id: str,
    version: str,
    rows: Sequence[ResolvedRow],
    unresolved: Sequence[Unresolved],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write ``samples.parquet``, ``coverage.json`` and ``unresolved.tsv`` under
    ``<out_dir>/<source_id>/<version>/`` and return that directory."""
    import pyarrow.parquet as pq

    kept, dropped = dedupe_by_sha(rows)
    target = Path(out_dir) / source_id / version
    target.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(kept), target / "samples.parquet")
    (target / "unresolved.tsv").write_text(unresolved_tsv(source_id, version, unresolved))
    report = {
        "source_id": source_id,
        "version": version,
        "n_rows": len(kept),
        "n_duplicate_sha256_dropped": dropped,
        "n_unresolved": len(unresolved),
        "columns": coverage(kept),
        **dict(extra or {}),
    }
    (target / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return target
