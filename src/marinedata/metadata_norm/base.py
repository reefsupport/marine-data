"""Per-source metadata normaliser interface (WP-U14a).

``normalise(source_id, version, staged, ctx) -> pa.Table`` returns one row per staged image
with all :data:`marinedata.sample_schema.FIELD_NAMES` columns (null when unknown) plus
``licence_class`` and ``provenance`` (``map<string,string>``: field -> where it came from).
Normalisers are pure: all I/O happens in :mod:`.stage`, which builds the :class:`NormContext`.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..sample_schema import FIELD_NAMES, arrow_schema

EXTRA_COLUMNS: tuple[str, ...] = ("licence_class", "provenance")
OUTPUT_COLUMNS: tuple[str, ...] = (*FIELD_NAMES, *EXTRA_COLUMNS)
Staged = Any  # pyarrow.Table | Sequence[Mapping] | None


@dataclass(frozen=True)
class NormContext:
    """Everything a normaliser may read besides the staged table. Never mutated."""

    registry_licence: str = ""
    """The registry/ingest-spec licence string (a fallback, never a per-row licence)."""
    attribution: str = ""
    citation: str = ""
    homepage: str = ""
    source_class: str = "unknown"
    per_row: bool = False
    checksums: Mapping[str, str] = field(default_factory=dict)
    """``CHECKSUMS.sha256``: relative path -> sha256."""
    ingest: Mapping[str, Any] = field(default_factory=dict)
    """``INGEST.json`` (fetch_date, version, upstream list, ...)."""
    labels: Mapping[str, Any] = field(default_factory=dict)
    """Per-source extras, keyed by stem or photo id (e.g. FathomNet per-image JSON)."""
    version: str = ""
    fetch_date_fallback: dt.date | None = None
    """Used when ``INGEST.json`` is missing (registry ``retrieved``/``fetched``, else the
    earliest object ``LastModified``); ``fetch_date_origin`` names which."""
    fetch_date_origin: str = ""
    default_platform: str = ""
    default_habitat: str = ""
    default_instrument: str = ""
    split_rule: Any = None
    """The source's registry ``SplitGroupRule`` (``None`` = fallback grouping only)."""
    events: Mapping[str, Any] = field(default_factory=dict)
    """Cached MERMAID sample events (see :func:`.geo.mermaid_fields`)."""
    meow: Sequence[Any] = ()
    """MEOW polygons (``geo_meow.load_meow_polygons``); empty = no ecoregion lookup."""


class Normaliser(Protocol):
    def __call__(self, source_id: str, version: str, staged: Staged, ctx: NormContext) -> Any: ...


def staged_rows(staged: Staged) -> list[dict[str, Any]]:
    """Rows of ``staged`` as dicts (a pyarrow table, a list of mappings, or ``None``)."""
    if staged is None:
        return []
    if hasattr(staged, "to_pylist"):
        return list(staged.to_pylist())
    return [dict(r) for r in staged]


def _provenance_type():
    import pyarrow as pa

    return pa.map_(pa.string(), pa.string())


def to_table(rows: Sequence[Mapping[str, Any]]):
    """``rows`` (dicts with SampleRow fields + ``licence_class`` + ``provenance``) as a table
    in :data:`OUTPUT_COLUMNS` order; absent keys become null."""
    import pyarrow as pa

    schema = arrow_schema()
    base = {f.name: f for f in schema}
    fields = [pa.field(n, base[n].type) for n in FIELD_NAMES]  # unknown -> null: all nullable
    fields += [pa.field("licence_class", pa.string()), pa.field("provenance", _provenance_type())]
    cols = {}
    for f in fields:
        vals = [r.get(f.name) for r in rows]
        if f.name == "provenance":
            vals = [list((v or {}).items()) for v in vals]
        elif f.name == "label_refs":
            vals = [list(v) if v is not None else [] for v in vals]
        elif f.name == "location_generalized":
            vals = [bool(v) for v in vals]
        cols[f.name] = pa.array(vals, type=f.type)
    return pa.table(cols, schema=pa.schema(fields))


def parse_date(value: Any) -> dt.date | None:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None
