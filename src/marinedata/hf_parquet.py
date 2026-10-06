"""Parquet shards with embedded images, laid out for the Hugging Face Hub (WS-D S55).

Pure layout mechanics — no registry, no release, no network. :mod:`marinedata.hf_export`
decides *what* rows exist; this module decides how they land on disk:

- **Embedded files.** An image or mask column is the ``datasets`` ``Image`` feature's own
  Arrow storage, ``struct<bytes: binary, path: string>``, and the feature map is written
  into the schema metadata under ``huggingface`` exactly as ``datasets`` writes it — so
  the Hub reads the files as native Parquet (no conversion, no 5 GB conversion cap) and
  renders the column as thumbnails.
- **Shard size.** Greedy by embedded-file bytes to :data:`SHARD_TARGET_BYTES` (500 MB,
  ``datasets.config.MAX_SHARD_SIZE``). The plan is made before anything is written, so
  every file name carries its final ``-of-MMMMM`` count and the naming is deterministic:
  ``data/<config>/<split>-NNNNN-of-MMMMM.parquet``.
- **Row groups.** Greedy to :data:`ROW_GROUP_TARGET_BYTES` (100 MB uncompressed,
  ``datasets.config.MAX_ROW_GROUP_SIZE``) — the Hub dataset viewer re-converts a Parquet
  file whose row groups exceed 100–300 MB uncompressed. One row group is the only thing
  ever held in memory, so writing is constant-memory in the corpus size.
- **Image row groups** are also capped at :data:`IMAGE_ROW_GROUP_ROWS` rows, and every file
  carries a page index (``write_page_index=True``) so the viewer reads only what it shows.
- **No statistics / dictionaries on blobs.** Min/max statistics on a multi-megabyte
  ``bytes`` column would copy two images into the footer of every row group.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SHARD_TARGET_BYTES = 500 * 1000**2
"""``datasets.config.MAX_SHARD_SIZE = "500MB"`` — the Hub's own ``push_to_hub`` default."""

ROW_GROUP_TARGET_BYTES = 100 * 1000**2
"""``datasets.config.MAX_ROW_GROUP_SIZE = "100MB"``; the viewer's ceiling is 100–300 MB."""

IMAGE_ROW_GROUP_ROWS = 100
"""Row-group row cap for configs with an embedded file: ``datasets`` writes 100 rows per row
group for images, and hub-docs ``datasets-image.md`` recommends exactly that; the viewer
reads whole row groups, so 100 MB groups risk ``TooBigContentError`` (hub-docs
``datasets-data-files-configuration.md``: "use smaller row groups and include a page index")."""

ROW_OVERHEAD_BYTES = 256
"""Allowance per row for the scalar columns and the struct ``path`` in size planning."""

SCALAR_TYPES = ("string", "bool", "int64", "double")
"""``double`` (float64) added for WP-2's pixel-free ``metadata`` config (lat/lon, depth,
quality scores) — no config before it needed a floating column."""
IMAGE = "image"

LIST_STRUCT_KINDS: dict[str, tuple[tuple[str, str], ...]] = {
    "class_map": (
        ("id", "int64"),
        ("label_native", "string"),
        ("taxon_node", "string"),
        ("coarse", "string"),
    ),
    "instances": (
        ("id", "int64"),
        ("label_native", "string"),
        ("taxon_node", "string"),
        ("coarse", "string"),
        ("bbox_xyxy_norm", "list<double>"),
    ),
}
"""MV-1 (masks v1): ``list<struct>`` column kinds. Each field is a scalar type or ``list<double>``;
the value of such a cell is a list of dicts (or ``None``)."""


class HFParquetError(Exception):
    """A shard plan or write that would not produce a valid Hub layout."""


@dataclass(frozen=True)
class EmbeddedBlob:
    """One embedded image cell whose bytes are produced lazily at write time (MV-1): a config
    with several image columns (``image`` + ``mask``), or a cell that is computed (a normalised
    mask PNG) rather than read from a file. ``nbytes`` is only the shard-planning estimate."""

    column: str
    path: str
    nbytes: int
    read: Callable[[], bytes]


@dataclass(frozen=True)
class ExportRow:
    """One Parquet row: scalar ``values`` plus embedded files: ``file`` (into the config's
    first image column) and/or lazily-read ``blobs`` (into the named image columns)."""

    values: dict[str, object]
    file: Path | None = None
    file_path: str | None = None
    """The ``Image`` struct's ``path`` — a readable name, never used to locate bytes."""
    file_bytes: int = 0
    blobs: tuple[EmbeddedBlob, ...] = ()

    @property
    def embedded_bytes(self) -> int:
        return self.file_bytes + sum(b.nbytes for b in self.blobs)

    @property
    def nbytes(self) -> int:
        return self.embedded_bytes + ROW_OVERHEAD_BYTES


@dataclass(frozen=True)
class ConfigSpec:
    """One Hub config (subset): ordered ``(column, type)``; type is a scalar or ``image``."""

    name: str
    columns: tuple[tuple[str, str], ...]
    description: str = ""
    shard_target_bytes: int = SHARD_TARGET_BYTES
    row_group_target_bytes: int = ROW_GROUP_TARGET_BYTES
    image_column: str | None = field(init=False, default=None)
    """The first image column (where ``ExportRow.file`` lands)."""
    image_columns: tuple[str, ...] = field(init=False, default=())

    def __post_init__(self) -> None:
        images = [name for name, kind in self.columns if kind == IMAGE]
        known = (*SCALAR_TYPES, IMAGE, *LIST_STRUCT_KINDS)
        bad = [kind for _, kind in self.columns if kind not in known]
        if bad:
            raise HFParquetError(f"{self.name}: bad column types {bad}")
        object.__setattr__(self, "image_column", images[0] if images else None)
        object.__setattr__(self, "image_columns", tuple(images))


def greedy_chunks(rows: Sequence[ExportRow], target_bytes: int) -> list[list[ExportRow]]:
    """Split ``rows`` in order into chunks of at most ``target_bytes`` (a single row larger
    than the target gets a chunk of its own). Empty input → no chunks."""
    chunks: list[list[ExportRow]] = []
    current: list[ExportRow] = []
    size = 0
    for row in rows:
        if current and size + row.nbytes > target_bytes:
            chunks.append(current)
            current, size = [], 0
        current.append(row)
        size += row.nbytes
    if current:
        chunks.append(current)
    return chunks


def row_groups(spec: ConfigSpec, rows: Sequence[ExportRow]) -> list[list[ExportRow]]:
    """Byte-greedy groups, further cut to :data:`IMAGE_ROW_GROUP_ROWS` rows when the config
    embeds a file."""
    groups = greedy_chunks(rows, spec.row_group_target_bytes)
    if spec.image_column is None:
        return groups
    n = IMAGE_ROW_GROUP_ROWS
    return [g[i : i + n] for g in groups for i in range(0, len(g), n)]


def shard_name(config: str, split: str, index: int, total: int) -> str:
    """``data/<config>/<split>-NNNNN-of-MMMMM.parquet`` — the ``datasets`` convention."""
    if not 0 <= index < total:
        raise HFParquetError(f"shard index {index} outside 0..{total - 1}")
    return f"data/{config}/{split}-{index:05d}-of-{total:05d}.parquet"


def features_metadata(spec: ConfigSpec) -> dict:
    """The ``datasets`` feature map for ``spec``, as stored under schema key ``huggingface``."""
    features: dict[str, dict] = {}
    for name, kind in spec.columns:
        if kind == IMAGE:
            features[name] = {"_type": "Image"}
        elif kind in LIST_STRUCT_KINDS:
            features[name] = {
                "feature": {f: _feature(t) for f, t in LIST_STRUCT_KINDS[kind]},
                "_type": "List",
            }
        else:
            features[name] = {"dtype": kind, "_type": "Value"}
    return {"info": {"features": features}}


def _feature(kind: str) -> dict:
    """``datasets`` feature dict of one struct field (``datasets`` >= 4 ``List`` spelling)."""
    if kind == "list<double>":
        return {"feature": {"dtype": "float64", "_type": "Value"}, "_type": "List"}
    return {"dtype": kind, "_type": "Value"}


def arrow_schema(spec: ConfigSpec):
    import pyarrow as pa

    types = {
        "string": pa.string(),
        "bool": pa.bool_(),
        "int64": pa.int64(),
        "double": pa.float64(),
    }
    fields = []
    for name, kind in spec.columns:
        if kind == IMAGE:
            fields.append(
                pa.field(name, pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
            )
        elif kind in LIST_STRUCT_KINDS:
            inner = [
                (f, pa.list_(pa.float64()) if t == "list<double>" else types[t])
                for f, t in LIST_STRUCT_KINDS[kind]
            ]
            fields.append(pa.field(name, pa.list_(pa.struct(inner))))
        else:
            fields.append(pa.field(name, types[kind]))
    meta = {b"huggingface": json.dumps(features_metadata(spec)).encode("utf-8")}
    return pa.schema(fields, metadata=meta)


def _image_cell(row: ExportRow, column: str, first: str | None) -> dict | None:
    for blob in row.blobs:
        if blob.column == column:
            return {"bytes": blob.read(), "path": blob.path}
    if column == first and row.file is not None:
        return {"bytes": row.file.read_bytes(), "path": row.file_path}
    return None


def _table(spec: ConfigSpec, rows: Sequence[ExportRow], schema):
    import pyarrow as pa

    data: dict[str, list] = {}
    for name, kind in spec.columns:
        if kind == IMAGE:
            data[name] = [_image_cell(r, name, spec.image_column) for r in rows]
        else:
            data[name] = [r.values.get(name) for r in rows]
    return pa.Table.from_pydict(data, schema=schema)


def write_shard(path: Path, spec: ConfigSpec, rows: Sequence[ExportRow]) -> int:
    """Write one shard atomically (``.tmp`` then rename); returns its size in bytes."""
    import pyarrow.parquet as pq

    if not rows:
        raise HFParquetError(f"{path}: refusing to write an empty shard")
    schema = arrow_schema(spec)
    scalars = [name for name, kind in spec.columns if kind in SCALAR_TYPES]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with pq.ParquetWriter(
        tmp,
        schema,
        compression="snappy",
        use_dictionary=scalars,
        write_statistics=scalars,
        write_page_index=True,
    ) as writer:
        for group in row_groups(spec, rows):
            writer.write_table(_table(spec, group, schema))
    tmp.replace(path)
    return path.stat().st_size


@dataclass(frozen=True)
class ShardPlan:
    config: str
    split: str
    name: str
    rows: tuple[ExportRow, ...]

    @property
    def planned_bytes(self) -> int:
        return sum(r.nbytes for r in self.rows)


def plan_config(spec: ConfigSpec, splits: dict[str, Sequence[ExportRow]]) -> list[ShardPlan]:
    """Every shard of one config, named with its final count. Empty splits get no file."""
    plans: list[ShardPlan] = []
    for split, rows in splits.items():
        chunks = greedy_chunks(rows, spec.shard_target_bytes)
        for index, chunk in enumerate(chunks):
            name = shard_name(spec.name, split, index, len(chunks))
            plans.append(ShardPlan(spec.name, split, name, tuple(chunk)))
    return plans


def files_per_folder(paths: Sequence[str]) -> dict[str, int]:
    """Direct entries per repo folder (files and sub-folders), root as ``""``."""
    entries: dict[str, set[str]] = {}
    for rel in paths:
        parts = rel.split("/")
        for depth in range(len(parts)):
            parent = "/".join(parts[:depth])
            entries.setdefault(parent, set()).add(parts[depth])
    return {folder: len(names) for folder, names in sorted(entries.items())}
