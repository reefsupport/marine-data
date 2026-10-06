"""Sparse geometry and metadata tables — parquet writers, explicit schema (D1 §2).

Four tables, all with rows sorted before write and ``write_statistics=False``: two
determinism requirements from the same cause. Parquet dictionary-encodes string
columns in first-seen order, so *unsorted* input rows would make the encoded bytes
depend on insertion order even though the decoded values are identical; sorting first
means the array handed to pyarrow is always in the same canonical order regardless of
how the caller assembled it. Embedded min/max statistics are dropped for the same
reason — they are computed from row order too.

``pyarrow`` is an optional dependency (the ``ingest`` extra) and is imported lazily, so
importing this module never requires it — only calling into it does, with an error
naming the extra to install, matching the existing ``integrations/`` pattern.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .checksums import file_digest


def _require_pyarrow():
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pyarrow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return pa


@dataclass(frozen=True)
class StagedImage:
    """One staged image — a ``metadata.parquet`` row.

    D1 §5 puts this dataclass in ``ingest.py``; that file is out of I1's scope (it is
    I2's), so it lives here instead. I2 imports it as ``marinedata.tables.StagedImage``.
    """

    stem: str
    partition: str
    upstream_path: str
    upstream_split: str | None
    width: int
    height: int
    split_group: str | None = None
    """The source's registry ``split_group`` rule applied to this row (D-group, WS-D
    step 3). ``None`` for ingest paths that have not been wired to compute it yet —
    the column stays nullable rather than every caller needing a placeholder value."""
    license: str | None = None
    """WP-R2: this image's own licence string, for ``licence_per_row`` sources (FathomNet, iNat,
    ...). The parquet column is written only when some row carries one."""


@dataclass(frozen=True)
class PointRow:
    """One ``labels/points.parquet`` row — pixel coords, 0-based, top-left origin.

    ``label_id``, ``form`` and ``region`` (D3a1) are nullable string extras for
    sources whose point annotations carry a stable label id, a sparse growth-form
    tag, or a biogeographic region alongside the label name — ``None`` for every
    caller that does not have one. No timestamp column, ever.
    """

    stem: str
    partition: str
    row: int
    col: int
    label: str
    schema_id: str
    label_id: str | None = None
    form: str | None = None
    region: str | None = None


@dataclass(frozen=True)
class ImageLabelRow:
    """One ``labels/image_labels.parquet`` row — a whole-image annotation (Roboflow
    image classification, WS-D S23), not a localized point or box.

    One row per positive class: a single-label source (v3i's folder-per-class layout)
    emits exactly one row per ``(partition, stem)``; a multi-label source (a Roboflow
    one-hot ``_classes.csv`` export) emits one row per column whose value is truthy for
    that image. There is deliberately no "negative" row for a 0 column — absence from
    this table already means "not this class", the same convention
    :func:`write_points_table` uses for an unannotated image.

    ``confidence`` is a nullable float extra for a source whose export carries a
    per-label score rather than a bare 0/1 flag — ``None`` for every caller that does
    not have one, same shape as :class:`PointRow`'s nullable extras.
    """

    stem: str
    partition: str
    label: str
    schema_id: str
    confidence: float | None = None


@dataclass(frozen=True)
class BoxRow:
    """One ``labels/boxes.parquet`` row — pixel coords, 0-based, top-left origin."""

    stem: str
    partition: str
    x: int
    y: int
    w: int
    h: int
    label: str
    schema_id: str


_WRITE_KWARGS = {"compression": "zstd", "version": "2.6", "write_statistics": False}


def write_metadata_table(path: Path, rows: Sequence[StagedImage]) -> str:
    """Write ``metadata.parquet``: one row per staged image. Returns the file's sha256."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("stem", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("upstream_path", pa.string(), nullable=False),
            pa.field("upstream_split", pa.string(), nullable=True),
            pa.field("width", pa.int32(), nullable=False),
            pa.field("height", pa.int32(), nullable=False),
            pa.field("split_group", pa.string(), nullable=True),
        ]
    )
    ordered = sorted(rows, key=lambda r: (r.partition, r.stem))
    columns = {
        "stem": [r.stem for r in ordered],
        "partition": [r.partition for r in ordered],
        "upstream_path": [r.upstream_path for r in ordered],
        "upstream_split": [r.upstream_split for r in ordered],
        "width": [r.width for r in ordered],
        "height": [r.height for r in ordered],
        "split_group": [r.split_group for r in ordered],
    }
    if any(r.license for r in ordered):
        schema = schema.append(pa.field("license", pa.string(), nullable=True))
        columns["license"] = [r.license for r in ordered]
    table = pa.table(columns, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **_WRITE_KWARGS)
    return file_digest(path)


def write_points_table(path: Path, rows: Sequence[PointRow]) -> str:
    """Write ``labels/points.parquet``. Returns the file's sha256."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("stem", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("row", pa.int32(), nullable=False),
            pa.field("col", pa.int32(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            pa.field("label_id", pa.string(), nullable=True),
            pa.field("form", pa.string(), nullable=True),
            pa.field("region", pa.string(), nullable=True),
        ]
    )
    ordered = sorted(rows, key=lambda r: (r.partition, r.stem, r.row, r.col))
    table = pa.table(
        {
            "stem": [r.stem for r in ordered],
            "partition": [r.partition for r in ordered],
            "row": [r.row for r in ordered],
            "col": [r.col for r in ordered],
            "label": [r.label for r in ordered],
            "schema_id": [r.schema_id for r in ordered],
            "label_id": [r.label_id for r in ordered],
            "form": [r.form for r in ordered],
            "region": [r.region for r in ordered],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **_WRITE_KWARGS)
    return file_digest(path)


def write_image_labels_table(path: Path, rows: Sequence[ImageLabelRow]) -> str:
    """Write ``labels/image_labels.parquet``. Returns the file's sha256."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("stem", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
            pa.field("confidence", pa.float32(), nullable=True),
        ]
    )
    ordered = sorted(rows, key=lambda r: (r.partition, r.stem, r.label))
    table = pa.table(
        {
            "stem": [r.stem for r in ordered],
            "partition": [r.partition for r in ordered],
            "label": [r.label for r in ordered],
            "schema_id": [r.schema_id for r in ordered],
            "confidence": [r.confidence for r in ordered],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **_WRITE_KWARGS)
    return file_digest(path)


def write_boxes_table(path: Path, rows: Sequence[BoxRow]) -> str:
    """Write ``labels/boxes.parquet``. Returns the file's sha256."""
    pa = _require_pyarrow()
    import pyarrow.parquet as pq

    schema = pa.schema(
        [
            pa.field("stem", pa.string(), nullable=False),
            pa.field("partition", pa.string(), nullable=False),
            pa.field("x", pa.int32(), nullable=False),
            pa.field("y", pa.int32(), nullable=False),
            pa.field("w", pa.int32(), nullable=False),
            pa.field("h", pa.int32(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("schema_id", pa.string(), nullable=False),
        ]
    )
    ordered = sorted(rows, key=lambda r: (r.partition, r.stem, r.x, r.y))
    table = pa.table(
        {
            "stem": [r.stem for r in ordered],
            "partition": [r.partition for r in ordered],
            "x": [r.x for r in ordered],
            "y": [r.y for r in ordered],
            "w": [r.w for r in ordered],
            "h": [r.h for r in ordered],
            "label": [r.label for r in ordered],
            "schema_id": [r.schema_id for r in ordered],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, **_WRITE_KWARGS)
    return file_digest(path)
