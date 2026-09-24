"""Convert an already-fetched Roboflow image-classification tree into the staged-tree
image-label convention (WS-D S23): ``images/<partition>/<stem>.<ext>``,
``metadata.parquet`` (:class:`~marinedata.tables.StagedImage`) and
``labels/image_labels.parquet`` (:class:`~marinedata.tables.ImageLabelRow`).

Two source shapes, one per converter function, neither touching the network — both
read a tree a fetch step already put on disk, the same split between "get the bytes"
and "turn them into the shared schema" :mod:`marinedata.ingest_s3` uses for points:

- :func:`convert_bespoke_metadata` — the S12b-era bespoke ``metadata.parquet``
  (``path``/``class``/``upstream_split``/``width``/``height``/``split_group``, no
  ``partition``/``stem``) this repo staged for v3i before ``StagedImage`` existed.
  Re-deriving ``partition``/``stem`` from it is exactly what fixes the S12b
  ``KeyError`` the release enumerator raised on a genuine ``staged-tree`` source.
- :func:`convert_classes_csv` — a Roboflow multiclass export's one-hot ``_classes.csv``
  sidecar (one column per class, 0/1 per image), the shape the other four sets (v6i,
  v1-yolov8s, v13i, v2i) ship in. Not wired into any registry entry yet (S24); this
  converter exists so that wiring is a registry change, not new code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .tables import ImageLabelRow, StagedImage, _require_pyarrow


@dataclass(frozen=True)
class ConvertedTree:
    """One converter's output — ready for :func:`write_staged_image_label_tree`.

    ``copy_map`` is the new-relative-path -> source-``Path`` mapping the writer copies
    from; kept separate from ``images`` (whose ``upstream_path`` is the *old* relative
    path, for provenance) so the writer never has to re-derive a destination path from
    a ``StagedImage`` row.
    """

    images: tuple[StagedImage, ...]
    labels: tuple[ImageLabelRow, ...]
    copy_map: dict[str, Path]


def convert_bespoke_metadata(
    old_root: Path, *, schema_id: str, partition: str = "default"
) -> ConvertedTree:
    """Re-derive rows from v3i's real on-disk shape: a bespoke ``metadata.parquet``
    with ``path``/``class``/``upstream_split``/``width``/``height``/``split_group``
    columns and images already sitting under ``old_root`` at each row's ``path``.

    ``upstream_split`` is read off the old row only to preserve provenance in
    ``upstream_path``'s sibling data — the produced :class:`StagedImage` always sets
    ``upstream_split=None`` (R3 Q10: 492/886 v3i groups straddle Roboflow's own
    train/valid/test, so it must never be honoured downstream).
    """
    _require_pyarrow()
    import pyarrow.parquet as pq

    table = pq.read_table(old_root / "metadata.parquet")
    images: list[StagedImage] = []
    labels: list[ImageLabelRow] = []
    copy_map: dict[str, Path] = {}
    for record in table.to_pylist():
        rel_path = record["path"]
        stem = Path(rel_path).stem
        ext = Path(rel_path).suffix
        new_rel = f"images/{partition}/{stem}{ext}"
        if new_rel in copy_map:
            raise ValueError(
                f"duplicate stem {stem!r} in partition {partition!r} staging {rel_path!r}"
            )
        copy_map[new_rel] = old_root / rel_path
        images.append(
            StagedImage(
                stem=stem,
                partition=partition,
                upstream_path=rel_path,
                upstream_split=None,
                width=record["width"],
                height=record["height"],
                split_group=record["split_group"],
            )
        )
        labels.append(
            ImageLabelRow(
                stem=stem, partition=partition, label=record["class"], schema_id=schema_id
            )
        )
    return ConvertedTree(images=tuple(images), labels=tuple(labels), copy_map=copy_map)


def convert_classes_csv(
    root: Path,
    *,
    schema_id: str,
    partition: str = "default",
    csv_name: str = "_classes.csv",
) -> ConvertedTree:
    """Re-derive rows from a Roboflow one-hot ``_classes.csv`` sidecar: first column is
    the filename, every other column is a class name whose value is truthy (anything
    but empty or ``"0"``) when that image carries the class — zero, one, or several
    per image. No width/height column in the CSV, so those are read off each image.
    """
    from PIL import Image

    with (root / csv_name).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"{root / csv_name}: no header row")
        filename_col, *class_cols = fieldnames

        images: list[StagedImage] = []
        labels: list[ImageLabelRow] = []
        copy_map: dict[str, Path] = {}
        for record in reader:
            filename = record[filename_col].strip()
            stem = Path(filename).stem
            ext = Path(filename).suffix
            new_rel = f"images/{partition}/{stem}{ext}"
            if new_rel in copy_map:
                raise ValueError(
                    f"duplicate stem {stem!r} in partition {partition!r} staging {filename!r}"
                )
            src_path = root / filename
            copy_map[new_rel] = src_path
            with Image.open(src_path) as im:
                width, height = im.size
            images.append(
                StagedImage(
                    stem=stem,
                    partition=partition,
                    upstream_path=filename,
                    upstream_split=None,
                    width=width,
                    height=height,
                    split_group=None,
                )
            )
            for label in (c.strip() for c in class_cols):
                value = str(record[label]).strip()
                if value and value != "0":
                    labels.append(
                        ImageLabelRow(
                            stem=stem, partition=partition, label=label, schema_id=schema_id
                        )
                    )
    return ConvertedTree(images=tuple(images), labels=tuple(labels), copy_map=copy_map)


def write_staged_image_label_tree(converted: ConvertedTree, version_root: Path):
    """Copy every image into its new path, write ``metadata.parquet`` +
    ``labels/image_labels.parquet``, then checksum the finished tree.

    Refuses outright if any source image is zero bytes — a 0-byte file would copy
    cleanly, hash cleanly, and pass every later shape check while carrying no picture
    at all, so this is checked before any bytes are copied, not discovered later by a
    downstream image-loader crash.
    """
    import os
    import shutil

    from . import checksums
    from .tables import write_image_labels_table, write_metadata_table

    for src_path in converted.copy_map.values():
        if src_path.stat().st_size == 0:
            raise ValueError(f"{src_path}: zero-byte source image, refusing to stage")

    for new_rel, src_path in converted.copy_map.items():
        dest = version_root / new_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            continue
        try:
            os.link(src_path, dest)
        except OSError:
            shutil.copy2(src_path, dest)

    write_metadata_table(version_root / "metadata.parquet", converted.images)
    write_image_labels_table(version_root / "labels" / "image_labels.parquet", converted.labels)
    return checksums.write_checksums(version_root)
