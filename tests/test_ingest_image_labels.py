"""Tests for :mod:`marinedata.ingest_image_labels` (WS-D S23).

``convert_bespoke_metadata`` fixtures mirror v3i's real on-disk shape (a bespoke
``path``/``class``/``upstream_split``/``width``/``height``/``split_group``
``metadata.parquet``, no ``partition``/``stem``) — the exact tree
:func:`marinedata.release.enumerate_release_rows` raised ``KeyError`` on before this
converter existed (S12b). ``convert_classes_csv`` fixtures use real minimal PNGs
(``Image.open`` reads the size for real, unlike the byte-stub ``_touch_image`` other
loader tests use).
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.ingest_image_labels import (
    convert_bespoke_metadata,
    convert_classes_csv,
    write_staged_image_label_tree,
)


def _bespoke_tree(root: Path) -> Path:
    (root / "images" / "default").mkdir(parents=True)
    for stem in ("a", "b"):
        (root / "images" / "default" / f"{stem}.jpg").write_bytes(b"JPEGDATA" + stem.encode())
    table = pa.table(
        {
            "path": ["images/default/a.jpg", "images/default/b.jpg"],
            "class": ["Healthy", "Unhealthy"],
            "upstream_split": ["train", "valid"],
            "width": [640, 640],
            "height": [640, 640],
            "split_group": ["roboflow-bleaching/a", "roboflow-bleaching/b"],
        }
    )
    pq.write_table(table, root / "metadata.parquet")
    return root


def test_convert_bespoke_metadata_derives_partition_and_stem(tmp_path: Path) -> None:
    """v3i's bespoke schema has no ``partition``/``stem`` — this is what fixes the
    S12b ``KeyError``: they are re-derived from ``path``, and ``upstream_split`` is
    dropped (R3 Q10: never honoured downstream)."""
    root = _bespoke_tree(tmp_path / "old")
    converted = convert_bespoke_metadata(root, schema_id="roboflow-bleaching-native")

    by_stem = {img.stem: img for img in converted.images}
    assert set(by_stem) == {"a", "b"}
    assert by_stem["a"].partition == "default"
    assert by_stem["a"].upstream_split is None
    assert by_stem["a"].split_group == "roboflow-bleaching/a"

    labels_by_stem = {row.stem: row.label for row in converted.labels}
    assert labels_by_stem == {"a": "Healthy", "b": "Unhealthy"}
    assert set(converted.copy_map) == {"images/default/a.jpg", "images/default/b.jpg"}


def test_write_staged_image_label_tree_produces_readable_staged_tree(tmp_path: Path) -> None:
    root = _bespoke_tree(tmp_path / "old")
    converted = convert_bespoke_metadata(root, schema_id="roboflow-bleaching-native")
    version_root = tmp_path / "new"

    manifest = write_staged_image_label_tree(converted, version_root)

    assert (version_root / "images" / "default" / "a.jpg").is_file()
    assert (version_root / "metadata.parquet").is_file()
    assert (version_root / "labels" / "image_labels.parquet").is_file()
    assert (version_root / "CHECKSUMS.sha256").is_file()
    assert manifest.files == 4  # 2 images + metadata.parquet + image_labels.parquet

    meta = pq.read_table(version_root / "metadata.parquet").to_pylist()
    assert {r["stem"] for r in meta} == {"a", "b"}
    assert {r["partition"] for r in meta} == {"default"}


def test_write_staged_image_label_tree_rejects_zero_byte_image(tmp_path: Path) -> None:
    root = _bespoke_tree(tmp_path / "old")
    (root / "images" / "default" / "a.jpg").write_bytes(b"")
    converted = convert_bespoke_metadata(root, schema_id="roboflow-bleaching-native")

    with pytest.raises(ValueError, match="zero-byte"):
        write_staged_image_label_tree(converted, tmp_path / "new")


def _classes_csv_tree(root: Path) -> Path:
    from PIL import Image

    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4)).save(root / "img0.jpg")
    Image.new("RGB", (4, 4)).save(root / "img1.jpg")
    (root / "_classes.csv").write_text(
        "filename,Healthy,Bleached\nimg0.jpg,1,0\nimg1.jpg,0,1\n", encoding="utf-8"
    )
    return root


def test_convert_classes_csv_one_row_per_positive_class(tmp_path: Path) -> None:
    root = _classes_csv_tree(tmp_path / "csv")
    converted = convert_classes_csv(root, schema_id="roboflow-bleaching-native")

    labels_by_stem: dict[str, list[str]] = {}
    for row in converted.labels:
        labels_by_stem.setdefault(row.stem, []).append(row.label)
    assert labels_by_stem == {"img0": ["Healthy"], "img1": ["Bleached"]}

    by_stem = {img.stem: img for img in converted.images}
    assert by_stem["img0"].width == 4
    assert by_stem["img0"].height == 4
    assert by_stem["img0"].split_group is None
