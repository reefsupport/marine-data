"""HK-4c driver: local stage mirror -> boxes annotation parquet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.registry import Registry, _default_root

SPEC = importlib.util.spec_from_file_location(
    "hk_fish_boxes", Path(__file__).parents[1] / "scripts" / "hk_fish_boxes.py"
)
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)


def test_driver_writes_boxes_parquet_from_a_local_stage(tmp_path):
    stage = tmp_path / "stage" / "uiis"
    (stage / "labels" / "files").mkdir(parents=True)
    img = {"id": 1, "file_name": "a.jpg", "width": 100, "height": 50}
    ann = {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 5, 20, 10], "iscrowd": 0,
           "segmentation": [[0]]}  # fmt: skip
    doc = {"images": [img], "annotations": [ann], "categories": [{"id": 1, "name": "fish"}]}
    (stage / "labels/files/annotations_train.json").write_text(json.dumps(doc))
    (stage / "labels/files/annotations_val.json").write_text(json.dumps({**doc, "images": []}))
    (stage / "CHECKSUMS.sha256").write_text(f"{'a' * 64}  images/a.jpg\n")
    stats = drv.write_source_boxes(
        tmp_path / "stage", tmp_path / "d", "uiis", Registry.load(_default_root())
    )
    assert (stats["images"], stats["boxes"], stats["classes"]) == (1, 1, 1)
    out = tmp_path / "d/_annotations/boxes/uiis/rev-44ca5db46599.parquet"
    assert pq.read_table(out).num_rows == 1


CORAL_POINTS_COLUMNS = [
    "image_sha256", "image", "source_id", "split_group", "split", "license", "licence_class",
    "attribution", "upstream_id", "upstream_url", "lat", "lon", "depth_m", "meow_realm",
    "habitat", "width", "height", "points", "boxes",
]  # fmt: skip


def _png(color: int) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 4), (color, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _fake_hf_and_boxes(tmp_path: Path) -> tuple[Path, Path]:
    hf, data = tmp_path / "hf", tmp_path / "d"
    (hf / "data/images").mkdir(parents=True)
    (hf / "data/metadata").mkdir(parents=True)
    (data / "_annotations/boxes/uiis").mkdir(parents=True)
    rows = [("a" * 64, "uiis", "train"), ("b" * 64, "uiis", "test"), ("c" * 64, "uiis", "train")]
    images = [
        {"image_sha256": s, "image": {"bytes": _png(i * 40), "path": f"{s[:4]}.png"},
         "source_id": sid, "split_group": f"uiis/{s[:4]}"}
        for i, (s, sid, _) in enumerate(rows)
    ]  # fmt: skip
    meta = [
        {"image_sha256": s, "source_id": sid, "split": sp, "split_group": f"uiis/{s[:4]}",
         "license": "Apache-2.0", "licence_class": "open", "attribution": "authors",
         "upstream_id": s[:4], "upstream_url": None, "lat": None, "lon": None, "depth_m": None,
         "meow_realm": None, "habitat": None}
        for s, sid, sp in rows
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(images), hf / "data/images/train-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(meta), hf / "data/metadata/train-00000-of-00001.parquet")
    box = {"source_id": "uiis", "ann_id": "uiis:0", "label_native": "fish", "is_crowd": False,
           "x_min": 0.1, "y_min": 0.2, "x_max": 0.5, "y_max": 0.9, "x_min_px": 1, "y_min_px": 1,
           "x_max_px": 4, "y_max_px": 3, "confidence": None}  # fmt: skip
    boxes = [
        {**box, "image_sha256": "a" * 64},
        {**box, "image_sha256": "a" * 64, "ann_id": "uiis:1", "label_native": "diver"},
        {**box, "image_sha256": "b" * 64, "ann_id": "uiis:2"},
    ]  # "c" has no box
    pq.write_table(pa.Table.from_pylist(boxes), data / "_annotations/boxes/uiis/rev-1.parquet")
    return hf, data


def test_package_joins_boxes_into_the_coral_points_layout(tmp_path):
    hf, data = _fake_hf_and_boxes(tmp_path)
    out = tmp_path / "out"
    stats = drv.package(hf, data, out)
    assert (stats["images"], stats["boxes"], stats["images_without_boxes"]) == (2, 3, 1)
    assert stats["classes"] == ["diver", "fish"]
    shard = pq.read_table(out / "data/train-00000-of-00001.parquet")
    assert shard.column_names == CORAL_POINTS_COLUMNS
    assert str(shard.schema.field("boxes").type) == (
        "list<element: struct<x_min: float, y_min: float, x_max: float, y_max: float, "
        "label: string>>"
    )
    first = shard.to_pylist()[0]
    assert (first["split"], first["split_group"], first["license"]) == (
        "train",
        "uiis/aaaa",
        "Apache-2.0",
    )
    assert (
        first["points"] == []
        and len(first["boxes"]) == 2
        and (first["width"], first["height"]) == (8, 4)
    )
    assert first["boxes"][0]["label"] == "fish" and abs(first["boxes"][0]["x_max"] - 0.5) < 1e-6
    ann = pq.read_table(out / "annotations/boxes.parquet").to_pylist()
    assert len(ann) == 3 and {a["split"] for a in ann} == {"train", "test"}
    assert ann[0]["license"] == "Apache-2.0" and ann[0]["label"] == "fish"
    assert json.loads((out / "build_stats.json").read_text())["images"] == 2
