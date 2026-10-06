"""HK-4c driver: local stage mirror -> boxes annotation parquet."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

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
