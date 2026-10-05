"""WP-U12: COCO instance masks (usis10k, uiis, uiis10k) into the unified masks table."""

from __future__ import annotations

import json

import pytest

from marinedata.annotation_schema import validate_rows
from marinedata.registry import Registry
from marinedata.task_layers.points_table import resolver_for
from marinedata.task_layers.sources import masks_instance as mi

SPEC = mi.INSTANCE_SOURCES["usis10k"]
SHAS = {("default", "USIS10K_zip_USIS10K_val_val_00001"): "a" * 64}
DOC = {
    "categories": [{"id": 2, "name": "fish"}, {"id": 4, "name": "aquatic plants"}],
    "images": [
        {"id": 0, "file_name": "val_00001.jpg", "width": 640, "height": 480},
        {"id": 1, "file_name": "val_99999.jpg", "width": 640, "height": 480},
    ],
    "annotations": [
        {"id": 7, "image_id": 0, "category_id": 2, "iscrowd": 0, "bbox": [1, 2, 3, 4],
         "segmentation": [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0]], "area": 8.0},
        {"id": 8, "image_id": 0, "category_id": 4, "iscrowd": 1,
         "segmentation": {"size": [480, 640], "counts": "abc"}},
        {"id": 9, "image_id": 0, "category_id": 2, "segmentation": []},
        {"id": 10, "image_id": 0, "category_id": 99, "segmentation": [[1, 1, 2, 2, 3, 3]]},
        {"id": 11, "image_id": 1, "category_id": 2, "segmentation": [[1, 1, 2, 2, 3, 3]]},
    ],
}  # fmt: skip


@pytest.fixture(scope="module")
def reg():
    return Registry.load()


@pytest.fixture(scope="module")
def built(reg):
    resolve = resolver_for(reg, SPEC.crosswalk_id)
    return mi.rows_from_doc(SPEC, DOC, "val", SHAS, resolve, reg)


def test_rows_validate_and_tally(built):
    rows, tally = built
    assert [r["attrs"] and json.loads(r["attrs"])["instance_id"] for r in rows] == ["7", "8"]
    assert dict(tally) == {"no_geometry": 1, "no_category": 1, "orphan_image": 1}
    validate_rows("masks", rows)


def test_polygon_rle_and_resolution(built):
    poly, rle = built[0]
    assert poly["mask_kind"] == "instance" and poly["mask_ref"] is None
    assert json.loads(poly["polygon"]) == [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0]] and poly["rle"] is None
    assert (
        poly["label_native"] == "fish"
        and poly["taxon_node_id"]
        and poly["match_type"] != "unmapped"
    )
    assert rle["polygon"] is None and json.loads(rle["rle"])["counts"] == "abc"
    attrs = json.loads(rle["attrs"])
    assert attrs["is_crowd"] is True and attrs["modality"] == "optical"
    assert attrs["licence_class"] == "open" and rle["ann_license"] == "Apache-2.0"


def test_stems_stride_and_gaps(reg):
    assert mi.staged_stem(SPEC, "val", "val_00001.jpg")[0] == "USIS10K_zip_USIS10K_val_val_00001"
    assert mi.staged_stem(mi.INSTANCE_SOURCES["uiis"], "val", "L_1.jpg")[0] == "val_L_1"
    rows = [{"i": i} for i in range(100)]
    assert [r["i"] for r in mi.stride(rows, 4)] == [0, 25, 50, 75]
    assert mi.stride(rows, None) is rows
    for spec in mi.INSTANCE_SOURCES.values():
        assert reg.crosswalk(spec.crosswalk_id)
    assert {"pingmapper-sss-seg", "aris-didson-fish-td"} <= set(mi.U7_GAPS)
