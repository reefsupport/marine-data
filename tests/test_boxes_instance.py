"""HK-4c: uiis / uiis10k / usis10k boxes from COCO instance documents (tiny fixtures)."""

from __future__ import annotations

import json

import pytest

from marinedata.registry import Registry, _default_root
from marinedata.task_layers import boxes_table as bt
from marinedata.task_layers.s3_keyed import StagedTree


@pytest.fixture(scope="module")
def reg():
    return Registry.load(_default_root())


def _tree(spec, files: dict[str, bytes], stems: list[str]) -> StagedTree:
    sums = "".join(f"{i:064x}  images/{s}.jpg\n" for i, s in enumerate(stems, 1))
    blobs = {f"sources/{spec.tree}/{rel}": d for rel, d in files.items()}
    blobs[f"sources/{spec.tree}/CHECKSUMS.sha256"] = sums.encode()
    return StagedTree(spec.tree, lambda key: blobs[key])


def _doc(names: list[str], boxes: list[tuple[int, int, list[int]]]) -> bytes:
    images = [{"id": i, "file_name": n, "width": 100, "height": 50} for i, n in enumerate(names, 1)]
    anns = [
        {"id": k, "image_id": i, "category_id": 1, "bbox": b, "iscrowd": 0, "segmentation": [[0]]}
        for k, (i, _, b) in enumerate(boxes, 1)
    ]
    return json.dumps({"images": images, "annotations": anns,
                       "categories": [{"id": 1, "name": "fish"}]}).encode()  # fmt: skip


def test_bindings_are_open_apache_coco_instances():
    for sid in ("uiis", "uiis10k", "usis10k"):
        spec = bt.BOX_SOURCES[sid]
        assert (spec.reader, spec.licence_class, spec.ann_license) == (
            "coco-instances", "open", "Apache-2.0")  # fmt: skip


def test_uiis_one_box_per_instance_split_from_document(reg):
    spec = bt.BOX_SOURCES["uiis"]
    train = _doc(["a.jpg", "b.jpg"], [(1, 1, [10, 5, 20, 10])])
    val = _doc(["c.jpg"], [(1, 1, [0, 0, 50, 25])])
    files = {"labels/files/annotations_train.json": train, "labels/files/annotations_val.json": val}
    res = bt.staged_boxes(spec, reg, tree=_tree(spec, files, ["train_a", "a", "val_c", "c"]))
    assert len(res.rows) == 2 and res.images == 2  # b.jpg has no box and no staged image
    first = res.rows[0]
    assert first["label_native"] == "fish" and first["taxon_node_id"] == "A146419"
    assert (first["x_min"], first["y_min"], first["x_max"], first["y_max"]) == pytest.approx(
        (0.1, 0.1, 0.3, 0.3))  # fmt: skip
    assert [r["upstream_split"] for r in res.rows] == ["train", "val"]


def test_usis10k_reads_multiclass_only_and_prefixed_stems(reg):
    spec = bt.BOX_SOURCES["usis10k"]
    rel = "labels/files/USIS10K_multi_class_annotations_multi_class_{}_annotations.json"
    files = {rel.format(sp): _doc(["x.jpg"], [(1, 1, [1, 1, 10, 10])] if sp == "test" else [])
             for sp in ("train", "val", "test")}  # fmt: skip
    stem = "USIS10K_zip_USIS10K_test_x"
    res = bt.staged_boxes(spec, reg, tree=_tree(spec, files, [stem]), limit=None)
    assert len(res.rows) == 1 and res.rows[0]["image_sha256"] == f"{1:064x}"
