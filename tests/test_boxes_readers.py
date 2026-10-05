"""WP-U6a: COCO / YOLO / FathomNet box readers and the unified ``boxes`` table (offline, stubs)."""

from __future__ import annotations

import json

import pytest

from marinedata.annotation_schema import validate_row, validate_rows
from marinedata.registry import Registry, _default_root
from marinedata.task_layers import boxes_table as bt
from marinedata.task_layers.configs import BOX_CONFIG_SOURCES, CONFIG_IDS, build_boxes_config
from marinedata.task_layers.sources.boxes_coco import (
    coco_annotation_boxes,
    coco_columnar_boxes,
    read_coco,
)
from marinedata.task_layers.sources.boxes_common import BoxFormatError, pixel_columns
from marinedata.task_layers.sources.boxes_fathomnet import read_fathomnet_json
from marinedata.task_layers.sources.boxes_yolo import read_yolo, yolo_names

SHA = "ab" * 32


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load(_default_root())


# ---- COCO -----------------------------------------------------------------------------------


def test_coco_xywh_to_normalised_xyxy_and_pixels():
    (b,), counts = coco_annotation_boxes(
        [{"bbox": [100, 50, 200, 100], "category_id": 3, "category_name": "fish"}],
        img_w=1000,
        img_h=500,
    )
    assert (b.x_min, b.y_min, b.x_max, b.y_max) == pytest.approx((0.1, 0.1, 0.3, 0.3))
    assert (b.native, b.native_id) == ("fish", "3")
    assert pixel_columns(b) == {
        "x_min_px": 100, "y_min_px": 50, "x_max_px": 300, "y_max_px": 150,
    }  # fmt: skip
    assert counts.kept == 1 and counts.clipped == 0


def test_coco_out_of_bounds_is_clipped_and_counted_degenerate_dropped():
    boxes, counts = coco_annotation_boxes(
        [
            {"bbox": [90, 0, 30, 10], "category_id": 1},  # 20 px past the right edge
            {"bbox": [500, 500, 10, 10], "category_id": 1},  # fully outside
            {"bbox": [10, 10, 0, 5], "category_id": 1},  # zero width
        ],
        img_w=100,
        img_h=100,
    )
    assert len(boxes) == 1 and boxes[0].x_max == 1.0 and boxes[0].attrs["clipped"] is True
    assert (counts.read, counts.kept, counts.clipped, counts.degenerate) == (3, 1, 1, 2)
    assert all(0.0 <= v <= 1.0 for v in (boxes[0].x_min, boxes[0].y_min, boxes[0].y_max))


def test_coco_crowd_and_ignore_flags():
    boxes, counts = coco_annotation_boxes(
        [
            {"bbox": [0, 0, 10, 10], "category_id": 1, "iscrowd": 1},
            {"bbox": [20, 20, 10, 10], "category_id": 1, "ignore": 1},
        ],
        img_w=100,
        img_h=100,
    )
    assert boxes[0].is_crowd and not boxes[1].is_crowd
    assert boxes[1].attrs["ignore"] is True and (counts.crowd, counts.ignored) == (1, 1)


def test_coco_document_licence_categories_orphans_and_empty_image():
    doc = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 200, "height": 100, "license": 1},
            {"id": 2, "file_name": "b.png", "width": 200, "height": 100},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [0, 0, 100, 50], "category_id": 7},
            {"image_id": 99, "bbox": [0, 0, 1, 1], "category_id": 7},
        ],
        "categories": [{"id": 7, "name": "coral"}],
        "licenses": [{"id": 1, "name": "CC BY-NC 4.0"}],
    }
    images, counts = read_coco(doc)
    assert images[0].licence == "CC BY-NC 4.0" and images[0].boxes[0].native == "coral"
    assert images[1].boxes == () and images[1].licence is None
    assert counts.orphan == 1 and counts.kept == 1


def test_coco_columnar_and_null_category_and_bad_bbox():
    objs = [{"id": [1, 2], "bbox": [[0, 0, 10, 10], [10, 10, 10, 10]], "category_id": [1.0, 88.0]}]
    boxes, _ = coco_columnar_boxes(objs, img_w=100, img_h=100)
    assert [b.native_id for b in boxes] == ["1", "88"] and boxes[0].native is None
    held_out, _ = coco_annotation_boxes(
        [{"bbox": [0, 0, 5, 5], "category_id": None}], img_w=10, img_h=10
    )
    assert held_out[0].native is None and held_out[0].native_id is None
    with pytest.raises(BoxFormatError):
        coco_annotation_boxes([{"bbox": [1, 2, 3]}], img_w=10, img_h=10)
    with pytest.raises(BoxFormatError):
        coco_annotation_boxes([{"bbox": [1, 2, 3, 4]}], img_w=None, img_h=None)


# ---- YOLO -----------------------------------------------------------------------------------


def test_yolo_centre_xywh_names_conf_and_polygon():
    names = yolo_names("names:\n- fish\n- shark\nnc: 2\n")
    assert names == {0: "fish", 1: "shark"} and yolo_names("names: {3: puffin}") == {3: "puffin"}
    text = "1 0.5 0.5 0.2 0.4\n\n# c\n0 0.25 0.25 0.1 0.1 0.9\n0 0.1 0.1 0.3 0.1 0.2 0.4\n"
    boxes, counts = read_yolo(text, names=names, img_w=100, img_h=200)
    assert boxes[0].native == "shark" and boxes[0].native_id == "1"
    assert (boxes[0].x_min, boxes[0].y_min, boxes[0].x_max, boxes[0].y_max) == pytest.approx(
        (0.4, 0.3, 0.6, 0.7)
    )
    assert boxes[1].confidence == 0.9
    assert boxes[2].attrs["from_polygon"] and boxes[2].x_max == pytest.approx(0.3)
    assert pixel_columns(boxes[0])["y_max_px"] == 140 and counts.kept == 3


def test_yolo_empty_file_clipping_and_malformed_line():
    boxes, counts = read_yolo("", names=None)
    assert boxes == () and counts.read == 0
    boxes, counts = read_yolo("0 0.95 0.5 0.2 0.2\n0 0.5 0.5 0 0\n")
    assert boxes[0].x_max == 1.0 and counts.clipped == 1 and counts.degenerate == 1
    assert boxes[0].native == "0"  # no names: the id is the native label
    for bad in ("fish 0.5 0.5 0.1 0.1", "0 0.5 0.5 0.1"):
        with pytest.raises(BoxFormatError):
            read_yolo(bad)


# ---- FathomNet JSON and licences -------------------------------------------------------------

FN = {
    "uuid": "u1", "sha256": SHA, "width": 720, "height": 368, "imageLicense": "CC-BY-NC-ND-4.0",
    "contributorsEmail": "someone@example.org",
    "boundingBoxes": [
        {"concept": "Aegina", "x": 224, "y": 233, "width": 24, "height": 46,
         "annotationLicense": "CC-BY-NC-4.0", "observer": "a@b.c", "reviewState": "VERIFIED"},
        {"concept": "Bathochordaeus", "x": 700, "y": 300, "width": 50, "height": 50,
         "annotationLicense": "CC0-1.0", "groupOf": True},
    ],
}  # fmt: skip


def test_licence_normalisation_and_strictest_class():
    assert bt.normalise_licence("CC BY-NC-ND 4.0") == "CC-BY-NC-ND-4.0"
    assert bt.normalise_licence("cc0") == "CC0-1.0" and bt.normalise_licence("FathomNet") is None
    assert bt.licence_class_of("CC0-1.0", "CC-BY-4.0", default=bt.ND) == bt.OPEN
    assert bt.licence_class_of("CC-BY-NC-4.0", "CC0-1.0", default=bt.OPEN) == bt.NC
    assert bt.licence_class_of("CC-BY-NC-ND-4.0", None, default=bt.OPEN) == bt.ND
    assert bt.licence_class_of(None, "FathomNet", default=bt.ND) == bt.ND  # source class


def test_fathomnet_rows_validate_with_per_row_licence_and_no_pii(reg):
    spec = bt.BOX_SOURCES["fathomnet"]
    image, counts = read_fathomnet_json(FN)
    assert counts.clipped == 1 and image.boxes[1].is_crowd  # 700+50 > 720
    resolve = bt.resolver_for_source(reg, spec)
    rows = [
        bt.box_row(spec=spec, registry=reg, resolve=resolve, ordinal=i, image_sha256=SHA, box=b,
                   image_licence=image.licence)
        for i, b in enumerate(image.boxes)
    ]  # fmt: skip
    validate_rows("boxes", rows)
    assert [r["match_type"] for r in rows] == ["unmapped", "unmapped"]  # no crosswalk before U8
    assert [r["label_native"] for r in rows] == ["Aegina", "Bathochordaeus"]
    # the image is NC-ND, so both rows are restricted-nd whatever the box licence says
    assert bt.licence_split(rows) == {bt.ND: 2}
    assert [r["ann_license"] for r in rows] == ["CC-BY-NC-4.0", "CC0-1.0"]
    assert "example.org" not in json.dumps(rows) and "a@b.c" not in json.dumps(rows)


def test_missing_licence_falls_back_to_source_class_and_unlabelled_sentinel(reg):
    spec = bt.BOX_SOURCES["fathomnet-fgvc25"]
    (box,), _ = coco_annotation_boxes([{"bbox": [0, 0, 5, 5]}], img_w=10, img_h=10)
    row = bt.box_row(spec=spec, registry=reg, resolve=bt.resolver_for_source(reg, spec),
                     ordinal=0, image_sha256=SHA, box=box)  # fmt: skip
    assert validate_row("boxes", row) == []
    assert row["label_native"] == "__unlabelled" and row["match_type"] == "unmapped"
    assert (
        json.loads(row["attrs"])["licence_class"] == bt.OPEN and row["ann_license"] == "CC-BY-4.0"
    )


def test_rf100_resolves_through_its_existing_crosswalk(reg):
    spec = bt.BOX_SOURCES["rf100-coral-lwptl"]
    (box,), _ = read_yolo("12 0.5 0.5 0.2 0.2", names=yolo_names("names: " + json.dumps(
        ["Arborescent", "Caespitose-a", "Caespitose-b", "Columnar", "Corymbose", "Digitate",
         "Encrusting", "Foliose", "Massive-Faviidae", "Massive-Merulinidae", "Massive-Mussidae",
         "Massive-Poritidae", "Solitary", "Tabular"])))  # fmt: skip
    row = bt.box_row(spec=spec, registry=reg, resolve=bt.resolver_for_source(reg, spec),
                     ordinal=0, image_sha256=SHA, box=box)  # fmt: skip
    assert row["label_native"] == "Solitary" and row["match_type"] != "unmapped"
    assert validate_row("boxes", row) == []


# ---- staged YOLO source, write, config, pending ------------------------------------------------


class StubTree:
    def __init__(self, files, meta, shas):
        self.files, self.meta, self.shas = files, meta, shas

    def get(self, rel):
        return self.files[rel].encode()

    def get_or_skip(self, rel):
        return self.files[rel].encode() if rel in self.files else None

    def table(self, rel):
        return self.meta


def test_staged_yolo_source_writes_reads_back_and_feeds_the_config(reg, tmp_path):
    spec = bt.BOX_SOURCES["roboflow-aquarium"]
    files = {
        spec.names_rel: "names: [fish, jellyfish]\n",
        "labels/a.txt": "0 0.5 0.5 0.2 0.2\n1 0.9 0.9 0.4 0.4\n",
        "labels/b.txt": "",
    }
    meta = [
        {"stem": "a", "width": "100", "height": "50", "label_refs": "['labels/a.txt']",
         "split_hint": "val"},
        {"stem": "b", "width": "100", "height": "50", "label_refs": ["labels/b.txt"],
         "split_hint": "train"},
        {"stem": "c", "width": "100", "height": "50", "label_refs": ["labels/missing.txt"]},
    ]  # fmt: skip
    tree = StubTree(files, meta, {("default", s): SHA[:-2] + f"0{i}" for i, s in enumerate("abc")})
    res = bt.staged_boxes(spec, reg, tree=tree)
    assert len(res.rows) == 2 and res.images == 1 and res.unlabelled_images == 1
    assert res.counts.clipped == 1 and res.counts.orphan == 1 and res.pending == ()
    assert {r["upstream_split"] for r in res.rows} == {"val"}
    validate_rows("boxes", list(res.rows))
    path = bt.write_boxes(tmp_path, spec.source_id, spec.version, list(res.rows))
    assert path.is_file()
    cfg = build_boxes_config(reg, tmp_path)
    assert cfg.config_id == "boxes" and len(cfg.rows) == 2 and "boxes" in CONFIG_IDS
    assert {r["licence_class"] for r in cfg.rows} == {"open"}
    assert cfg.unmapped_by_source == {"roboflow-aquarium": 1.0}
    assert set(BOX_CONFIG_SOURCES) == set(bt.BOX_SOURCES)


def test_pending_boxes_have_no_sha_and_are_validated(reg, tmp_path):
    spec = bt.BOX_SOURCES["fathomnet"]
    image, _ = read_fathomnet_json(FN)
    resolve = bt.resolver_for_source(reg, spec)
    rows = [
        {**bt.box_row(spec=spec, registry=reg, resolve=resolve, ordinal=i, image_sha256=None,
                      box=b), "image_key": "u1"}
        for i, b in enumerate(image.boxes)
    ]  # fmt: skip
    assert bt.write_pending_boxes(tmp_path / "p.pending.parquet", rows) == 2
    with pytest.raises(ValueError, match="image_key missing"):
        bt.write_pending_boxes(tmp_path / "q.parquet", [{**rows[0], "image_key": None}])


def test_internal_only_filter_is_shared_with_masks():
    rows = [{"attrs": json.dumps({"licence_class": "internal-only"})}, {"attrs": "{}"}]
    assert bt.drop_internal_only(rows) == [rows[1]]
