"""WP-U6a/b: COCO / YOLO / FathomNet / VOC / CSV(MOT) box readers, the FGVC per-image licence join and
the unified ``boxes`` table (offline, stubs)."""  # noqa: E501

from __future__ import annotations  # noqa: I001

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
from marinedata.task_layers.boxes_licence_join import FathomnetLicenceJoin, image_uuid
from marinedata.task_layers.sources.boxes_common import BoxFormatError, pixel_columns
from marinedata.task_layers.sources.boxes_csv import (
    MOT_GT,
    CsvLayout,
    parse_seqinfo,
    read_csv_boxes,
)
from marinedata.task_layers.sources.boxes_fathomnet import read_fathomnet_json
from marinedata.task_layers.sources.boxes_voc import read_voc
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
    spec = bt.BOX_SOURCES["roboflow-aquarium"]  # (the FGVC sets fall back to restricted-nd, WP-U6b)
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


# ---- WP-U6b: VOC XML ----------------------------------------------------------------------------

VOC = """<annotation><filename>a.jpg</filename><size><width>200</width><height>100</height></size>
<object><name>fish</name><pose>Left</pose><truncated>1</truncated><difficult>0</difficult>
<bndbox><xmin>20</xmin><ymin>10</ymin><xmax>120</xmax><ymax>60</ymax></bndbox></object>
<object><name>crab</name><truncated>0</truncated><difficult>1</difficult>
<bndbox><xmin>150</xmin><ymin>50</ymin><xmax>260</xmax><ymax>90</ymax></bndbox></object>
<object><name>zero</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""


def test_voc_flags_pixels_clipping_and_degenerate():
    image, counts = read_voc(VOC)
    assert (image.filename, image.width, image.height) == ("a.jpg", 200, 100)
    fish, crab = image.boxes
    assert (fish.x_min, fish.y_min, fish.x_max, fish.y_max) == pytest.approx((0.1, 0.1, 0.6, 0.6))
    assert fish.native == "fish" and fish.attrs == {"difficult": False, "truncated": True,
                                                    "pose": "Left"}  # fmt: skip
    assert crab.attrs["difficult"] is True and crab.attrs["truncated"] is False
    assert crab.x_max == 1.0 and crab.attrs["clipped"] is True  # xmax 260 > width 200
    assert (counts.read, counts.kept, counts.clipped, counts.degenerate) == (3, 2, 1, 1)
    assert pixel_columns(fish)["x_max_px"] == 120


def test_voc_size_override_and_errors():
    no_size = "<annotation><object><name>x</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>5</xmax><ymax>5</ymax></bndbox></object></annotation>"  # noqa: E501
    with pytest.raises(BoxFormatError, match="image size"):
        read_voc(no_size)
    (b,), _ = read_voc(no_size, img_w=10, img_h=10)[0].boxes, None
    assert b.x_max == 0.5
    with pytest.raises(BoxFormatError, match="not XML"):
        read_voc("<annotation>")
    with pytest.raises(BoxFormatError, match="expected <annotation>"):
        read_voc("<other/>")
    bad = VOC.replace("<xmin>20</xmin>", "<xmin>x</xmin>")
    with pytest.raises(BoxFormatError, match="bndbox"):
        read_voc(bad)


# ---- WP-U6b: CSV (headed, and MOT ground truth) ----------------------------------------------------  # noqa: E501


def test_csv_headed_xyxy_groups_by_image_with_label_and_confidence():
    text = "file,xmin,ymin,xmax,ymax,cls,score\na.jpg,10,10,50,30,fish,0.9\na.jpg,0,0,20,20,crab,\nb.jpg,5,5,5,9,fish,1\n"  # noqa: E501
    layout = CsvLayout(coords=("xmin", "ymin", "xmax", "ymax"), fmt="xyxy", image="file",
                       label="cls", confidence="score")  # fmt: skip
    got = read_csv_boxes(text, layout, img_w=100, img_h=50)
    (fish, crab), counts_a = got["a.jpg"]
    assert (fish.native, fish.confidence, crab.confidence) == ("fish", 0.9, None)
    assert (fish.x_min, fish.x_max, fish.y_max) == pytest.approx((0.1, 0.5, 0.6))
    assert counts_a.kept == 2
    assert got["b.jpg"][0] == () and got["b.jpg"][1].degenerate == 1  # kept as an empty image


def test_csv_unit_cxcywh_and_errors():
    layout = CsvLayout(
        coords=("cx", "cy", "w", "h"), fmt="cxcywh", unit=True, names=("cx", "cy", "w", "h")
    )
    ((boxes, _),) = read_csv_boxes("0.5,0.5,0.2,0.4\n", layout).values()
    assert (boxes[0].x_min, boxes[0].y_min) == pytest.approx((0.4, 0.3))
    with pytest.raises(BoxFormatError, match="no column 'zz'"):
        read_csv_boxes("a,b\n1,2\n", CsvLayout(coords=("a", "b", "zz", "a")))
    with pytest.raises(BoxFormatError, match="cells"):
        read_csv_boxes("0.5,0.5\n", layout)
    with pytest.raises(BoxFormatError, match="not a number"):
        read_csv_boxes(
            "a,b,c,d\nx,1,1,1\n", CsvLayout(coords=("a", "b", "c", "d")), img_w=9, img_h=9
        )


MOT = "1,32,1376,816,76,44,1,2,1,\n1,5,-32,10,80,26,0,5,0.5,\n2,32,1376,816,76,44,1,2,1,\n"


def test_mot_gt_layout_trailing_comma_ignore_flag_and_clipping():
    got = read_csv_boxes(MOT, MOT_GT, img_w=1920, img_h=1080)
    (a, b), counts = got["1"]
    assert (a.native, a.native_id, a.attrs["track_id"], a.attrs["visibility"]) == (None, "2", 32, 1)
    assert b.attrs["ignore"] is True and b.attrs["clipped"] is True and b.x_min == 0.0
    assert b.attrs["visibility"] == 0.5 and b.confidence is None  # flag is NOT a confidence
    assert (counts.kept, counts.ignored, counts.clipped) == (2, 1, 1)
    assert len(got["2"][0]) == 1
    assert parse_seqinfo("[Sequence]\nimWidth=960\nimHeight=544\n") == (960, 544)
    assert parse_seqinfo("junk") == (None, None)


# ---- WP-U6b: FGVC per-image licence join ---------------------------------------------------------

U_ND, U_OPEN, U_NONE, U_OUT = (
    "0008bd80-3b93-44df-b3a3-617ab08810b4",
    "1118bd80-3b93-44df-b3a3-617ab08810b4",
    "2228bd80-3b93-44df-b3a3-617ab08810b4",
    "3338bd80-3b93-44df-b3a3-617ab08810b4",
)
RECORDS = {
    U_ND: {"imageLicense": "CC-BY-NC-ND-4.0"},
    U_OPEN: {"imageLicense": "CC-BY-4.0"},
    U_NONE: {"imageLicense": "FathomNet"},  # not a recognised licence
}


def make_join(calls=None):
    def fetch_record(uuid):
        calls is None or calls.append(uuid)
        if uuid not in RECORDS:
            raise OSError("missing")
        return json.dumps(RECORDS[uuid]).encode()

    return FathomnetLicenceJoin(lambda: {U_ND, U_OPEN, U_NONE}, fetch_record, workers=2)


class TablesTree(StubTree):
    def __init__(self, tables, shas):
        super().__init__({}, None, shas)
        self.tables = tables

    def table(self, rel):
        return self.tables[rel]


def test_image_uuid_from_stem_filename_or_url():
    assert image_uuid("images_zip_content_images_" + U_ND.upper()) == U_ND
    assert image_uuid(None, f"x/{U_OPEN}.png") == U_OPEN
    assert image_uuid("no-uuid-here") is None


def test_join_lookup_caches_and_never_requests_unstaged_uuids():
    calls: list[str] = []
    join = make_join(calls)
    got = join.lookup(iter([U_ND, U_OUT, None, U_ND]))
    assert got[U_ND].status == "matched" and got[U_ND].licence == "CC-BY-NC-ND-4.0"
    assert got[U_OUT].status == "unmatched" and got[None].status == "unmatched"
    join.lookup([U_ND, U_OPEN])
    assert sorted(calls) == [U_ND, U_OPEN]  # U_OUT never fetched, U_ND fetched once


def test_fgvc25_rows_take_the_matching_fathomnet_image_licence_else_restricted_nd(reg):
    spec = bt.BOX_SOURCES["fathomnet-fgvc25"]
    assert spec.licence_class == bt.ND and spec.licence_join  # never open by default
    ann = json.dumps([{"category_id": 1, "category_name": "fish", "bbox": [1, 1, 10, 10]}])
    stems = [U_ND, U_OPEN, U_NONE, U_OUT]
    labels = [{"stem": s, "key": k, "value": v} for s in stems
              for k, v in (("annotations_json", ann), ("split", "train"))]  # fmt: skip
    tree = TablesTree(
        {"labels/image_labels.parquet": labels,
         "metadata.parquet": [{"stem": s, "width": "100", "height": "100"} for s in stems]},
        {("default", s): SHA for s in stems},
    )  # fmt: skip
    res = bt.staged_boxes(spec, reg, tree=tree, join=make_join())
    validate_rows("boxes", list(res.rows))
    by_uuid = {r["ann_id"]: r for r in res.rows}
    assert len(by_uuid) == 4
    got = {json.loads(r["attrs"])["licence_join"]: [] for r in res.rows}
    for r in res.rows:
        a = json.loads(r["attrs"])
        got[a["licence_join"]].append(a["licence_class"])
        assert r["ann_license"] == "CC-BY-4.0"  # the annotations' own licence stays
    assert sorted(got["matched"]) == ["open", "restricted-nd", "restricted-nd"]
    assert got["unmatched"] == ["restricted-nd"]
    assert bt.licence_split(res.rows) == {"restricted-nd": 3, "open": 1}
    nd_row = next(
        r for r in res.rows if json.loads(r["attrs"]).get("image_licence") == "CC-BY-NC-ND-4.0"
    )
    assert json.loads(nd_row["attrs"])["licence_class"] == "restricted-nd"


def test_fgvc23_columnar_rows_are_joined_on_the_file_name_uuid(reg):
    spec = bt.BOX_SOURCES["fathomnet-fgvc23"]
    lines = "\n".join(
        json.dumps({"image_id": i, "file_name": f"{u}.png", "width": 100, "height": 100,
                    "objects": [{"id": [1], "bbox": [[1.0, 1.0, 9.0, 9.0]], "category_id": [3.0]}]})
        for i, u in enumerate([U_OPEN, U_OUT])
    )  # fmt: skip
    meta = [{"stem": f"images_zip_content_images_{u}", "upstream_id": f"images.zip#content/images/{u}.png"}  # noqa: E501
            for u in (U_OPEN, U_OUT)]  # fmt: skip
    tree = TablesTree({"metadata.parquet": meta},
                      {("default", m["stem"]): SHA for m in meta})  # fmt: skip
    tree.files["labels/files/metadata.jsonl"] = lines
    res = bt.staged_boxes(spec, reg, tree=tree, join=make_join())
    classes = {json.loads(r["attrs"])["licence_join"]: json.loads(r["attrs"])["licence_class"]
               for r in res.rows}  # fmt: skip
    assert classes == {"matched": "open", "unmatched": "restricted-nd"}
    assert {r["label_native"] for r in res.rows} == {"3"}  # id only: no name guessed


# ---- WP-U6b: ruod (COCO docs), brackishmot (MOT), obsea-fish (flat YOLO, pending) -----------------  # noqa: E501


def test_ruod_two_coco_documents_are_joined_on_split_and_file_stem(reg):
    spec = bt.BOX_SOURCES["ruod"]
    assert spec.licence_class == bt.INTERNAL_ONLY
    cats = [{"id": 5, "name": "fish"}]

    def doc(files):
        return json.dumps({
            "images": [{"id": i, "file_name": f, "width": 100, "height": 100}
                       for i, f in enumerate(files, 1)],
            "annotations": [{"id": 1, "image_id": 1, "bbox": [10, 10, 20, 20], "category_id": 5,
                             "iscrowd": 0, "ignore": 0}],
            "categories": cats,
        })  # fmt: skip

    tree = StubTree(
        {"labels/files/coco_annotations_instances_train.json": doc(["000002.jpg", "000003.jpg"]),
         "labels/files/coco_annotations_instances_val.json": doc(["000009.jpg"])},
        None,
        {("default", "coco_train_000002"): SHA, ("default", "coco_val_000009"): SHA},
    )  # fmt: skip
    res = bt.staged_boxes(spec, reg, tree=tree)
    validate_rows("boxes", list(res.rows))
    assert len(res.rows) == 2 and {r["upstream_split"] for r in res.rows} == {"train", "val"}
    assert res.counts.orphan == 1  # train 000003 is not staged
    assert {json.loads(r["attrs"])["licence_class"] for r in res.rows} == {"internal-only"}
    assert bt.drop_internal_only(res.rows) == []


def test_brackishmot_frames_map_to_image_stems_with_seqinfo_size(reg):
    spec = bt.BOX_SOURCES["brackishmot"]
    seq = "BrackishMOT_test_brackishMOT-03"
    files = {f"labels/files/{seq}_gt_gt.txt": MOT,
             f"labels/files/{seq}_seqinfo.ini": "[Sequence]\nimWidth=1920\nimHeight=1080\n"}  # fmt: skip  # noqa: E501
    meta = [{"stem": f"{seq}_img1_{n:06d}", "split_hint": "test", "width": "1920", "height": "1080"}
            for n in (1, 2, 3)]  # fmt: skip
    tree = StubTree(files, meta, {("default", m["stem"]): SHA for m in meta})
    res = bt.staged_boxes(spec, reg, tree=tree)
    validate_rows("boxes", list(res.rows))
    assert len(res.rows) == 3 and res.images == 2 and res.unlabelled_images == 1  # frame 3: no gt
    assert res.counts.ignored == 1 and res.counts.clipped == 1
    assert {r["label_native"] for r in res.rows} == {"2", "5"}  # class ids, names not staged
    assert {r["upstream_split"] for r in res.rows} == {"test"}
    assert {json.loads(r["attrs"])["licence_class"] for r in res.rows} == {"internal-only"}


def test_obsea_flat_yolo_boxes_are_pending_keyed_by_the_upstream_image_stem(reg):
    spec = bt.BOX_SOURCES["obsea-fish"]
    base = f"sources/{spec.tree}/"
    head = "23sp_4120img_34945annots_2688res"
    files = {
        base + spec.names_rel: "names: ['Chromis chromis', 'Diver']\n",
        f"{base}labels/files/{head}_valid_labels_img1.txt": "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",  # noqa: E501
        f"{base}labels/files/{head}_test_labels_img2.txt": "",
        f"{base}labels/files/notalabel.txt": "0 0.5 0.5 0.2 0.2\n",
    }
    res = bt.staged_boxes(
        spec, reg,
        fetch=lambda k: files[k].encode(),
        lister=lambda prefix: [k for k in files if k.startswith(prefix)] + [prefix + "x.yaml"],
    )  # fmt: skip
    assert res.rows == () and len(res.pending) == 2 and res.unlabelled_images == 1
    assert {r["image_key"] for r in res.pending} == {f"{head}_valid_images_img1"}
    assert {r["upstream_split"] for r in res.pending} == {"val"}
    assert len(res.unparsable) == 1  # notalabel.txt
    assert bt.validate_pending(res.pending) == []
