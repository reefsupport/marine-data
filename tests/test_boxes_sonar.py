"""WP-U6c: sonar + synthetic-debris box readers and their crosswalks (no network: fake staged trees)."""

from __future__ import annotations

import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.registry import Registry, _default_root
from marinedata.task_layers import boxes_table as bt
from marinedata.task_layers.boxes_sonar import staged_sonar
from marinedata.task_layers.s3_keyed import StagedTree
from marinedata.task_layers.sources.boxes_common import BoxFormatError
from marinedata.task_layers.sources.boxes_synthetic import read_synthetic_json


@pytest.fixture(scope="module")
def reg():
    return Registry.load(_default_root())


def _pq(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def _tree(spec, files: dict[str, bytes], stems: list[str]) -> StagedTree:
    sums = "".join(f"{i:064x}  images/{s}.png\n" for i, s in enumerate(stems, 1))
    sums += "".join(f"{'f' * 64}  {rel}\n" for rel in files)
    blobs = {f"sources/{spec.tree}/{rel}": data for rel, data in files.items()}
    blobs[f"sources/{spec.tree}/CHECKSUMS.sha256"] = sums.encode()
    return StagedTree(spec.tree, lambda key: blobs[key])


def _meta(stems, w, h, upstream=None):
    return _pq([{"stem": s, "width": str(w), "height": str(h),
                 "upstream_id": (upstream or "{}").format(s)} for s in stems])  # fmt: skip


def test_synthetic_json_reader_normalises_pixels_and_names_fauna():
    doc = {"image": "img_0000.png", "placed": [{"class": "Tire", "box": [0, 0, 384, 384]}],
           "fauna": [{"box": [384, 384, 768, 768]}]}  # fmt: skip
    boxes, counts = read_synthetic_json(doc, img_w=768, img_h=768)
    assert [b.native for b in boxes] == ["Tire", "Fauna"]
    assert (boxes[0].x_max, boxes[0].y_max) == (0.5, 0.5) and boxes[1].x_min == 0.5
    assert counts.kept == 2 and boxes[1].attrs["layer"] == "fauna"
    with pytest.raises(BoxFormatError):
        read_synthetic_json({"image": "x.png"}, img_w=1, img_h=1)


def test_synthetic_seabed_binds_final_composite_optical_nc(reg):
    spec = bt.BOX_SOURCES["synthetic-seabed-debris"]
    final = "synthetic_seabed_dataset_v1_0_zip_synthetic_seabed_dataset_final_img_0000"
    bg = final.replace("final", "backgrounds")
    doc = {"image": "img_0000.png", "source": "gen_full_a", "fauna": [{"box": [0, 0, 77, 77]}],
           "placed": [{"class": "Tire", "box": [0, 0, 384, 384]}]}  # fmt: skip
    files = {"metadata.parquet": _meta([final, bg], 768, 768),
             "labels/files/unresolved-0.json": json.dumps(doc).encode()}  # fmt: skip
    res = staged_sonar(spec, reg, _tree(spec, files, [final, bg]), None)
    assert len(res.rows) == 2 and res.images == 1 and not res.unparsable
    tire = next(r for r in res.rows if r["label_native"] == "Tire")
    assert tire["taxon_node_id"] == "NT_DEBRIS" and tire["annotator_type"] == "pseudo"
    attrs = json.loads(tire["attrs"])
    assert attrs["modality"] == "optical" and attrs["licence_class"] == "restricted-nc"
    assert all(r["match_type"] != "unmapped" for r in res.rows)  # Fauna -> Animalia


def test_sss_mine_reads_yolo_text_from_parquet_and_keeps_class_ids(reg):
    spec = bt.BOX_SOURCES["sss-mine-detection"]
    stems = ["2010_0001", "2010_0002"]
    labels = _pq([{"stem": stems[0], "key": "txt", "value": "0 0.5 0.5 0.2 0.2\n1 0.25 0.25 0.1 0.1"},
                  {"stem": stems[1], "key": "txt", "value": ""}])  # fmt: skip
    files = {"metadata.parquet": _meta(stems, 416, 416), "labels/image_labels.parquet": labels}
    res = staged_sonar(spec, reg, _tree(spec, files, stems), None)
    assert [r["label_native"] for r in res.rows] == ["0", "1"] and res.unlabelled_images == 1
    assert {r["taxon_node_id"] for r in res.rows} == {"NT_UNKNOWN"}
    assert json.loads(res.rows[0]["attrs"])["modality"] == "sonar"
    assert res.rows[0]["ann_license"] == "CC-BY-4.0" and res.rows[0]["x_min"] == pytest.approx(0.4)


def test_swdd_matches_coco_frames_by_upstream_file_name(reg):
    spec = bt.BOX_SOURCES["swdd-sss-wall"]
    stem = "SWDD_v2_zip_ZOTERO-REPO_SWDD_SSS-Video-frames_Images_frame_000001"
    up = "SWDD.v2.zip#ZOTERO-REPO/SWDD/SSS-Video-frames/Images/frame_000001.jpg"
    doc = {"images": [{"id": 1, "file_name": "frame_000001.jpg", "width": 1570, "height": 390},
                      {"id": 2, "file_name": "frame_000002.jpg", "width": 1570, "height": 390}],
           "categories": [{"id": 0, "name": "0.0"}],
           "annotations": [{"id": 1, "image_id": 1, "category_id": 0.0, "bbox": [157, 39, 157, 39],
                            "iscrowd": 0},
                           {"id": 2, "image_id": 2, "category_id": 0.0, "bbox": [0, 0, 10, 10],
                            "iscrowd": 0}]}  # fmt: skip
    files = {"metadata.parquet": _meta([stem], 1570, 390, up),
             "labels/files/unresolved-0.json": json.dumps(doc).encode()}  # fmt: skip
    res = staged_sonar(spec, reg, _tree(spec, files, [stem]), None)
    assert len(res.rows) == 1 and res.counts.orphan == 1  # frame 2 has no staged image
    row = res.rows[0]
    assert row["label_native"] == "0.0" and row["taxon_node_id"] == "NT_WRECK"
    assert (row["x_min"], row["y_min"], row["x_max"], row["y_max"]) == pytest.approx(
        (0.1,) * 2 + (0.2,) * 2
    )
    assert json.loads(row["attrs"])["modality"] == "sonar"
