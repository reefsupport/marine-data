"""WP-8d task-label producers: opcode allowlist, ndjson rasteriser, parquet schemas.

Synthetic fixtures only — no network, no S3, no Docker (per the brief: these are unit
tests, the real restage/scan/convert runs are one-off and logged in the report).
"""

from __future__ import annotations

import io
import json
import pickle

import pyarrow.parquet as pq
import pytest

from marinedata.task_layers.sources import coralvqa, rs_labelled, seaview
from marinedata.task_layers.sources.coralseg import (
    class_counts_from_mask,
)
from marinedata.task_layers.sources.coralseg import (
    write_semseg_parquet as coralseg_write_semseg,
)

# -- SEAVIEW opcode allowlist ----------------------------------------------------------


def test_scan_globals_accepts_plain_pandas_pickle():
    """pandas>=3.0 defaults ``future.infer_string=True``, which backs even a plain
    int64 DataFrame's column-label Index with PyArrow string arrays and pulls in
    ``pyarrow.lib`` globals — real (verified 2026-09-25), but out of D-Z2's literal
    ``pandas.*, numpy.*`` allowlist, so pinned off here to test the allowlist as
    specified rather than this installation's opt-in string backend."""
    pd = pytest.importorskip("pandas")
    with pd.option_context("future.infer_string", False):
        df = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64")
        buf = io.BytesIO(pickle.dumps(df, protocol=2))  # protocol 2: plain GLOBAL opcodes
    found = seaview.scan_globals(buf)
    assert found  # pandas/numpy reconstruction globals were seen
    assert seaview.check_allowlist(found) == []


def test_scan_globals_rejects_os_system_reduce():
    class Evil:
        def __reduce__(self):
            import os

            return (os.system, ("echo pwned",))

    buf = io.BytesIO(pickle.dumps({"x": Evil()}, protocol=2))
    found = seaview.scan_globals(buf)
    bad = seaview.check_allowlist(found)
    assert bad, "os.system must never be allowlisted"
    assert any(name == "system" for _module, name in bad)


def test_check_allowlist_datetime_and_ordereddict():
    import collections
    import datetime as dt

    buf = io.BytesIO(
        pickle.dumps(
            {"t": dt.datetime(2026, 1, 1), "o": collections.OrderedDict([("a", 1)])},
            protocol=2,
        )
    )
    found = seaview.scan_globals(buf)
    assert seaview.check_allowlist(found) == []


def test_is_allowed_rejects_unknown_module():
    assert seaview._is_allowed("subprocess", "Popen") is False
    assert seaview._is_allowed("numpy.core.multiarray", "_reconstruct") is True
    assert seaview._is_allowed("builtins", "list") is True
    assert seaview._is_allowed("builtins", "eval") is False


# -- rs_labelled ndjson rasteriser -------------------------------------------------------


def _ndjson_record(objects: list[dict], width=100, height=100, external_id="img1.jpg") -> dict:
    return {
        "data_row": {"external_id": external_id},
        "media_attributes": {"width": width, "height": height},
        "projects": {
            "p1": {
                "labels": [{"annotations": {"objects": objects, "classifications": []}}],
            }
        },
    }


def test_rasterize_record_mask_url_is_unusable():
    record = _ndjson_record(
        [{"name": "Hard Coral", "annotation_kind": "ImageSegmentationMask", "mask": {"url": "x"}}]
    )
    result = rs_labelled.rasterize_record(record)
    assert result.usable is False
    assert result.class_counts is None


def test_rasterize_record_polygon_fills_pixels():
    square = [
        {"x": 10, "y": 10},
        {"x": 30, "y": 10},
        {"x": 30, "y": 30},
        {"x": 10, "y": 30},
    ]
    record = _ndjson_record([{"name": "Hard Coral", "polygon": square}])
    result = rs_labelled.rasterize_record(record)
    assert result.usable is True
    assert result.class_counts is not None
    assert result.class_counts["Hard Coral"] > 0


def test_rasterize_record_empty_objects_is_usable_with_no_counts():
    record = _ndjson_record([])
    result = rs_labelled.rasterize_record(record)
    assert result.usable is True
    assert result.class_counts == {}


def test_build_semseg_rows_keys_by_basename_sha():
    square = [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}]
    record = _ndjson_record([{"name": "Soft Coral", "polygon": square}], external_id="a/b/img1.jpg")
    line = json.dumps(record)
    rows, usable, unusable = rs_labelled.build_semseg_rows([line], {"img1.jpg": "deadbeef"})
    assert usable == 1 and unusable == 0
    assert len(rows) == 1
    assert rows[0]["sha256"] == "deadbeef"
    assert rows[0]["source_id"] == "rs-labelled-masks"
    counts = json.loads(rows[0]["class_counts"])
    assert counts["Soft Coral"] > 0


def test_build_semseg_rows_unmatched_sha_is_dropped():
    square = [{"x": 0, "y": 0}, {"x": 10, "y": 0}, {"x": 10, "y": 10}, {"x": 0, "y": 10}]
    record = _ndjson_record([{"name": "Soft Coral", "polygon": square}], external_id="unknown.jpg")
    rows, usable, unusable = rs_labelled.build_semseg_rows([json.dumps(record)], {})
    assert usable == 1 and unusable == 0
    assert rows == []


# -- CoralVQA parquet schema --------------------------------------------------------------


def test_parse_line_train_conversation_shape():
    line = json.dumps(
        {
            "image": "1.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\n[vqa] how many corals?"},
                {"from": "gpt", "value": "3"},
            ],
        }
    )
    parsed = coralvqa.parse_line(line)
    assert parsed == {
        "image": "1.jpg",
        "question": "how many corals?",
        "answer": "3",
        "qtype": "vqa",
    }


def test_parse_line_test_flat_shape_has_no_answer():
    line = json.dumps(
        {"question_id": 1, "image": "1.jpg", "text": "what color?", "category": "coral color"}
    )
    parsed = coralvqa.parse_line(line)
    assert parsed["answer"] is None
    assert parsed["qtype"] == "coral color"
    assert parsed["question"] == "what color?"


def test_build_rows_sha256_is_null_for_unstaged_images(tmp_path):
    line = json.dumps(
        {
            "image": "1.jpg",
            "conversations": [
                {"from": "human", "value": "<image>\n[vqa] q"},
                {"from": "gpt", "value": "a"},
            ],
        }
    )
    rows = coralvqa.build_rows([line], "train")
    assert rows[0]["sha256"] is None
    assert rows[0]["split_hint"] == "train"
    out = tmp_path / "vqa.parquet"
    n = coralvqa.write_vqa_parquet(rows, out)
    assert n == 1
    table = pq.read_table(out)
    assert set(table.column_names) >= {"sha256", "source_id", "label_origin", "qtype"}


# -- Coralseg mask class counts + parquet schema -----------------------------------------


def test_class_counts_from_mask_red_channel():
    from PIL import Image

    im = Image.new("RGB", (4, 2), (0, 0, 0))
    for x in range(2):
        im.putpixel((x, 0), (1, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    counts = class_counts_from_mask(buf.getvalue())
    assert counts == {"0": 6, "1": 2}


def test_coralseg_write_semseg_parquet_schema(tmp_path):
    import datetime as dt

    from marinedata.sample_schema import SampleRow

    row = SampleRow(
        sample_id="coralseg-ucsd-mosaics/train_s1",
        source_id="coralseg-ucsd-mosaics",
        source_version="unversioned",
        stem="train_s1",
        image_path="images/train_s1.jpg",
        image_sha256="feedface",
        image_bytes=10,
        image_format="jpeg",
        license="NO-LICENCE-STATED",
        attribution="Coralseg (UCSD)",
        fetch_date=dt.date(2026, 9, 25),
        label_refs=("labels/files/train_s1.png",),
    )
    out = tmp_path / "semseg.parquet"
    n = coralseg_write_semseg([row], {"feedface": {"0": 9, "1": 1}}, out)
    assert n == 1
    table = pq.read_table(out)
    assert set(table.column_names) == {
        "sha256",
        "source_id",
        "label_origin",
        "mask_key",
        "class_counts",
    }
    assert table.column("sha256").to_pylist() == ["feedface"]
    assert table.column("mask_key").to_pylist() == ["labels/files/train_s1.png"]
