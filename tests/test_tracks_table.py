"""WP-U9: the MOT track reader and the unified ``tracks`` table (offline, stubs)."""

# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

from marinedata.annotation_schema import validate_rows
from marinedata.registry import Registry, _default_root
from marinedata.task_layers import tracks_table as tt
from marinedata.task_layers.sources.boxes_common import BoxFormatError
from marinedata.task_layers.sources.tracks_mot import (
    MOT_CHALLENGE,
    SOT_GROUNDTRUTH,
    parse_seqinfo,
    read_mot,
)

SHA = "ab" * 32


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load(_default_root())


def test_mot_challenge_frames_are_zero_based_and_native_frame_is_kept():
    boxes, counts = read_mot("1,7,10,10,20,20,1,2,1,\n2,7,12,10,20,20,1,2,1,\n", MOT_CHALLENGE,
                             img_w=100, img_h=100)  # fmt: skip
    assert [(b.frame_idx, b.native_frame, b.track_id) for b in boxes] == [(0, 1, "7"), (1, 2, "7")]
    assert boxes[0].box.native_id == "2" and counts.kept == 2
    assert boxes[0].box.x_max == pytest.approx(0.3)


def test_mot_ignored_zero_visibility_clipped_and_degenerate_rows():
    text = "\n".join([
        "1,7,10,10,20,20,0,1,0.5",  # flag 0: ignore
        "2,7,90,90,30,30,1,1,1",  # runs 20 px past the corner: clipped
        "3,7,10,10,20,20,1,1,0",  # visibility 0: fully occluded
        "4,7,500,500,5,5,1,1,1",  # fully outside: no area left, dropped
        "5,7,10,10,0,20,1,1,1",  # zero width, dropped
    ])  # fmt: skip
    boxes, counts = read_mot(text, MOT_CHALLENGE, img_w=100, img_h=100)
    assert [b.frame_idx for b in boxes] == [0, 1, 2]
    assert boxes[0].box.attrs["ignore"] is True and counts.ignored == 1
    assert boxes[1].box.attrs["clipped"] is True and boxes[1].box.x_max == 1.0
    assert counts.clipped == 1  # a box fully outside is dropped, not counted as clipped
    assert boxes[2].box.attrs["fully_occluded"] is True and boxes[2].box.attrs["visibility"] == 0
    assert (counts.read, counts.kept, counts.degenerate) == (5, 3, 2)


def test_sot_groundtruth_blank_and_absent_lines_do_not_shift_frames():
    boxes, counts = read_mot("10,10,20,20\n0,0,0,0\n\nnan,nan,nan,nan\n5,5,10,10\n",
                             SOT_GROUNDTRUTH, img_w=100, img_h=100)  # fmt: skip
    assert [(b.frame_idx, b.native_frame) for b in boxes] == [(0, 0), (4, 4)]
    assert counts.degenerate == 2 and boxes[0].track_id == "1"


def test_mot_errors_are_format_errors():
    with pytest.raises(BoxFormatError):
        read_mot("1,7,a,b,c,d\n", MOT_CHALLENGE, img_w=10, img_h=10)
    with pytest.raises(BoxFormatError):
        read_mot("0,7,1,1,2,2\n", MOT_CHALLENGE, img_w=10, img_h=10)  # frame < first frame
    with pytest.raises(BoxFormatError):
        read_mot("1,7,1,1,2,2\n", MOT_CHALLENGE, img_w=None, img_h=None)  # pixel box, no size
    assert parse_seqinfo("[Sequence]\nimWidth=1920\nimHeight=1080\n") == (1920, 1080)
    assert parse_seqinfo("garbage") == (None, None)


def _brackish_fetch(key: str) -> bytes:
    seq = "BrackishMOT_test_s1"
    files = {
        "CHECKSUMS.sha256": f"{SHA}  images/{seq}_img1_000001.jpg\n{'cd' * 32}  labels/files/{seq}_gt_gt.txt\n",
        f"labels/files/{seq}_gt_gt.txt": "1,7,10,10,20,20,1,6,1\n2,7,90,90,30,30,1,2,1\n3,7,10,10,20,20,0,1,0\n4,7,500,500,5,5,1,1,1\n",
        f"labels/files/{seq}_seqinfo.ini": "[Sequence]\nimWidth=100\nimHeight=100\n",
    }  # fmt: skip
    return files[key.split("/", 3)[3]].encode()


def test_brackishmot_tracks_resolve_class_licence_and_sha(reg):
    spec = tt.TRACK_SOURCES["brackishmot"]
    res = tt.staged_tracks(spec, reg, fetch=_brackish_fetch)
    rows = list(res.rows)
    validate_rows("tracks", rows)
    assert not res.pending and res.videos == 1 and res.counts.degenerate == 1
    assert [(r["frame_idx"], r["track_id"], r["image_sha256"]) for r in rows] == [
        (0, "7", SHA), (1, "7", None), (2, "7", None),
    ]  # fmt: skip
    assert rows[0]["video_id"] == "BrackishMOT_test_s1" and rows[0]["upstream_split"] == "test"
    assert (
        rows[0]["taxon_node_id"] == "A1740301" and rows[0]["match_type"] != "unmapped"
    )  # jellyfish
    assert rows[1]["label_native_id"] == "2" and rows[1]["taxon_node_id"] == "A106673"
    attrs = [json.loads(r["attrs"]) for r in rows]
    assert {a["licence_class"] for a in attrs} == {"internal-only"}
    assert attrs[0]["frame_staged"] is True and attrs[1]["frame_staged"] is False
    assert attrs[2]["ignore"] is True and rows[0]["is_crowd"] is None
    assert tt.write_tracks  # the table writer is importable next to the reader


def test_muot3m_staged_frames_without_a_sha_are_pending_and_the_licence_is_nd(reg, tmp_path):
    spec = tt.TRACK_SOURCES["muot3m"]
    base = f"sources/{spec.tree}/"
    keys = {
        base + "images/": [base + f"images/test_Video_001_Video_001_mp4_frame_{n:06d}.jpg" for n in (0, 3)],
        base + "labels/files/": [base + "labels/files/test_Video_001_groundtruth.txt", base + "labels/files/test_Video_001_captions.txt"],
    }  # fmt: skip
    gt = b"10,10,20,20\n0,0,0,0\nnan,nan,nan,nan\n5,5,10,10\n6,6,10,10\n"
    res = tt.staged_tracks(spec, reg, fetch=lambda _k: gt, lister=lambda p: keys[p],
                           image_size=lambda _k: (200, 100))  # fmt: skip
    assert [(r["frame_idx"], r["image_key"]) for r in res.pending] == [
        (0, "test_Video_001_Video_001_mp4_frame_000000"), (3, "test_Video_001_Video_001_mp4_frame_000003"),
    ]  # fmt: skip
    assert [(r["frame_idx"], r["image_sha256"]) for r in res.rows] == [(4, None)]
    assert res.pending[0]["x_max"] == pytest.approx(0.15) and res.pending[0][
        "y_max"
    ] == pytest.approx(0.3)
    assert (
        res.pending[0]["label_native"] == "__unlabelled"
        and res.pending[0]["match_type"] == "unmapped"
    )
    allr = [*res.rows, *res.pending]
    assert {json.loads(r["attrs"])["licence_class"] for r in allr} == {"restricted-nd"}
    assert {r["ann_license"] for r in allr} == {"CC-BY-NC-ND-4.0"} and res.counts.degenerate == 2
    assert tt.validate_pending(res.pending) == []
    assert tt.write_pending_tracks(tmp_path / "p.parquet", list(res.pending)) == 2
    bad = [{**res.pending[0], "image_sha256": SHA}]
    assert tt.validate_pending(bad)  # a pending row never carries a sha


def test_muot3m_video_without_a_staged_frame_is_reported_not_guessed(reg):
    spec = tt.TRACK_SOURCES["muot3m"]
    base = f"sources/{spec.tree}/"
    keys = {
        base + "images/": [],
        base + "labels/files/": [base + "labels/files/test_Video_009_groundtruth.txt"],
    }
    res = tt.staged_tracks(spec, reg, fetch=lambda _k: b"1,1,2,2\n", lister=lambda p: keys[p], image_size=lambda _k: None)  # fmt: skip
    assert not res.rows and not res.pending and "no staged frame" in res.unparsable[0]
