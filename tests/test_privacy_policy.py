"""WP-R9: the release privacy policy, one image at each score band, wired into the export."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

from marinedata.privacy import policy as pp


def _scan_row(sha: str, *scores: float) -> dict:
    boxes = [{"x": 0.0, "y": 0.0, "w": 40.0, "h": 40.0, "score": s} for s in scores]
    return {"image_sha256": sha, "face_boxes": boxes, "face_detector_version": "yunet_2023mar"}


def _row(sha: str, source: str = "src-a"):
    return SimpleNamespace(image_sha256=sha, source_id=source)


def test_thresholds_are_one_pair_of_constants():
    assert (pp.FACE_EXCLUDE_SCORE, pp.FACE_FLAG_SCORE) == (0.85, 0.60)


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (None, "none"),
        (0.59, "none"),
        (0.60, "flag"),
        (0.849, "flag"),
        (0.85, "exclude"),
        (0.99, "exclude"),
    ],
)
def test_band_edges(score, expected):
    assert pp.band(score) == expected


def test_one_image_at_each_band():
    rows = [_scan_row("a" * 64, 0.9), _scan_row("b" * 64, 0.7), _scan_row("c" * 64, 0.55)]
    rows.append(_scan_row("d" * 64))  # no face at all
    outcome = pp.evaluate(rows, {"a" * 64: "src-x", "b" * 64: "src-y"})
    (ex,) = outcome.exclusions
    assert ex == {
        "image_id": "a" * 64,
        "source_id": "src-x",
        "score": 0.9,
        "detector": "yunet",
        "detector_version": "yunet_2023mar",
    }
    assert outcome.flags == {"b" * 64: 0.7}  # 0.55 and no-face: nothing
    assert outcome.excluded == {"a" * 64}


def test_the_highest_box_decides():
    outcome = pp.evaluate([_scan_row("a" * 64, 0.62, 0.91)], {})
    assert outcome.excluded == {"a" * 64} and not outcome.flags


def test_apply_to_rows_drops_the_excluded_image_from_every_task(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    scan = [_scan_row("a" * 64, 0.9), _scan_row("b" * 64, 0.7)]
    pq.write_table(pa.Table.from_pylist(scan), tmp_path / "privacy.parquet")
    rows = {"t1": [_row("a" * 64), _row("b" * 64)], "t2": [_row("a" * 64), _row("c" * 64)]}
    kept, outcome = pp.apply_to_rows(rows, tmp_path / "privacy.parquet")
    assert [r.image_sha256 for r in kept["t1"]] == ["b" * 64]
    assert [r.image_sha256 for r in kept["t2"]] == ["c" * 64]
    assert rows["t1"][0].image_sha256 == "a" * 64  # input untouched
    assert outcome.exclusions[0]["source_id"] == "src-a"
    same, empty = pp.apply_to_rows(rows, None)
    assert same == rows and not empty.exclusions


def test_exclusions_are_recorded_in_release_json_and_the_card_prints_the_thresholds(tmp_path):
    release = tmp_path / "RELEASE.json"
    release.write_text(json.dumps({"flavour": "open"}))
    outcome = pp.evaluate([_scan_row("a" * 64, 0.9), _scan_row("b" * 64, 0.7)], {"a" * 64: "s"})
    pp.record_in_release(release, outcome)
    data = json.loads(release.read_text())
    assert data["flavour"] == "open"
    assert data["privacy_exclusions"][0]["image_id"] == "a" * 64
    assert data["privacy_policy"] == {
        "exclude_score": 0.85, "flag_score": 0.6, "excluded": 1, "flagged": 1,
    }  # fmt: skip
    text = "\n".join(pp.card_lines(data))
    assert "0.85" in text and "0.60" in text and "possible_face" in text and "excludes 1" in text
