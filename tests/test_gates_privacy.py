from __future__ import annotations

import pytest
from _gates_build import FakeDetector, make_build

from marinedata.gates import summarize
from marinedata.gates.privacy import run_privacy_gate
from marinedata.gates.privacy_detectors import load_detector


def test_people_rows_are_counted_not_failed_and_without_detector_it_is_skipped(tmp_path):
    result = run_privacy_gate(make_build(tmp_path, class_names=("coral", "Diver")))
    assert result.skipped and not result.passed and not result.failures
    assert result.counts["people_rows"] == 4 and result.counts["people_labels"] == {"Diver": 4}


def test_no_people_labels_counts_zero(tmp_path):
    result = run_privacy_gate(make_build(tmp_path))
    assert result.counts["people_rows"] == 0 and result.skipped


@pytest.mark.parametrize("kind", ["face", "plate"])
def test_detector_hit_above_threshold_fails(tmp_path, kind):
    result = run_privacy_gate(make_build(tmp_path), detector=FakeDetector(0.9, kind))
    assert not result.passed and result.skipped is None
    assert all(f"{kind}_detected:" in line for line in result.failures)
    assert result.counts["images_scanned"] == 20 and result.counts["failures_total"] == 20


def test_detector_below_threshold_passes_and_threshold_is_a_flag(tmp_path):
    build = make_build(tmp_path)
    assert run_privacy_gate(build, detector=FakeDetector(0.5)).passed
    assert run_privacy_gate(build, detector=FakeDetector(0.9), threshold=0.95).passed


def test_detector_without_images_is_skipped(tmp_path):
    build = make_build(tmp_path, images=[], masks=False, extra_metadata=[{}])
    assert run_privacy_gate(build, detector=FakeDetector()).skipped


def test_load_detector_specs():
    assert load_detector(None) is None
    with pytest.raises(ValueError, match=r"package\.module:attribute"):
        load_detector("nocolon")
    assert load_detector("_gates_build:FakeDetector").name == "fake"


def test_privacy_is_required_only_when_it_ran():
    gates = {"licence": {"passed": True}, "privacy": {"passed": False, "skipped": "no detector"}}
    assert "privacy" not in summarize(gates)["required_for_publish"]
    ran = {**gates, "privacy": {"passed": True}}
    assert "privacy" in summarize(ran)["required_for_publish"]
