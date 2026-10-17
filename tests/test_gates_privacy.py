from __future__ import annotations

import re

import pytest
from _gates_build import FakeDetector, Img, gps_image, make_build

from marinedata.gates import summarize
from marinedata.gates.config import exif_gps_allowed, load_gates_config
from marinedata.gates.privacy import exif_has_gps, run_privacy_gate
from marinedata.gates.privacy_detectors import load_detector
from marinedata.licence_class import _registry_root


def _two_images(root):
    """The 2-image fixture: a clean image (src-a) and one with a synthetic GPSInfo IFD (src-b)."""
    clean, gps = Img(0, "train", "src-a"), Img(1, "train", "src-b", png=gps_image(1))
    return make_build(root, images=[clean, gps], masks=False), clean, gps


def test_people_rows_are_counted_not_failed(tmp_path):
    result = run_privacy_gate(make_build(tmp_path, class_names=("coral", "Diver")))
    assert result.passed and result.skipped is None and not result.failures
    assert result.counts["people_rows"] == 4 and result.counts["people_labels"] == {"Diver": 4}


def test_clean_build_passes_without_a_detector_and_scans_every_image(tmp_path):
    result = run_privacy_gate(make_build(tmp_path))
    assert result.passed and result.skipped is None and not result.failures
    assert result.counts["people_rows"] == 0 and result.counts["images_scanned"] == 20
    assert result.counts["exif_gps_images"] == 0 and "detector" not in result.counts


def test_exif_gps_image_fails_and_the_row_names_shard_sha_and_source(tmp_path):
    build, clean, gps = _two_images(tmp_path)
    result = run_privacy_gate(build)
    assert not result.passed and result.skipped is None and len(result.failures) == 1
    line = result.failures[0]
    assert re.match(r"images/train-\d{5}-of-\d{5}\.parquet\[\d+\] ", line)
    assert f"image_sha256={gps.digest}" in line and "source=src-b" in line
    assert "exif_gps_present" in line and clean.digest not in line
    assert result.counts["images_scanned"] == 2 and result.counts["exif_gps_images"] == 1
    assert result.counts["exif_gps_allowed"] == 0


def test_exif_gps_is_allowed_per_source(tmp_path):
    build, _, _ = _two_images(tmp_path)
    allowed = run_privacy_gate(build, allow_exif_gps={"src-b"})
    assert allowed.passed and not allowed.failures
    assert allowed.counts["exif_gps_images"] == 1 and allowed.counts["exif_gps_allowed"] == 1
    assert allowed.counts["exif_gps_allowed_sources"] == ["src-b"]
    other = run_privacy_gate(build, allow_exif_gps={"src-a"})
    assert not other.passed and len(other.failures) == 1


def test_allow_list_comes_from_registry_gates_yaml(tmp_path):
    build, _, _ = _two_images(tmp_path)
    root = tmp_path / "reg"
    registry = _registry_root(root)
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "gates.yaml").write_text(
        "privacy:\n  allow_exif_gps:\n    src-b: true\n    src-a: false\n"
    )
    assert exif_gps_allowed(root) == frozenset({"src-b"})
    assert run_privacy_gate(build, registry_root=root).passed
    (registry / "gates.yaml").write_text("privacy:\n  allow_exif_gps:\n    src-b: sometimes\n")
    with pytest.raises(ValueError, match="allow_exif_gps"):
        run_privacy_gate(build, registry_root=root)


def test_the_shipped_registry_allows_exif_gps_only_for_seaview():
    # D-EXIF (2026-10-17): the Seaview survey publishes per-image lat/lon by design.
    assert load_gates_config()["privacy"]["allow_exif_gps"] == {"seaview-survey-imagery": True}
    assert exif_gps_allowed() == frozenset({"seaview-survey-imagery"})


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_exif_has_gps_reads_the_gpsinfo_tag(fmt):
    assert exif_has_gps(gps_image(3, fmt))
    assert not exif_has_gps(gps_image(3, fmt, gps=False))


def test_unreadable_image_bytes_fail_the_row(tmp_path):
    bad = Img(2, "train", "src-a", png=b"definitely not an image")
    result = run_privacy_gate(make_build(tmp_path, images=[Img(0), bad], masks=False))
    assert not result.passed and result.counts["exif_unreadable"] == 1
    assert len(result.failures) == 1 and "exif_unreadable" in result.failures[0]
    assert bad.digest in result.failures[0]


@pytest.mark.parametrize("kind", ["face", "plate"])
def test_detector_hit_above_threshold_fails(tmp_path, kind):
    result = run_privacy_gate(make_build(tmp_path), detector=FakeDetector(0.9, kind))
    assert not result.passed and result.skipped is None
    assert all(f"{kind}_detected:" in line for line in result.failures)
    assert result.counts["images_scanned"] == 20 and result.counts["failures_total"] == 20


def test_detector_and_exif_failures_are_reported_together(tmp_path):
    build, _, _ = _two_images(tmp_path)
    result = run_privacy_gate(build, detector=FakeDetector(0.9, "face"))
    text = "\n".join(result.failures)
    assert "exif_gps_present" in text and "face_detected" in text and len(result.failures) == 3


def test_detector_below_threshold_passes_and_threshold_is_a_flag(tmp_path):
    build = make_build(tmp_path)
    assert run_privacy_gate(build, detector=FakeDetector(0.5)).passed
    assert run_privacy_gate(build, detector=FakeDetector(0.9), threshold=0.95).passed


def test_build_without_image_bytes_is_skipped(tmp_path):
    build = make_build(tmp_path, images=[], masks=False, extra_metadata=[{}])
    assert run_privacy_gate(build).skipped
    assert run_privacy_gate(build, detector=FakeDetector()).skipped


def test_load_detector_specs():
    assert load_detector(None) is None
    with pytest.raises(ValueError, match=r"package\.module:attribute"):
        load_detector("nocolon")
    assert load_detector("_gates_build:FakeDetector").name == "fake"


def test_privacy_is_required_only_when_it_ran():
    gates = {"licence": {"passed": True}, "privacy": {"passed": False, "skipped": "no image bytes"}}
    assert "privacy" not in summarize(gates)["required_for_publish"]
    ran = {**gates, "privacy": {"passed": True}}
    assert "privacy" in summarize(ran)["required_for_publish"]
