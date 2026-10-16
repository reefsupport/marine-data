"""The human privacy review list (registry/privacy/review.yaml) and what reads it.

Loader validation, release gate G8 (cleared / withhold / unknown), the --privacy-review option and
the v1.1.0 build dropping a withheld image from every table.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from _gates_build import gps_image
from PIL import Image
from test_gates_v11 import write_shard

from marinedata.cli_release_gate_v11 import _options, add_v11_options, wants_v11
from marinedata.gates import privacy_v11
from marinedata.gates.privacy_detectors import Detection
from marinedata.gates.v11_runner import RUNNERS, V11Options
from marinedata.privacy.review import (
    FIELDS,
    ReviewList,
    default_review_path,
    load_review,
    parse_review,
)
from marinedata.registry import Registry
from marinedata.release.build import build_release
from marinedata.task_layers import v11_tables
from marinedata.task_layers.configs.common import ConfigResult
from marinedata.task_layers.v11_tables import (
    V11Inputs,
    build_v11_tables,
    inputs_from_build,
    write_v11_report,
)

SHA = "ab" * 32
NOAA_SHA = "d47bcff3348010b94ccb8adc339f6354ad8d5b813cff3efb370f8a721e917733"


def entry(**kw) -> dict:
    return {
        "image_sha256": SHA, "verdict": "cleared", "source_id": "srcA",
        "detector": "yunet f12e12798e83", "score": 0.86, "reason": "coral close-up, no person",
        "reviewer": "R. Eviewer", "reviewed_at": "2026-10-16", **kw,
    }  # fmt: skip


def doc(*entries: dict) -> dict:
    return {"schema_version": 1, "entries": list(entries)}


def write_list(path: Path, *entries: dict) -> Path:
    path.write_text(yaml.safe_dump(doc(*entries)), encoding="utf-8")
    return path


# loader


def test_a_valid_list_splits_into_cleared_and_withheld():
    other = "cd" * 32
    got = parse_review(doc(entry(), entry(image_sha256=other, verdict="withhold")))
    assert isinstance(got, ReviewList) and len(got.entries) == 2
    assert list(got.cleared) == [SHA] and list(got.withheld) == [other]
    assert got.cleared[SHA].score == 0.86 and got.cleared[SHA].reviewed_at == "2026-10-16"


def test_the_sha_is_case_folded_and_a_yaml_date_is_accepted():
    got = parse_review(doc(entry(image_sha256=SHA.upper(), reviewed_at=dt.date(2026, 10, 16))))
    assert got.entries[0].image_sha256 == SHA and got.entries[0].reviewed_at == "2026-10-16"


@pytest.mark.parametrize("data", [{"schema_version": 1}, {"schema_version": 1, "entries": []}])
def test_a_list_without_entries_is_valid_and_empty(data):
    assert parse_review(data).entries == ()


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (doc(entry(verdict="maybe")), "verdict"),
        (doc(entry(image_sha256="abc")), "image_sha256"),
        (doc(entry(image_sha256="g" * 64)), "image_sha256"),
        (doc(entry(image_sha256=123)), "image_sha256"),
        (doc(entry(), entry(verdict="withhold")), "duplicate image_sha256"),
        (doc(entry(), entry(image_sha256=SHA.upper())), "duplicate image_sha256"),
        (doc(entry(note="x")), "unknown field"),
        (doc(entry(reason="  ")), "reason"),
        (doc(entry(reviewer=None)), "missing field.*reviewer"),
        (doc(entry(score=2)), "score"),
        (doc(entry(score=-0.1)), "score"),
        (doc(entry(score="0.9")), "score"),
        (doc(entry(score=True)), "score"),
        (doc(entry(reviewed_at="16/10/2026")), "reviewed_at"),
        (doc(entry(reviewed_at="2026-13-40")), "reviewed_at"),
        (doc(entry(reviewed_at=20261016)), "reviewed_at"),
        (doc("not a mapping"), "expected a mapping"),
        ({"schema_version": 1, "entries": {"a": 1}}, "entries must be a list"),
        ({"schema_version": 2, "entries": []}, "schema_version"),
        ({"schema_version": True, "entries": []}, "schema_version"),
        ({"entries": []}, "schema_version"),
        ([], "expected a mapping"),
    ],
)
def test_a_bad_list_is_a_value_error(data, match):
    with pytest.raises(ValueError, match=match):
        parse_review(data)


@pytest.mark.parametrize("field", FIELDS)
def test_every_field_is_required(field):
    row = {k: v for k, v in entry().items() if k != field}
    with pytest.raises(ValueError, match=f"missing field.*{field}"):
        parse_review(doc(row))


def test_load_review_reads_a_file_with_an_unquoted_yaml_date(tmp_path):
    lines = "\n    ".join(f"{k}: {v}" for k, v in entry().items())
    explicit = tmp_path / "list.yaml"
    explicit.write_text(f"schema_version: 1\nentries:\n  - {lines}\n")
    assert load_review(explicit).cleared[SHA].reviewed_at == "2026-10-16"


def test_load_review_errors_and_the_registry_default(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_review(tmp_path / "absent.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("entries: [unclosed")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_review(broken)
    assert load_review(registry_root=tmp_path).entries == ()  # no default file: no review
    (tmp_path / "privacy").mkdir()
    write_list(tmp_path / "privacy" / "review.yaml", entry())
    assert list(load_review(registry_root=tmp_path).cleared) == [SHA]


def test_the_committed_list_clears_the_noaa_bleaching_false_positive():
    got = load_review().cleared[NOAA_SHA]
    assert default_review_path().is_file() and got.source_id == "noaa-pifsc-bleaching"
    assert (got.detector, got.score, got.reviewed_at) == ("yunet f12e12798e83", 0.86, "2026-10-16")
    assert "no person visible" in got.reason and "pending confirmation" in got.reviewer
    assert not load_review().withheld


# release gate G8


class FakeDetector:
    name, version = "fake", "1"

    def __init__(self, flagged=()):
        self.flagged, self.seen = set(flagged), []

    def detect(self, image):
        self.seen.append(image)
        return [Detection("face", 0.97)] if image in self.flagged else []


def jpeg(colour):
    out = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(out, "JPEG")
    return out.getvalue()


def image_row(data):
    sha = hashlib.sha256(data).hexdigest()
    return {"image_sha256": sha, "source_id": "srcA", "image": {"bytes": data, "path": None}}


def sha_of(data):
    return hashlib.sha256(data).hexdigest()


def gate(tmp_path, images, flagged=(), *entries, **kw):
    build = tmp_path / "build"
    write_shard(build, "images", "train", [image_row(d) for d in images])
    review = write_list(tmp_path / "review.yaml", *entries)
    detector = FakeDetector(flagged)
    result = privacy_v11.run_privacy_v11_gate(
        build, detector=detector, review_path=review, allow_exif_gps=(), **kw
    )
    return result, detector


IMG_A, IMG_B = jpeg((200, 30, 30)), jpeg((30, 30, 200))


def test_a_cleared_detection_is_counted_and_listed_not_failed(tmp_path):
    cleared = entry(image_sha256=sha_of(IMG_A), reviewer="R. Eviewer")
    result, _ = gate(tmp_path, [IMG_A, IMG_B], {IMG_A}, cleared)
    assert result.passed, result.failures
    assert result.counts["cleared_reviewed"] == 1 and result.counts["images_with_detections"] == 0
    assert result.counts["cleared_reviewed_entries"] == [
        {"image_sha256": sha_of(IMG_A), "reviewer": "R. Eviewer", "reviewed_at": "2026-10-16"}
    ]
    assert (
        result.counts["review_entries_cleared"] == 1
        and result.counts["detected_image_sha256"] == []
    )


def test_a_cleared_entry_for_an_image_that_does_not_fire_is_not_counted(tmp_path):
    result, _ = gate(tmp_path, [IMG_A], (), entry(image_sha256=sha_of(IMG_A)))
    assert result.passed and result.counts["cleared_reviewed"] == 0


def test_a_review_clears_only_its_own_sha(tmp_path):
    result, _ = gate(tmp_path, [IMG_A, IMG_B], {IMG_A, IMG_B}, entry(image_sha256=sha_of(IMG_A)))
    assert not result.passed and result.counts["cleared_reviewed"] == 1
    assert result.counts["detected_image_sha256"] == [sha_of(IMG_B)]
    assert len(result.failures) == 1 and "face_detected" in result.failures[0]


def test_an_unknown_detection_still_fails_with_an_unrelated_list(tmp_path):
    result, _ = gate(tmp_path, [IMG_B], {IMG_B}, entry())
    assert not result.passed and result.counts["cleared_reviewed"] == 0
    assert sha_of(IMG_B) in result.failures[0] and "face_detected" in result.failures[0]


def test_a_withheld_image_still_in_the_build_fails(tmp_path):
    withheld = entry(image_sha256=sha_of(IMG_B), verdict="withhold")
    result, _ = gate(tmp_path, [IMG_A, IMG_B], {IMG_B}, withheld)
    assert not result.passed and result.counts["withheld_image_present"] == 1
    assert len(result.failures) == 1 and "withheld_image_present" in result.failures[0]
    assert sha_of(IMG_B) in result.failures[0] and result.counts["review_entries_withhold"] == 1


def test_a_withheld_image_fails_even_when_its_row_carries_no_bytes(tmp_path):
    build = tmp_path / "build"
    row = {"image_sha256": SHA, "source_id": "srcA", "image": {"bytes": None, "path": None}}
    write_shard(build, "images", "train", [row])
    review = write_list(tmp_path / "review.yaml", entry(verdict="withhold"))
    result = privacy_v11.run_privacy_v11_gate(
        build, detector=FakeDetector(), review_path=review, allow_exif_gps=()
    )
    assert not result.passed and "withheld_image_present" in result.failures[0]


def test_a_review_clears_the_face_not_the_exif_gps(tmp_path):
    located = gps_image(1, "JPEG")
    result, _ = gate(tmp_path, [located], {located}, entry(image_sha256=sha_of(located)))
    assert result.counts["cleared_reviewed"] == 1 and not result.passed
    assert len(result.failures) == 1 and "exif_gps_present" in result.failures[0]


@pytest.mark.parametrize("how", ["invalid", "absent"])
def test_an_unusable_review_list_fails_the_gate_closed(tmp_path, how):
    build = tmp_path / "build"
    write_shard(build, "images", "train", [image_row(IMG_A)])
    review = tmp_path / "review.yaml"
    if how == "invalid":
        write_list(review, entry(verdict="maybe"))
    detector = FakeDetector()
    result = privacy_v11.run_privacy_v11_gate(build, detector=detector, review_path=review)
    assert not result.passed and result.counts["review_list"] == "unavailable"
    assert result.failures[0].startswith("privacy review list unavailable") and not detector.seen


def test_the_registry_list_is_the_default(tmp_path):
    (tmp_path / "reg" / "privacy").mkdir(parents=True)
    write_list(tmp_path / "reg" / "privacy" / "review.yaml", entry(image_sha256=sha_of(IMG_A)))
    build = tmp_path / "build"
    write_shard(build, "images", "train", [image_row(IMG_A)])
    kw = {"detector": FakeDetector({IMG_A}), "allow_exif_gps": ()}
    cleared = privacy_v11.run_privacy_v11_gate(build, registry_root=tmp_path / "reg", **kw)
    assert cleared.passed and cleared.counts["cleared_reviewed"] == 1
    bare = privacy_v11.run_privacy_v11_gate(build, registry_root=tmp_path, **kw)  # no list there
    assert not bare.passed and bare.counts["review_entries_cleared"] == 0


# --privacy-review


def test_the_privacy_review_option_reaches_the_gate(tmp_path, monkeypatch):
    parser = argparse.ArgumentParser()
    add_v11_options(parser)
    given = parser.parse_args(["--privacy-review", "r.yaml"])
    assert given.privacy_review == Path("r.yaml") and parser.parse_args([]).privacy_review is None
    args = argparse.Namespace(
        v11=False, only=None, v1_dir=None, hidden_test=[], benchmark=[], nc_id=[],
        taxonomy_version=None, rehearsal=False, build_dir=None, flavour=None, release_dir=None,
        empty_threshold=0.05, detector=None, privacy_threshold=0.85, privacy_review=Path("r.yaml"),
    )  # fmt: skip
    assert wants_v11(args, v10_selected=False) and _options(args).privacy_review == Path("r.yaml")
    seen = {}

    def spy(build, **kw):
        seen.update(kw)
        return "gate result"

    monkeypatch.setattr(privacy_v11, "run_privacy_v11_gate", spy)
    options = V11Options(privacy_review=Path("r.yaml"), privacy_threshold=0.9)
    assert RUNNERS["privacy"](tmp_path, options) == "gate result"
    assert seen["review_path"] == Path("r.yaml") and seen["threshold"] == 0.9
    assert V11Options().privacy_review is None


# build: a withheld image leaves every v1.1.0 table

A, B, C, D = (f"{i:064x}" for i in (1, 2, 3, 4))


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def spine(monkeypatch, *items):
    """Stand in for the staged-tree walk (needs S3 libraries): ``(sha, split_group)`` images."""
    meta = {
        sha: {
            "image_sha256": sha,
            "source_id": "srcA",
            "split_group": group,
            "bytes_config": "images",
        }
        for sha, group in items
    }
    counts = {"images": len(meta), "sha256_mismatches": 0, "licence_dropped": 0}
    counts |= {"rows": len(meta), "shared": 0}
    monkeypatch.setattr(v11_tables, "_metadata_rows", lambda r, i: (meta, {}, {"srcA": counts}))


def task(sha, source_id="srcA"):
    return {"image_sha256": sha, "source_id": source_id, "match_type": "exact"}


def build_tables(registry, tmp_path, boxes=(A, B, C), **kw):
    inputs = V11Inputs(
        taxonomy_version="3.0.0", roots={"srcA": tmp_path},
        assignments={"g1": "val", "g2": "val", "g3": "test_ood"}, **kw,
    )  # fmt: skip
    results = {
        "boxes": ConfigResult("boxes", tuple(task(s) for s in boxes), {}),
        "points": ConfigResult("points", tuple(task(s) for s in (A, B, C)), {}),
    }
    return build_v11_tables(registry, results, inputs)


def shas(out, cid):
    return {r["image_sha256"] for r in out[cid].rows}


def test_a_privacy_withheld_image_leaves_every_table_and_is_counted(
    registry, tmp_path, monkeypatch
):
    spine(monkeypatch, (A, "g1"), (B, "g2"), (C, "g3"))
    out = build_tables(registry, tmp_path, (A, B, C, D), privacy_withheld=frozenset({B, D}))
    for cid in ("metadata", "boxes", "points"):
        assert shas(out, cid) == {A, C}, cid  # D is no spine image, its task row goes too
    report = out["metadata"].notes["report"]
    assert report["withheld_images"] == 1 and report["withheld_images_privacy"] == 1
    (tmp_path / "out").mkdir()
    written = json.loads(write_v11_report(out, tmp_path / "out").read_text())
    assert written["withheld_images"] == 1 and written["withheld_images_privacy"] == 1


def test_withheld_images_count_near_dup_groups_and_privacy_reviews_together(
    registry, tmp_path, monkeypatch
):
    spine(monkeypatch, (A, "g1"), (B, "g2"), (C, "g3"))
    out = build_tables(
        registry, tmp_path, withheld_groups=frozenset({"g2"}), privacy_withheld=frozenset({C})
    )
    assert shas(out, "metadata") == {A} and shas(out, "points") == {A}
    report = out["metadata"].notes["report"]
    assert report["withheld_images"] == 2 and report["withheld_images_privacy"] == 1


def test_without_a_review_nothing_is_withheld(registry, tmp_path, monkeypatch):
    spine(monkeypatch, (A, "g1"), (B, "g2"), (C, "g3"))
    report = build_tables(registry, tmp_path)["metadata"].notes["report"]
    assert report["withheld_images"] == 0 and report["withheld_images_privacy"] == 0


def test_the_build_inputs_carry_the_review_withhold_set():
    split_map = SimpleNamespace(assignments={"g1": "val"}, withheld=())
    review = parse_review(doc(entry(image_sha256=B, verdict="withhold"), entry(image_sha256=A)))
    inputs = inputs_from_build(
        taxonomy_version="3.0.0", roots={}, manifests=[], split_map=split_map,
        privacy_withheld=review.withheld,
    )  # fmt: skip
    assert inputs.privacy_withheld == frozenset({B})
    assert (
        inputs_from_build(
            taxonomy_version="3.0.0", roots={}, manifests=[], split_map=split_map
        ).privacy_withheld
        == frozenset()
    )
    assert inspect.signature(build_release).parameters["privacy_review"].default is None
