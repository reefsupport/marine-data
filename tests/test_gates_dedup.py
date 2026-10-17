from __future__ import annotations

import hashlib
import io

import pyarrow.parquet as pq
import pytest
from _gates_build import Img, default_images, make_build, smooth_png
from PIL import Image

from marinedata import licence_class
from marinedata.gates import MAX_FAILURES
from marinedata.gates.dedup import run_dedup_gate
from marinedata.gates.phash import phash
from marinedata.gates.runner import run_gate


def _has(result, kind: str) -> bool:
    return any(f"{kind}:" in line for line in result.failures)


def _build(tmp_path, *extra: Img):
    return make_build(tmp_path, images=[*default_images(), *extra], masks=False)


def test_clean_build_passes_and_writes_phash_table(tmp_path):
    build = _build(tmp_path)
    result = run_dedup_gate(build)
    assert result.passed, result.failures
    table = pq.read_table(build / "phash.parquet")
    assert table.num_rows == 20 and {"image_sha256", "phash"} <= set(table.column_names)


@pytest.mark.parametrize(
    ("kind", "extra"),
    [
        ("exact_dup_across_splits", Img(0, "test", group="gx")),
        ("exact_dup_across_sources", Img(0, "train", "src-b", group="gy")),
        ("exact_dup_within_split", Img(0, "train", group="gz")),
        ("sha256_mismatch", Img(40, sha="0" * 64)),
        ("near_dup_across_splits", Img(0, "test", shift=6, group="gn")),
        ("near_dup_across_sources", Img(0, "train", "src-b", shift=6, group="gm")),
        ("undecodable_image", Img(41, png=b"not an image")),
    ],
)
def test_failure_modes(tmp_path, kind, extra):
    result = run_dedup_gate(_build(tmp_path, extra))
    assert not result.passed and _has(result, kind), result.failures


def test_near_pair_inside_one_split_and_source_is_counted_not_failed(tmp_path):
    result = run_dedup_gate(_build(tmp_path, Img(0, shift=6, group="gk")))
    assert result.passed, result.failures
    assert result.counts["near_pairs_same_split_source"] >= 1


def test_phash_bits_threshold(tmp_path):
    hashes = [phash(Image.open(io.BytesIO(smooth_png(s)))) for s in (0, 1)]
    distance = bin(hashes[0] ^ hashes[1]).count("1")
    build = make_build(tmp_path, images=[Img(0, "train"), Img(1, "test")], masks=False)
    assert not run_dedup_gate(build, phash_bits=distance).passed
    assert run_dedup_gate(build, phash_bits=distance - 1).passed


def test_image_free_build_is_skipped_never_passed(tmp_path):
    build = make_build(tmp_path, images=[], masks=False, extra_metadata=[{}])
    result = run_dedup_gate(build)
    assert not result.passed and result.skipped and "pixels" in result.skipped
    assert result.to_dict()["skipped"] == result.skipped


def test_dedup_gate_json_not_ok_fails(tmp_path):
    build = _build(tmp_path)
    (build / "DEDUP_GATE.json").write_text('{"ok": false}')
    assert _has(run_dedup_gate(build), "dedup_report")
    (build / "DEDUP_GATE.json").write_text('{"ok": true}')
    assert run_dedup_gate(build).passed


def _upright(seed: int) -> tuple[str, bytes]:
    """(sha256 of the upstream bytes, the same picture re-encoded after an EXIF rotation)."""
    upstream, upright = smooth_png(seed), io.BytesIO()
    Image.open(io.BytesIO(upstream)).transpose(Image.Transpose.ROTATE_90).save(upright, "PNG")
    return hashlib.sha256(upstream).hexdigest(), upright.getvalue()


def test_exif_transposed_row_keeps_the_upstream_sha_and_passes(tmp_path):
    sha, upright = _upright(40)
    result = run_dedup_gate(_build(tmp_path, Img(40, sha=sha, png=upright, exif_transposed=True)))
    assert result.passed, result.failures


def test_the_same_row_without_the_flag_is_still_a_sha256_mismatch(tmp_path):
    sha, upright = _upright(40)
    assert _has(run_dedup_gate(_build(tmp_path, Img(40, sha=sha, png=upright))), "sha256_mismatch")


def test_exif_transposed_row_must_decode_and_match_its_size(tmp_path):
    sha, upright = _upright(40)
    row = Img(40, sha=sha, png=upright, exif_transposed=True, row_size=(32, 64))
    wrong = run_dedup_gate(_build(tmp_path / "a", row))
    assert _has(wrong, "size_mismatch") and not _has(wrong, "sha256_mismatch")
    broken = Img(41, sha="0" * 64, png=b"not an image", exif_transposed=True)
    result = run_dedup_gate(_build(tmp_path / "b", broken))
    assert _has(result, "undecodable_image") and not _has(result, "sha256_mismatch")


# -- D-FORK B: family-aware reporting of same-split cross-source near pairs --------------------

IN_FAMILY = "near_pairs_across_sources_in_family"
FAMILY = {"src-a": "universe", "src-b": "universe"}
"""What ``source_families()`` returns for two forks of one upstream set."""


def _fork_of_image_0(split: str = "train") -> Img:
    """A brightened copy of default image 0 (train, src-a) published by the other source."""
    return Img(0, split, "src-b", shift=6, group="gf")


def test_same_split_near_pair_inside_a_family_is_reported_and_passes(tmp_path):
    build = _build(tmp_path, _fork_of_image_0())
    assert _has(run_dedup_gate(build), "near_dup_across_sources")  # no family declared: v1.0 rule
    result = run_dedup_gate(build, families=FAMILY)
    assert result.passed, result.failures
    assert result.counts[IN_FAMILY] >= 1
    assert "failed_near_dup_across_sources" not in result.counts
    sample = result.to_dict()["counts"][f"{IN_FAMILY}_sample"]
    assert 1 <= len(sample) <= result.counts[IN_FAMILY]
    assert all("near_dup_in_family" in line and "family universe" in line for line in sample)
    assert any("src-a vs src-b" in line or "src-b vs src-a" in line for line in sample)


def test_in_family_pair_across_splits_still_fails(tmp_path):
    result = run_dedup_gate(_build(tmp_path, _fork_of_image_0("test")), families=FAMILY)
    assert not result.passed and _has(result, "near_dup_across_splits"), result.failures
    assert IN_FAMILY not in result.counts


@pytest.mark.parametrize(
    "families",
    [
        {"src-a": "universe", "src-b": "other"},
        {"src-a": "universe"},
        {"src-b": "universe"},
        {},
        None,
    ],
    ids=["two-families", "only-a-declared", "only-b-declared", "empty", "none"],
)
def test_same_split_near_pair_outside_a_family_still_fails(tmp_path, families):
    result = run_dedup_gate(_build(tmp_path, _fork_of_image_0()), families=families)
    assert not result.passed and _has(result, "near_dup_across_sources"), result.failures
    assert IN_FAMILY not in result.counts


def test_family_does_not_excuse_an_exact_duplicate_across_sources(tmp_path):
    extra = Img(0, "train", "src-b", group="gy")
    result = run_dedup_gate(_build(tmp_path, extra), families=FAMILY)
    assert not result.passed and _has(result, "exact_dup_across_sources"), result.failures


def test_in_family_sample_is_capped_while_the_count_is_complete(tmp_path):
    forks = [Img(0, "train", "src-a", shift=s, group="ga") for s in range(1, 9)]
    forks += [Img(0, "train", "src-b", shift=s, group="gb") for s in range(11, 19)]
    result = run_dedup_gate(_build(tmp_path, *forks), families=FAMILY)
    assert result.passed, result.failures
    assert result.counts[IN_FAMILY] > MAX_FAILURES
    assert len(result.counts[f"{IN_FAMILY}_sample"]) == MAX_FAILURES
    assert result.counts[f"{IN_FAMILY}_sample"] == sorted(result.counts[f"{IN_FAMILY}_sample"])


def test_a_build_without_family_pairs_adds_no_family_keys(tmp_path):
    result = run_dedup_gate(_build(tmp_path), families=FAMILY)
    assert result.passed and IN_FAMILY not in result.counts
    assert f"{IN_FAMILY}_sample" not in result.counts


def test_release_gate_runner_hands_the_registry_family_map_to_the_gate(tmp_path, monkeypatch):
    build = _build(tmp_path, _fork_of_image_0())
    assert not run_gate("dedup", build).passed  # src-a / src-b belong to no registry family
    monkeypatch.setattr(licence_class, "source_families", lambda root=None: FAMILY)
    result = run_gate("dedup", build)
    assert result.passed, result.failures
    assert result.counts[IN_FAMILY] >= 1
