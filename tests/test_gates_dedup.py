from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from _gates_build import Img, default_images, make_build, smooth_png
from PIL import Image

from marinedata import licence_class, neardup
from marinedata.gates import MAX_FAILURES, dedup
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


def test_the_linked_twins_count_is_always_reported_and_is_zero_without_a_linked_pair(tmp_path):
    result = run_dedup_gate(_build(tmp_path))  # no link table: nothing can be a linked twin
    assert result.passed, result.failures
    assert type(result.counts["near_pairs_linked_twins"]) is int
    assert result.counts["near_pairs_linked_twins"] == 0
    failing = run_dedup_gate(_build(tmp_path / "dup", Img(0, "test", group="gx")))
    assert not failing.passed and failing.counts["near_pairs_linked_twins"] == 0


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


# -- the two Seaview sources share the registry family `seaview-survey` --------------------------

REGISTRY = Path(__file__).resolve().parents[1] / "registry"
SEAVIEW = ("seaview-survey-imagery", "reef-support-seaview-labels")
ROBOFLOW_FORK = "roboflow-coral-bleaching-final-v6i"


def _seaview_copy_build(tmp_path, copy_split: str = "train", copy_source: str = SEAVIEW[1]):
    """A Seaview frame (train) and a re-encoded copy of it: same pHash, other sha256."""
    images = [
        Img(0, "train", SEAVIEW[0], group="sv-imagery"),
        Img(0, copy_split, copy_source, shift=6, group="sv-copy"),
    ]
    return make_build(tmp_path, images=images, masks=False)


def test_seaview_copy_in_the_same_split_is_disclosed_through_the_registry_family(tmp_path):
    build = _seaview_copy_build(tmp_path)
    assert _has(run_dedup_gate(build), "near_dup_across_sources")  # no family declared: v1.0 rule
    result = run_dedup_gate(build, families=licence_class.source_families(REGISTRY))
    assert result.passed, result.failures
    assert result.counts[IN_FAMILY] == 1
    (line,) = result.to_dict()["counts"][f"{IN_FAMILY}_sample"]
    assert "family seaview-survey" in line and all(source in line for source in SEAVIEW)


def test_seaview_copy_across_splits_still_fails(tmp_path):
    build = _seaview_copy_build(tmp_path, copy_split="test")
    result = run_dedup_gate(build, families=licence_class.source_families(REGISTRY))
    assert not result.passed and _has(result, "near_dup_across_splits"), result.failures
    assert IN_FAMILY not in result.counts


def test_a_seaview_source_shares_no_family_with_a_roboflow_fork(tmp_path):
    build = _seaview_copy_build(tmp_path, copy_source=ROBOFLOW_FORK)
    result = run_dedup_gate(build, families=licence_class.source_families(REGISTRY))
    assert not result.passed and _has(result, "near_dup_across_sources"), result.failures
    assert IN_FAMILY not in result.counts


# -- confirm-hash: a pHash-64 pair at the threshold only nominates a cross-source pair ----------

UNCONFIRMED = "near_pairs_across_sources_unconfirmed"


def _unrelated_pair(split: str = "train") -> tuple[Img, Img]:
    """Two textured scenes of two sources that no registry family joins. Their pHash-64 differ in
    4 bits by chance (within the gate's 8) while their pHash-256 differ in 56 and each carries
    about 7 bits of grey entropy."""
    return Img(665, split, "src-a", group="gu1"), Img(2736, split, "src-b", group="gu2")


def _failure(result, kind: str) -> str:
    return next(line for line in result.failures if f"{kind}:" in line)


def test_threshold_pair_the_confirm_hash_tells_apart_is_waived_and_counted(tmp_path):
    a, b = _unrelated_pair()
    result = run_dedup_gate(_build(tmp_path, a, b))
    assert result.passed, result.failures
    assert result.counts[UNCONFIRMED] == 1
    assert "failed_near_dup_across_sources" not in result.counts
    (line,) = result.to_dict()["counts"][f"{UNCONFIRMED}_sample"]
    assert line.startswith("images/train near_dup_unconfirmed: hamming 4 <= 8, pHash-256 56 ")
    assert line.endswith("(src-a vs src-b)")
    assert a.digest[:12] not in line and b.digest[:12] not in line


def test_the_waiver_is_per_pair_a_rehost_next_to_it_still_fails(tmp_path):
    rehost = Img(665, "train", "src-b", shift=6, group="gr")  # src-a's frame, brightened, by src-b
    result = run_dedup_gate(_build(tmp_path, *_unrelated_pair(), rehost))
    assert not result.passed and result.counts["failed_near_dup_across_sources"] >= 1
    assert result.counts[UNCONFIRMED] == 1
    assert "pHash-256 0 (waiver needs > 32)" in _failure(result, "near_dup_across_sources")


def test_rehost_at_the_threshold_fails_and_says_why(tmp_path):
    result = run_dedup_gate(_build(tmp_path, Img(0, "train", "src-b", shift=6, group="gm")))
    assert not result.passed and UNCONFIRMED not in result.counts
    line = _failure(result, "near_dup_across_sources")
    assert "(src-a vs src-b); pHash-256 " in line and "(waiver needs > 32)" in line


@pytest.mark.parametrize(("entropy", "waived"), [(3.9, False), (4.0, True)])
def test_a_flat_frame_is_never_waived_the_entropy_floor_is_4_bits(
    tmp_path, monkeypatch, entropy, waived
):
    monkeypatch.setattr(neardup, "grey_entropy_bits", lambda image: entropy)
    result = run_dedup_gate(_build(tmp_path, *_unrelated_pair()))
    assert result.passed is waived, result.failures
    if not waived:
        assert UNCONFIRMED not in result.counts
        assert f"min grey entropy {entropy} (needs >= 4)" in _failure(
            result, "near_dup_across_sources"
        )


def test_one_flat_frame_is_enough_to_keep_the_pair_failing(tmp_path, monkeypatch):
    a, b = _unrelated_pair()
    calls = iter([7.0, 1.0] * 40)
    monkeypatch.setattr(neardup, "grey_entropy_bits", lambda image: next(calls))
    result = run_dedup_gate(_build(tmp_path, a, b))
    assert not result.passed and _has(result, "near_dup_across_sources")


def test_a_pair_whose_bytes_cannot_be_re_read_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup, "_fingerprints", lambda build_dir, wanted: {})
    result = run_dedup_gate(_build(tmp_path, *_unrelated_pair()))
    assert not result.passed and UNCONFIRMED not in result.counts
    assert "confirm-hash unavailable" in _failure(result, "near_dup_across_sources")


def test_a_threshold_pair_across_splits_is_never_waived(tmp_path):
    a, b = Img(665, "train", "src-a", group="gu1"), Img(2736, "test", "src-b", group="gu2")
    result = run_dedup_gate(_build(tmp_path, a, b))
    assert not result.passed and _has(result, "near_dup_across_splits"), result.failures
    assert UNCONFIRMED not in result.counts


def test_unconfirmed_sample_is_capped_while_the_count_is_complete(tmp_path):
    a = [Img(665, "train", "src-a", shift=s, group="gu1") for s in range(1, 9)]
    b = [Img(2736, "train", "src-b", shift=s, group="gu2") for s in range(11, 19)]
    result = run_dedup_gate(_build(tmp_path, *a, *b))
    assert result.passed, result.failures
    assert result.counts[UNCONFIRMED] == 64 > MAX_FAILURES
    sample = result.counts[f"{UNCONFIRMED}_sample"]
    assert len(sample) == MAX_FAILURES and sample == sorted(sample)


def test_a_build_without_threshold_pairs_adds_no_unconfirmed_keys(tmp_path):
    result = run_dedup_gate(_build(tmp_path), families=FAMILY)
    assert result.passed and UNCONFIRMED not in result.counts
    assert f"{UNCONFIRMED}_sample" not in result.counts


def test_in_family_pairs_are_not_confirmed_they_stay_in_family(tmp_path):
    a, b = _unrelated_pair()
    result = run_dedup_gate(_build(tmp_path, a, b), families=FAMILY)
    assert result.passed and result.counts[IN_FAMILY] == 1
    assert UNCONFIRMED not in result.counts


def test_fingerprints_address_rows_across_row_groups(tmp_path):
    images = [Img(s, "train") for s in (3, 5, 8, 13, 21, 34, 55)]
    build = make_build(tmp_path, images=images, masks=False)
    shard = build / "data" / "images" / "train-00000-of-00001.parquet"
    pq.write_table(pq.read_table(shard), shard, row_group_size=3)
    assert pq.ParquetFile(shard).num_row_groups == 3  # 3 + 3 + 1 rows
    wanted = {f"images/{shard.name}[{i}]": i for i in (0, 2, 3, 6)}
    found = dedup._fingerprints(build, [*wanted, "images/not-a-row", "phash.parquet"])
    assert set(found) == set(wanted)
    for where, row in wanted.items():
        assert found[where] == dedup._fingerprint(images[row].data)
