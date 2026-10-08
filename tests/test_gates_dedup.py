from __future__ import annotations

import hashlib
import io

import pyarrow.parquet as pq
import pytest
from _gates_build import Img, default_images, make_build, smooth_png
from PIL import Image

from marinedata.gates.dedup import run_dedup_gate
from marinedata.gates.phash import phash


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
