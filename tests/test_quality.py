"""Tests for :mod:`marinedata.quality` and ``marinedata quality``.

The fixture corpus (12 synthetic images below) is built from fully deterministic pixel
arrays — constants and hand-laid blocks, never randomness — so every assertion is an
exact, hand-checkable value rather than a comparison against the function's own output.
All images are constructed at exactly 512 px short side, where
:func:`marinedata.quality._resize_short_side` is a verified no-op (see
``test_resize_short_side_512_is_identity``), so the pixel counts used to hand-derive
expected fractions below are the *same* pixels the scorer sees.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from marinedata.cli_quality import discover_inputs, run_quality
from marinedata.quality import (
    QUALITY_THRESHOLDS,
    QualityScores,
    _resize_short_side,
    compute_flags,
    score_array,
    score_bytes,
)

SHORT_SIDE = 512
LONG_SIDE = 600  # short side is the 512 dimension throughout this file


def _png_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return buf.getvalue()


def _constant(value: int, h: int = SHORT_SIDE, w: int = LONG_SIDE) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _half_split(
    lo: int, hi: int, lo_frac: float, h: int = SHORT_SIDE, w: int = LONG_SIDE
) -> np.ndarray:
    """First ``lo_frac`` of columns at ``lo``, the rest at ``hi`` — an exact, countable split."""
    lo_cols = round(w * lo_frac)
    arr = np.full((h, w, 3), hi, dtype=np.uint8)
    arr[:, :lo_cols] = lo
    return arr


def _checkerboard(h: int = SHORT_SIDE, w: int = LONG_SIDE, tile: int = 8) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((yy // tile) + (xx // tile)) % 2) * 255
    return np.stack([pattern, pattern, pattern], axis=-1).astype(np.uint8)


def _almost_blank(
    background: int = 100, outlier: int = 200, h: int = SHORT_SIDE, w: int = LONG_SIDE
) -> np.ndarray:
    arr = _constant(background, h, w)
    arr[0, 0] = outlier
    return arr


# --- resize identity, the premise every other test relies on -----------------------


def test_resize_short_side_512_is_identity():
    arr = _half_split(0, 128, 0.5)
    assert np.array_equal(_resize_short_side(arr, SHORT_SIDE), arr)


# --- exact-value fixtures ------------------------------------------------------------

FIXTURES: dict[str, np.ndarray] = {
    "black": _constant(0),
    "white": _constant(255),
    "mid_gray": _constant(127),
    "near_black_30pct": _half_split(0, 128, 0.30),  # 30% at 0, 70% safe background
    "near_white_30pct": _half_split(255, 128, 0.30),  # 30% at 255, 70% safe background
    "clip_lo15_hi85": _half_split(0, 255, 0.15),  # 15% near-black, 85% near-white
    "checkerboard": _checkerboard(),
    "checkerboard_small": _checkerboard(h=100, w=150, tile=4),  # below min_side floor
    "blurred_gradient": np.repeat(
        np.linspace(0, 255, LONG_SIDE, dtype=np.uint8)[None, :, None], SHORT_SIDE, axis=0
    ).repeat(3, axis=2),
    "almost_blank": _almost_blank(),
    "decode_fail": None,  # not a real image
}


def test_black_image_all_scores_are_zero():
    s = score_array(FIXTURES["black"])
    assert s.decode_ok is True
    assert (s.width, s.height, s.min_side) == (LONG_SIDE, SHORT_SIDE, SHORT_SIDE)
    assert s.q_blur == 0.0
    assert s.q_clip_lo == 1.0
    assert s.q_clip_hi == 0.0
    assert s.q_entropy == 0.0
    assert s.q_blank is True
    assert s.q_uiqm == 0.0


def test_white_image_all_scores_are_zero_except_clip_hi():
    s = score_array(FIXTURES["white"])
    assert s.q_blur == 0.0
    assert s.q_clip_lo == 0.0
    assert s.q_clip_hi == 1.0
    assert s.q_entropy == 0.0
    assert s.q_blank is True
    assert s.q_uiqm == 0.0


def test_mid_gray_image_all_scores_are_zero_except_clip():
    s = score_array(FIXTURES["mid_gray"])
    assert s.q_blur == 0.0
    assert s.q_clip_lo == 0.0
    assert s.q_clip_hi == 0.0
    assert s.q_entropy == 0.0
    assert s.q_blank is True
    assert s.q_uiqm == 0.0


def test_near_black_30pct_clip_fraction_is_exact():
    s = score_array(FIXTURES["near_black_30pct"])
    assert s.q_clip_lo == pytest.approx(0.30)
    assert s.q_clip_hi == 0.0
    assert (s.q_clip_lo + s.q_clip_hi) > QUALITY_THRESHOLDS["clip_max"]
    assert s.q_blank is False


def test_near_white_30pct_clip_fraction_is_exact():
    s = score_array(FIXTURES["near_white_30pct"])
    assert s.q_clip_hi == pytest.approx(0.30)
    assert s.q_clip_lo == 0.0


def test_clip_both_sums_past_threshold():
    s = score_array(FIXTURES["clip_lo15_hi85"])
    assert s.q_clip_lo == pytest.approx(0.15)
    assert s.q_clip_hi == pytest.approx(0.85)
    assert (s.q_clip_lo + s.q_clip_hi) == pytest.approx(1.0)


def test_checkerboard_is_sharp_and_not_flagged_blank():
    s = score_array(FIXTURES["checkerboard"])
    assert s.q_blur > 0.0
    assert s.q_blank is False
    assert s.q_entropy > 0.0


def test_blurred_gradient_has_lower_blur_than_checkerboard():
    sharp = score_array(FIXTURES["checkerboard"])
    smooth = score_array(FIXTURES["blurred_gradient"])
    assert smooth.q_blur < sharp.q_blur


def test_small_image_min_side_below_floor():
    s = score_array(FIXTURES["checkerboard_small"])
    assert s.min_side == 100
    assert s.min_side < QUALITY_THRESHOLDS["min_side_min"]


def test_almost_blank_one_pixel_still_blank_by_mode_fraction():
    s = score_array(FIXTURES["almost_blank"])
    total = SHORT_SIDE * LONG_SIDE
    mode_frac = (total - 1) / total
    assert mode_frac > QUALITY_THRESHOLDS["blank_mode_frac"]
    assert s.q_blank is True


def test_decode_failure_never_raises():
    s = score_bytes(b"not an image, just bytes")
    assert s.decode_ok is False
    assert s.width is None and s.q_blur is None and s.q_uiqm is None


def test_png_roundtrip_matches_array_scoring():
    arr = FIXTURES["near_black_30pct"]
    from_bytes = score_bytes(_png_bytes(arr))
    from_array = score_array(arr)
    assert from_bytes == from_array


# --- flags ---------------------------------------------------------------------------


def test_compute_flags_decode_failure_is_the_only_flag():
    row = QualityScores(
        decode_ok=False, width=None, height=None, min_side=None, q_blur=None,
        q_clip_lo=None, q_clip_hi=None, q_uiqm=None, q_entropy=None, q_blank=None,
    ).to_dict()
    assert compute_flags(row, blur_p1=5.0) == ["decode_failed"]


def test_compute_flags_low_res_and_clip_and_blank_together():
    row = score_array(FIXTURES["black"]).to_dict()
    row["min_side"] = 100  # force below floor for this assertion
    flags = compute_flags(row, blur_p1=None)
    assert set(flags) == {"clip_high", "low_res", "blank"}


def test_compute_flags_blur_low_uses_corpus_percentile():
    row = score_array(FIXTURES["blurred_gradient"]).to_dict()
    below = compute_flags(row, blur_p1=row["q_blur"] + 1.0)
    above = compute_flags(row, blur_p1=row["q_blur"] - 1.0)
    assert "blur_low" in below
    assert "blur_low" not in above


# --- CLI: discovery, determinism, resumability ----------------------------------------


def _write_fixture_dir(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    names_and_arrays = [
        ("black.png", FIXTURES["black"]),
        ("white.jpg", FIXTURES["white"]),
        ("mid_gray.png", FIXTURES["mid_gray"]),
        ("near_black_30pct.png", FIXTURES["near_black_30pct"]),
        ("near_white_30pct.png", FIXTURES["near_white_30pct"]),
        ("clip_both.png", FIXTURES["clip_lo15_hi85"]),
        ("checkerboard.png", FIXTURES["checkerboard"]),
        ("checkerboard_small.jpg", FIXTURES["checkerboard_small"]),
        ("blurred_gradient.png", FIXTURES["blurred_gradient"]),
        ("almost_blank.png", FIXTURES["almost_blank"]),
    ]
    for name, arr in names_and_arrays:
        fmt = "JPEG" if name.endswith(".jpg") else "PNG"
        Image.fromarray(arr).save(d / name, format=fmt)
    (d / "broken.png").write_bytes(b"not an image, just bytes")
    return d


def test_discover_inputs_sorted_and_complete(tmp_path):
    d = _write_fixture_dir(tmp_path)
    units = discover_inputs(d)
    assert [u.path.name for u in units] == sorted(p.name for p in d.iterdir())
    assert all(u.kind == "file" for u in units)


def test_run_quality_scores_every_row_including_decode_failure(tmp_path):
    d = _write_fixture_dir(tmp_path)
    out = tmp_path / "quality.parquet"
    result = run_quality(d, out, workers=1)
    assert result["rows"] == 11  # 10 real images + 1 broken

    import pyarrow.parquet as pq

    table = pq.read_table(out).to_pylist()
    by_flags = {tuple(sorted(r["flags"])) for r in table}
    assert ("decode_failed",) in by_flags


def test_output_is_byte_identical_across_two_runs(tmp_path):
    d = _write_fixture_dir(tmp_path)
    out1 = tmp_path / "q1.parquet"
    out2 = tmp_path / "q2.parquet"
    run_quality(d, out1, workers=1)
    run_quality(d, out2, workers=1)
    assert out1.read_bytes() == out2.read_bytes()


def test_output_is_byte_identical_across_worker_counts(tmp_path):
    d = _write_fixture_dir(tmp_path)
    out_1w = tmp_path / "q_1w.parquet"
    out_nw = tmp_path / "q_nw.parquet"
    run_quality(d, out_1w, workers=1)
    run_quality(d, out_nw, workers=4)
    assert out_1w.read_bytes() == out_nw.read_bytes()


def test_resume_skips_already_scored_rows_and_stays_byte_identical(tmp_path):
    d = _write_fixture_dir(tmp_path)
    out = tmp_path / "quality.parquet"
    baseline = tmp_path / "baseline.parquet"
    run_quality(d, baseline, workers=1)

    result1 = run_quality(d, out, workers=1)
    assert result1["new"] == result1["rows"]

    result2 = run_quality(d, out, workers=1)
    assert result2["new"] == 0
    assert result2["rows"] == result1["rows"]
    assert out.read_bytes() == baseline.read_bytes()


def test_run_quality_on_hf_images_parquet_dir(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    hf_dir = tmp_path / "images"
    hf_dir.mkdir()
    rows = [
        {
            "image_sha256": f"sha-{i}",
            "image": {"bytes": _png_bytes(arr), "path": f"{i}.png"},
            "source_id": "synthetic",
            "source_ids": "synthetic",
            "split_group": "train",
        }
        for i, arr in enumerate([FIXTURES["black"], FIXTURES["white"], FIXTURES["checkerboard"]])
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, hf_dir / "train-00000-of-00001.parquet")

    units = discover_inputs(hf_dir)
    assert len(units) == 1 and units[0].kind == "parquet"

    out = tmp_path / "quality.parquet"
    result = run_quality(hf_dir, out, workers=1)
    assert result["rows"] == 3
    scored = pq.read_table(out).to_pylist()
    assert {r["image_sha256"] for r in scored} == {"sha-0", "sha-1", "sha-2"}
