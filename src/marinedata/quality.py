"""Per-sample image quality scores (rubric D9 — "quality filtering").

Pure, deterministic functions of a decoded RGB array: no I/O, no global state, no
randomness. Every score is a function of pixel values only, so re-scoring the same
bytes on any machine, at any worker count, produces the same float bits.

Scores:

- ``q_blur``    — variance of the Laplacian on a grayscale copy resized so the short
  side is 512 px (``cv2.INTER_AREA``, a fixed, non-random interpolation). Low variance
  means a smooth/out-of-focus image.
- ``q_clip_lo`` / ``q_clip_hi`` — fraction of grayscale pixels at or below 2 / at or
  above 253 (near-black / near-white clipping), on the same 512-short-side resize.
- ``q_uiqm``    — the Underwater Image Quality Measure of Panetta, Gao & Agaian,
  "Human-Visual-System-Inspired Underwater Image Quality Measures", IEEE Journal of
  Oceanic Engineering, 2016 (doi:10.1109/JOE.2015.2469915). ``UIQM = c1*UICM + c2*UISM
  + c3*UIConM`` with the paper's coefficients ``c1=0.0282, c2=0.2953, c3=3.5753``:
  UICM (colourfulness, from the opponent RG/YB channels' trimmed mean/variance), UISM
  (sharpness, a Sobel-edge-weighted EME over 8x8 blocks per channel) and UIConM
  (contrast, a Michelson-contrast log-AMEE over 8x8 blocks on luminance).
- ``q_entropy`` — Shannon entropy (base 2) of the 256-bin grayscale histogram.
- ``q_blank``   — ``True`` if the grayscale std is below 3, or one 8-bit grey level
  covers more than 98% of pixels (near-solid-colour frame).
- ``min_side``, ``width``, ``height`` — of the *original* decoded image (before the
  512-short-side resize used for the scores above).
- ``decode_ok`` — whether the bytes decoded as an image at all.

``flags`` reasons come from exactly one threshold table, :data:`QUALITY_THRESHOLDS`,
so every consumer (CLI, card, tests) reads the same numbers.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version

import cv2
import numpy as np
from PIL import Image

#: One table for every "flag" reason. Nothing outside this module or the corpus-wide
#: blur percentile (passed in separately, since it needs the whole run) decides a
#: threshold.
QUALITY_THRESHOLDS: dict[str, float] = {
    "blur_percentile": 1.0,  # q_blur below this percentile of the corpus -> "blur_low"
    "clip_max": 0.25,  # q_clip_lo + q_clip_hi above this -> "clip_high"
    "min_side_min": 256,  # min_side below this -> "low_res"
    "blank_std_max": 3.0,  # grayscale std below this -> part of q_blank
    "blank_mode_frac": 0.98,  # one grey level covering more than this -> part of q_blank
}

#: Short side (px) every score other than width/height/min_side is computed at.
SCORE_SHORT_SIDE = 512

#: UIQM coefficients, Panetta et al. 2016, eq. (13).
_UIQM_C1_UICM = 0.0282
_UIQM_C2_UISM = 0.2953
_UIQM_C3_UICONM = 3.5753

#: Trim fraction for UICM's alpha-trimmed mean/variance (paper uses alpha_L=alpha_R=0.1).
_UICM_ALPHA = 0.1

#: Block size (px) for the EME (UISM) and log-AMEE (UIConM) block statistics.
_BLOCK_SIZE = 8


def library_versions() -> dict[str, str]:
    """Runtime versions of the pinned imaging stack, for recording in run output."""

    def _v(dist: str, fallback: str) -> str:
        try:
            return version(dist)
        except PackageNotFoundError:
            return fallback

    return {
        "numpy": np.__version__,
        "pillow": _v("pillow", "unknown"),
        "opencv-python-headless": _v("opencv-python-headless", cv2.__version__),
    }


@dataclass(frozen=True)
class QualityScores:
    """One row of scores for one decoded image. Field order matches the output schema."""

    decode_ok: bool
    width: int | None
    height: int | None
    min_side: int | None
    q_blur: float | None
    q_clip_lo: float | None
    q_clip_hi: float | None
    q_uiqm: float | None
    q_entropy: float | None
    q_blank: bool | None

    def to_dict(self) -> dict:
        return asdict(self)


_FAILED = QualityScores(
    decode_ok=False,
    width=None,
    height=None,
    min_side=None,
    q_blur=None,
    q_clip_lo=None,
    q_clip_hi=None,
    q_uiqm=None,
    q_entropy=None,
    q_blank=None,
)


def score_bytes(raw: bytes) -> QualityScores:
    """Decode ``raw`` and score it. Never raises: a decode failure is ``decode_ok=False``."""
    try:
        with Image.open(__import__("io").BytesIO(raw)) as img:
            img = img.convert("RGB")
            rgb = np.asarray(img)
    except Exception:
        return _FAILED
    if rgb.size == 0 or rgb.ndim != 3:
        return _FAILED
    return score_array(rgb)


def _resize_short_side(gray: np.ndarray, short_side: int) -> np.ndarray:
    h, w = gray.shape[:2]
    scale = short_side / min(h, w)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(gray, (new_w, new_h), interpolation=interp)


def score_array(rgb: np.ndarray) -> QualityScores:
    """Score an already-decoded ``H x W x 3`` uint8 RGB array."""
    height, width = rgb.shape[0], rgb.shape[1]
    min_side = min(height, width)

    # One resize, shared by every score below, so they all see the same pixels.
    rgb_resized = _resize_short_side(rgb, SCORE_SHORT_SIDE)
    gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

    q_blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    total_px = gray.size
    q_clip_lo = float(np.count_nonzero(gray <= 2)) / total_px
    q_clip_hi = float(np.count_nonzero(gray >= 253)) / total_px

    q_entropy = _shannon_entropy(gray)
    q_blank = _is_blank(gray)
    q_uiqm = _uiqm(rgb_resized)

    return QualityScores(
        decode_ok=True,
        width=int(width),
        height=int(height),
        min_side=int(min_side),
        q_blur=q_blur,
        q_clip_lo=q_clip_lo,
        q_clip_hi=q_clip_hi,
        q_uiqm=q_uiqm,
        q_entropy=q_entropy,
        q_blank=q_blank,
    )


def _shannon_entropy(gray: np.ndarray) -> float:
    hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / total
    return float(-np.sum(p * np.log2(p)))


def _is_blank(gray: np.ndarray) -> bool:
    std = float(gray.std())
    if std < QUALITY_THRESHOLDS["blank_std_max"]:
        return True
    hist = np.bincount(gray.reshape(-1), minlength=256)
    mode_frac = float(hist.max()) / gray.size
    return mode_frac > QUALITY_THRESHOLDS["blank_mode_frac"]


# --- UIQM (Panetta et al. 2016) ---------------------------------------------------


def _trim(values: np.ndarray, alpha: float) -> np.ndarray:
    x = np.sort(values.reshape(-1))
    k = x.size
    t = math.floor(alpha * k)
    return x[t : k - t] if k - 2 * t > 0 else x


def _trimmed_mean(values: np.ndarray, alpha: float) -> float:
    return float(_trim(values, alpha).mean())


def _trimmed_variance(values: np.ndarray, mean: float, alpha: float) -> float:
    # Paper's s_alpha^2 uses the same alpha-trimmed sample for the second moment.
    return float(np.mean((_trim(values, alpha) - mean) ** 2))


def _uicm(rgb: np.ndarray) -> float:
    r = rgb[:, :, 0].astype(np.float64)
    g = rgb[:, :, 1].astype(np.float64)
    b = rgb[:, :, 2].astype(np.float64)
    rg = r - g
    yb = (r + g) / 2.0 - b

    rg_mu = _trimmed_mean(rg, _UICM_ALPHA)
    yb_mu = _trimmed_mean(yb, _UICM_ALPHA)
    rg_var = _trimmed_variance(rg, rg_mu, _UICM_ALPHA)
    yb_var = _trimmed_variance(yb, yb_mu, _UICM_ALPHA)

    lightness = math.sqrt(rg_mu**2 + yb_mu**2)
    spread = math.sqrt(rg_var + yb_var)
    return -0.0268 * lightness + 0.1586 * spread


def _blockwise(x: np.ndarray, block: int):
    """Yield non-overlapping ``block x block`` tiles, dropping a ragged remainder."""
    h, w = x.shape[:2]
    k2, k1 = h // block, w // block
    if k1 == 0 or k2 == 0:
        yield x
        return
    cropped = x[: k2 * block, : k1 * block]
    for row in range(k2):
        for col in range(k1):
            yield cropped[row * block : (row + 1) * block, col * block : (col + 1) * block]


def _eme(x: np.ndarray, block: int) -> float:
    tiles = list(_blockwise(x, block))
    if not tiles:
        return 0.0
    total = 0.0
    for tile in tiles:
        lo = float(tile.min())
        hi = float(tile.max())
        lo = lo if lo > 0 else 1.0
        hi = hi if hi > 0 else 1.0
        total += math.log(hi / lo)
    return (2.0 / len(tiles)) * total


def _uism(rgb: np.ndarray) -> float:
    weights = (0.299, 0.587, 0.114)  # luma weights, applied per channel (R, G, B)
    total = 0.0
    for channel_idx, weight in enumerate(weights):
        channel = rgb[:, :, channel_idx].astype(np.float64)
        sobel_x = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
        edge = np.hypot(sobel_x, sobel_y) * channel
        total += weight * _eme(edge, _BLOCK_SIZE)
    return total


def _log_amee(x: np.ndarray, block: int) -> float:
    tiles = list(_blockwise(x, block))
    if not tiles:
        return 0.0
    total = 0.0
    for tile in tiles:
        lo = float(tile.min())
        hi = float(tile.max())
        denom = hi + lo
        if denom == 0:
            continue
        contrast = (hi - lo) / denom
        if contrast <= 0:
            continue
        total += contrast * math.log(contrast)
    return -(1.0 / len(tiles)) * total


def _uiconm(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64)
    return _log_amee(gray, _BLOCK_SIZE)


def _uiqm(rgb: np.ndarray) -> float:
    return (
        _UIQM_C1_UICM * _uicm(rgb)
        + _UIQM_C2_UISM * _uism(rgb)
        + _UIQM_C3_UICONM * _uiconm(rgb)
    )


# --- Flags -------------------------------------------------------------------------


def compute_flags(row: dict, *, blur_p1: float | None) -> list[str]:
    """Reasons a row would be filtered, from :data:`QUALITY_THRESHOLDS` only.

    ``blur_p1`` is the corpus's 1st percentile of ``q_blur`` among decodable, non-blank
    rows — a run-level statistic, not a fixed constant, so it is passed in rather than
    stored in the threshold table.
    """
    if not row.get("decode_ok"):
        return ["decode_failed"]
    flags: list[str] = []
    if blur_p1 is not None and row["q_blur"] < blur_p1:
        flags.append("blur_low")
    if (row["q_clip_lo"] + row["q_clip_hi"]) > QUALITY_THRESHOLDS["clip_max"]:
        flags.append("clip_high")
    if row["min_side"] < QUALITY_THRESHOLDS["min_side_min"]:
        flags.append("low_res")
    if row["q_blank"]:
        flags.append("blank")
    return flags


def blur_percentile_threshold(blur_values: list[float]) -> float | None:
    """The corpus-wide ``blur_p1`` used by :func:`compute_flags`, or ``None`` if empty."""
    if not blur_values:
        return None
    percentile = QUALITY_THRESHOLDS["blur_percentile"]
    return float(np.percentile(np.asarray(blur_values, dtype=np.float64), percentile))
