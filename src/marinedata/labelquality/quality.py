"""Cheap image-quality features and the bleach-direction quality gate (WP-9).

The blind model audit (``docs/label-quality/model-audit.tsv``) shows the confident-
learning probe confuses whiteness, colour cast and blur with bleaching: flags toward
BLEACHED/UNHEALTHY are far less reliable (37%) than flags toward HEALTHY (87%). This
module computes four cheap per-image features — blur, colour cast, luminance and
saturation — pure PIL/numpy, no heavyweight CV library — and a gate that keeps a
bleach-direction flag only if the image is sharp, colour-balanced, well exposed and the
probe's margin is high enough. Thresholds are tuned by grouped 5-fold CV on the audited
set; see :func:`tune_gate` and ``docs/label-quality.md``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np


def _rgb_array(image_bytes: bytes) -> np.ndarray:
    """Decode to an ``(h, w, 3)`` float32 RGB array, 0-255."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float32)


def laplacian_variance(rgb: np.ndarray) -> float:
    """Variance of the discrete Laplacian of the greyscale image — a cheap blur proxy.

    Low variance means few sharp edges, i.e. a blurry image. Pure numpy, valid
    convolution over a 3x3 Laplacian kernel (no scipy/cv2 dependency).
    """
    grey = rgb.mean(axis=2)
    lap = (
        grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * grey[1:-1, 1:-1]
    )
    return float(lap.var())


def color_cast_score(rgb: np.ndarray) -> float:
    """Grey-world colour-cast score: normalised spread of the per-channel means.

    0 = perfectly grey-world balanced; grows toward 1 as one channel dominates (green
    water tint, orange dive-light cast, etc).
    """
    means = rgb.reshape(-1, 3).mean(axis=0)
    grand = float(means.mean())
    if grand <= 0:
        return 0.0
    return float((means.max() - means.min()) / grand)


def mean_luminance(rgb: np.ndarray) -> float:
    """Rec. 709 luma, normalised to [0, 1]."""
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    return float(luma.mean() / 255.0)


def saturation_p50(rgb: np.ndarray) -> float:
    """Median HSV saturation over all pixels."""
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    return float(np.percentile(sat, 50))


@dataclass(frozen=True)
class QualityFeatures:
    blur: float
    color_cast: float
    luminance: float
    saturation_p50: float


def compute_features(image_bytes: bytes) -> QualityFeatures:
    """The four cheap quality features for one image."""
    rgb = _rgb_array(image_bytes)
    return QualityFeatures(
        blur=laplacian_variance(rgb),
        color_cast=color_cast_score(rgb),
        luminance=mean_luminance(rgb),
        saturation_p50=saturation_p50(rgb),
    )


@dataclass(frozen=True)
class BleachGate:
    """Tuned quality gate for bleach-direction confident-learning flags.

    A flag toward BLEACHED/UNHEALTHY is kept only if ``passes`` is true: the image
    clears every quality floor/ceiling AND the probe margin (``suggested_prob -
    self_confidence``) is at least ``margin_min``. ``enabled=False`` means the CV
    objective (precision >= 60% at recall >= 75% of true bleach-direction flags) was
    not reached on the audited set — the gate exists but must not be applied by
    default; every consumer must check ``enabled`` before filtering.
    """

    blur_min: float
    color_cast_max: float
    luminance_min: float
    luminance_max: float
    margin_min: float
    enabled: bool = False

    def passes(self, features: QualityFeatures, margin: float) -> bool:
        return (
            features.blur >= self.blur_min
            and features.color_cast <= self.color_cast_max
            and self.luminance_min <= features.luminance <= self.luminance_max
            and margin >= self.margin_min
        )


def grid_search(
    features: list[QualityFeatures],
    margins: np.ndarray,
    labels: np.ndarray,
    *,
    min_recall: float = 0.75,
) -> tuple[BleachGate, float, float]:
    """Best (precision, recall) threshold combo over a percentile grid.

    ``labels`` is 1 for a confirmed-correct bleach-direction flag, 0 otherwise. Returns
    ``(gate, precision, recall)`` for the combo maximising precision subject to
    ``recall >= min_recall``; ``enabled`` is left ``False`` (the caller decides).
    """
    blur = np.array([f.blur for f in features])
    cast = np.array([f.color_cast for f in features])
    lum = np.array([f.luminance for f in features])
    pct = (0, 10, 25, 40, 50)
    blur_grid = [0.0, *np.percentile(blur, pct).tolist()]
    cast_grid = [*np.percentile(cast, [50, 60, 75, 90, 100]).tolist()]
    lum_lo_grid = [0.0, *np.percentile(lum, [5, 15, 25]).tolist()]
    lum_hi_grid = [*np.percentile(lum, [75, 85, 95]).tolist(), 1.0]
    margin_grid = np.percentile(margins, [0, 10, 25, 40, 50, 60]).tolist()

    n_pos = int(labels.sum())
    best: tuple[BleachGate, float, float] | None = None
    for bmin in blur_grid:
        for cmax in cast_grid:
            for llo in lum_lo_grid:
                for lhi in lum_hi_grid:
                    if lhi <= llo:
                        continue
                    keep_quality = (blur >= bmin) & (cast <= cmax) & (lum >= llo) & (lum <= lhi)
                    for mmin in margin_grid:
                        keep = keep_quality & (margins >= mmin)
                        kept_n = int(keep.sum())
                        if kept_n == 0:
                            continue
                        correct = int((labels[keep] == 1).sum())
                        precision = correct / kept_n
                        recall = correct / n_pos if n_pos else 0.0
                        if recall < min_recall:
                            continue
                        if (
                            best is None
                            or precision > best[1]
                            or (precision == best[1] and recall > best[2])
                        ):
                            gate = BleachGate(bmin, cmax, llo, lhi, mmin, enabled=False)
                            best = (gate, precision, recall)
    if best is None:
        # No combo clears the recall floor: the maximally permissive gate (pass-through).
        gate = BleachGate(0.0, float("inf"), 0.0, 1.0, float(margins.min()), enabled=False)
        precision = float(labels.mean()) if len(labels) else 0.0
        return gate, precision, 1.0
    return best


# Tuned on the 85 toward-BLEACHED/UNHEALTHY decided rows of ``model-audit.tsv`` (2026-09-25),
# fit on the full set (:func:`grid_search`, ``min_recall=0.75``): train precision 43.1% at
# recall 80.6%. Grouped 5-fold CV (each image its own group; see ``docs/label-quality.md``)
# gives an out-of-fold precision of 35.2% at recall 61.3% — *below* the un-gated baseline of
# 36.5% precision. Blur, colour cast and luminance do not separate confirmed-correct
# bleach-direction flags from confirmed-wrong ones on this sample: the CL probe's mistakes
# are not explained by these four cheap features. Precision never reaches the 60% floor, so
# the gate is tuned but shipped **off** (``enabled=False``); no consumer should turn it on
# without re-tuning on a larger audited set.
TUNED_GATE = BleachGate(
    blur_min=0.0,
    color_cast_max=0.446835458278656,
    luminance_min=0.29486014246940606,
    luminance_max=1.0,
    margin_min=0.5642041161046708,
    enabled=False,
)
