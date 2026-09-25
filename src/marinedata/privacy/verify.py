"""Second-stage face verifier for the D-I2 blur decision (WP-5e).

The first-stage scan (:mod:`.scan`) runs YuNet once, over the *whole* image, at
whatever scale the source photo happens to be. That is a screening pass, not a
release decision — the D-I audit (WP-5d, ``docs/privacy-audit-2026-09-25.tsv``)
measured only 18/150 = 12.0% of face-flagged images as a real face at all.

This module re-runs the same open-weight detector (no new model, no new
licence surface) a second time, but on an EXPANDED CROP around the first-stage
box (``expand_box``, default 50% padding on each side) instead of the full
frame. Two independent effects make the second pass discriminate better than
just re-scoring the same box:

- A small, real face inside a large low-resolution reef photo is scored
  against the whole frame's scale on pass one; cropped and re-fed on pass two,
  YuNet sees it at a size close to its training distribution and typically
  scores it higher.
- A coral-texture / scan-line false positive usually depended on surrounding
  context (a repeating polyp pattern, a chromatic-aberration edge that runs
  off the crop) that vanishes once the frame is cut down to just the padded
  box — the second pass frequently finds nothing at all there.

``FaceVerifier.verify(...)`` returns the best detection score inside the crop
(0.0 if YuNet finds nothing), plus whether ``_looks_occluded`` fires on the
crop's own landmarks. :data:`VERIFY_THRESHOLD` is the score at/above which a
candidate counts as verifier-confirmed for the D-I2 blur decision.

**Threshold provenance (WP-5f, fully fit)**: the D-I audit was extended with
120+ additional stratified candidates (WP-5f, ``docs/privacy-audit-2026-09-25.tsv``),
bringing the audited-true-face count to **26** (>= the brief's 25-positive
floor). :data:`VERIFY_THRESHOLD` is the highest score that still recalls
>= 95% of those 26 (:func:`fit_threshold_for_min_recall`): recall 96.2%
(25/26, Wilson 95% CI [81.1%, 99.3%]), precision 12.3% — precision is below
the brief's 20% floor, so the policy explicitly accepts that trade (blur
every candidate the second stage scores at/above this lower bound) rather
than raise the threshold and miss a real face. See "D-I2" in
``docs/PRIVACY.md`` for the count this blurs across the full corpus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scan import _looks_occluded

VERIFY_MODEL_VERSION = "yunet_2023mar_2ndpass"
# Fit on 26 audited true faces (WP-5f, >= the brief's 25-positive floor): the
# highest score that keeps recall >= 95% on those 26 (fit_threshold_for_min_recall).
# Precision at this threshold is 12.3%, below the brief's 20% floor -- kept anyway
# because raising it would drop recall below 95% (see docs/PRIVACY.md, D-I2).
VERIFY_THRESHOLD = 0.3292


@dataclass(frozen=True)
class VerifyResult:
    score: float
    occluded: bool

    @property
    def verified(self) -> bool:
        return self.score >= VERIFY_THRESHOLD


def expand_box(
    box: Mapping[str, float], img_w: int, img_h: int, pad_frac: float = 0.5
) -> tuple[int, int, int, int]:
    """Pixel crop ``(x0, y0, x1, y1)`` for ``box`` padded ``pad_frac`` per side, clamped."""
    bx, by, bw, bh = float(box["x"]), float(box["y"]), float(box["w"]), float(box["h"])
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x0 = max(0, round(bx - pad_x))
    y0 = max(0, round(by - pad_y))
    x1 = min(img_w, round(bx + bw + pad_x))
    y1 = min(img_h, round(by + bh + pad_y))
    return x0, y0, x1, y1


class FaceVerifier:
    """Wraps a second ``cv2.FaceDetectorYN`` instance, run on expanded crops only."""

    def __init__(self, weights_path: str | Path, score_threshold: float = 0.3) -> None:
        import cv2

        self._cv2 = cv2
        # Low internal threshold: we want every candidate response back so the
        # *release* decision can apply VERIFY_THRESHOLD ourselves, not have
        # cv2 silently drop borderline ones before we see them.
        self._model = cv2.FaceDetectorYN.create(
            str(weights_path), "", (0, 0), score_threshold, 0.3, 5000
        )

    def verify(
        self, image_bgr: Any, box: Mapping[str, float], pad_frac: float = 0.5
    ) -> VerifyResult:
        """Best detection inside ``box`` expanded ``pad_frac``; ``score=0.0`` if none fires."""
        h, w = image_bgr.shape[:2]
        x0, y0, x1, y1 = expand_box(box, w, h, pad_frac=pad_frac)
        if x1 <= x0 or y1 <= y0:
            return VerifyResult(score=0.0, occluded=False)
        crop = image_bgr[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]
        self._model.setInputSize((cw, ch))
        _, faces = self._model.detect(crop)
        if faces is None or len(faces) == 0:
            return VerifyResult(score=0.0, occluded=False)
        best = max(faces, key=lambda row: row[14])
        score = float(best[14])
        landmarks = [(float(best[4 + 2 * i]), float(best[5 + 2 * i])) for i in range(5)]
        return VerifyResult(score=score, occluded=_looks_occluded(landmarks))


def recall_precision_at_threshold(
    scores: Sequence[float], truth: Sequence[bool], threshold: float
) -> tuple[float, float, int]:
    """``(recall, precision, n_true)`` of ``score >= threshold`` against ``truth``.

    ``scores`` and ``truth`` are paired sequences. ``n_true`` is
    ``sum(truth)`` — pass it and the true-positive count to
    ``audit_stats.wilson_ci`` for the recall confidence interval the brief
    asks for.
    """
    if len(scores) != len(truth):
        raise ValueError("scores and truth must be the same length")
    n_true = sum(1 for t in truth if t)
    tp = sum(1 for s, t in zip(scores, truth, strict=True) if t and s >= threshold)
    predicted_positive = sum(1 for s in scores if s >= threshold)
    recall = tp / n_true if n_true else 0.0
    precision = tp / predicted_positive if predicted_positive else 0.0
    return recall, precision, n_true


def fit_threshold_for_min_recall(
    scores: Sequence[float], truth: Sequence[bool], min_recall: float = 0.95
) -> float | None:
    """The highest threshold in ``scores`` whose recall on ``truth`` is >= ``min_recall``.

    Scans candidate thresholds from highest to lowest score so the first one
    to qualify is the tightest (highest-precision) threshold that still meets
    the recall floor. Returns ``None`` if no threshold in ``scores`` reaches
    ``min_recall`` (including when ``truth`` has no positives at all).
    """
    for t in sorted(set(scores), reverse=True):
        recall, _precision, n_true = recall_precision_at_threshold(scores, truth, t)
        if n_true and recall >= min_recall:
            return t
    return None
