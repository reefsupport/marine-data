"""WP-5e: the second-stage face verifier (D-I2).

Pure-function tests only (no ONNX weights needed) — mirrors ``test_privacy.py``'s
split between logic tests here and weight-dependent behaviour gated behind
``MARINEDATA_INTEGRATION`` (see ``test_integration.py``).
"""

from __future__ import annotations

from marinedata.privacy.verify import (
    VERIFY_THRESHOLD,
    VerifyResult,
    expand_box,
    fit_threshold_for_min_recall,
    recall_precision_at_threshold,
)


def test_expand_box_pads_and_clamps_to_image() -> None:
    box = {"x": 10, "y": 10, "w": 20, "h": 20, "score": 0.9}
    x0, y0, x1, y1 = expand_box(box, img_w=100, img_h=100, pad_frac=0.5)
    assert (x0, y0, x1, y1) == (0, 0, 40, 40)  # 10 - 10 clamped to 0; 30 + 10


def test_expand_box_clamps_at_far_edge() -> None:
    box = {"x": 90, "y": 90, "w": 20, "h": 20, "score": 0.9}
    _x0, _y0, x1, y1 = expand_box(box, img_w=100, img_h=100, pad_frac=0.5)
    assert x1 == 100 and y1 == 100


def test_verify_result_verified_uses_module_threshold() -> None:
    below = VerifyResult(score=VERIFY_THRESHOLD - 0.01, occluded=False)
    at = VerifyResult(score=VERIFY_THRESHOLD, occluded=False)
    assert below.verified is False
    assert at.verified is True


def test_recall_precision_at_threshold_basic() -> None:
    # 3 true faces, verifier scores two of them >= threshold; 1 false positive
    # also clears the threshold.
    scores = [0.9, 0.6, 0.1, 0.7]
    truth = [True, True, True, False]
    recall, precision, n_true = recall_precision_at_threshold(scores, truth, threshold=0.5)
    assert n_true == 3
    assert recall == 2 / 3
    assert precision == 2 / 3  # tp=2 of 3 predicted positive (0.9, 0.6, 0.7 >= 0.5)


def test_recall_precision_at_threshold_no_true_positives_is_zero_not_nan() -> None:
    recall, precision, n_true = recall_precision_at_threshold([0.1, 0.2], [False, False], 0.5)
    assert n_true == 0
    assert recall == 0.0
    assert precision == 0.0


def test_recall_precision_at_threshold_length_mismatch_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        recall_precision_at_threshold([0.1], [True, False], 0.5)


def test_fit_threshold_for_min_recall_picks_highest_qualifying_threshold() -> None:
    # 4 true faces at 0.9/0.6/0.4/0.2; recall>=75% needs the top 3, i.e. threshold=0.4.
    scores = [0.9, 0.6, 0.4, 0.2, 0.95, 0.05]
    truth = [True, True, True, True, False, False]
    t = fit_threshold_for_min_recall(scores, truth, min_recall=0.75)
    assert t == 0.4


def test_fit_threshold_for_min_recall_full_recall_needs_lowest_true_score() -> None:
    scores = [0.9, 0.6, 0.4]
    truth = [True, True, True]
    assert fit_threshold_for_min_recall(scores, truth, min_recall=1.0) == 0.4


def test_fit_threshold_for_min_recall_returns_none_with_no_positives() -> None:
    assert fit_threshold_for_min_recall([0.9, 0.1], [False, False], min_recall=0.95) is None
    assert fit_threshold_for_min_recall([], [], min_recall=0.95) is None
