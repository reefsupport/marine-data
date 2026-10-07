"""Golden-value tests for :mod:`marinedata.eval.metrics` (design §4.2): each metric
against its reference library (sklearn / pycocotools) or, where none applies
(vqa_acc, ece), a hand-computed fixture — within 1e-6.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from marinedata.eval.metrics import (
    coco_map,
    ece,
    macro_f1,
    miou,
    normalise_vqa_answer,
    point_acc,
    point_macro_recall,
    vqa_acc,
)

TOL = 1e-6


def test_macro_f1_matches_sklearn() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 5, size=200)
    y_pred = rng.integers(0, 6, size=200)  # class 5 never appears in y_true
    present = sorted(set(y_true.tolist()))
    expected = sklearn.f1_score(y_true, y_pred, average="macro", labels=present)
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, abs=TOL)


def test_macro_f1_absent_predicted_class_only_costs_recall() -> None:
    # 2 GT classes (0, 1); model predicts an absent class 9 for one class-0 sample.
    y_true = [0, 0, 1, 1]
    y_pred = [0, 9, 1, 1]
    # class 0: tp=1 fn=1 fp=0 -> f1 = 2/3 ; class 1: tp=2 fn=0 fp=0 -> f1=1.0
    assert macro_f1(y_true, y_pred) == pytest.approx((2 / 3 + 1.0) / 2, abs=TOL)


def test_miou_matches_hand_computed_and_sklearn_jaccard() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    y_true = np.array([0, 0, 1, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 0])
    got = miou(y_true, y_pred, num_classes=3)
    per_class = sklearn.jaccard_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    assert got == pytest.approx(float(np.mean(per_class)), abs=TOL)


def test_miou_drops_zero_union_classes() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    # class 2 has 0 union (never in GT or pred) and must not drag the mean to <1.0
    assert miou(y_true, y_pred, num_classes=3) == pytest.approx(1.0, abs=TOL)


def test_miou_ignore_index_excluded() -> None:
    y_true = np.array([0, 0, 255, 1])
    y_pred = np.array([0, 1, 1, 1])
    # ignore_index row dropped -> remaining y_true=[0,0,1], y_pred=[0,1,1]:
    # class0 tp=1 fp=0 fn=1 -> iou .5; class1 tp=1 fp=1 fn=0 -> iou .5
    assert miou(y_true, y_pred, num_classes=2, ignore_index=255) == pytest.approx(0.5, abs=TOL)


def _tiny_coco_fixture() -> tuple[dict, list[dict]]:
    gt = {
        "images": [{"id": 1, "width": 10, "height": 10}, {"id": 2, "width": 10, "height": 10}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 4, 4],
                "area": 16,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [1, 1, 4, 4],
                "area": 16,
                "iscrowd": 0,
            },
        ],
        "categories": [{"id": 1, "name": "coral"}],
    }
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [1, 1, 4, 4], "score": 0.8},
    ]
    return gt, preds


def test_coco_map_matches_pycocotools_directly() -> None:
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt, preds = _tiny_coco_fixture()
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds)
    ev = COCOeval(coco_gt, coco_dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()

    result = coco_map(gt, preds)
    assert result.map == pytest.approx(float(ev.stats[0]), abs=TOL)
    assert result.map50 == pytest.approx(float(ev.stats[1]), abs=TOL)
    # Perfect boxes at every IoU threshold -> AP == 1.0
    assert result.map == pytest.approx(1.0, abs=TOL)


def test_point_acc_matches_sklearn_accuracy() -> None:
    sklearn = pytest.importorskip("sklearn.metrics")
    y_true = [0, 1, 1, 2, 2, 2]
    y_pred = [0, 1, 0, 2, 2, 1]
    assert point_acc(y_true, y_pred) == pytest.approx(
        sklearn.accuracy_score(y_true, y_pred), abs=TOL
    )


def test_point_macro_recall_hand_computed() -> None:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    # class0 recall .5, class1 recall 1.0 -> mean .75
    assert point_macro_recall(y_true, y_pred) == pytest.approx(0.75, abs=TOL)


def test_normalise_vqa_answer() -> None:
    assert normalise_vqa_answer("A Coral!") == "coral"
    assert normalise_vqa_answer("two fish.") == "2 fish"


def test_vqa_acc_hand_computed() -> None:
    preds = ["a coral", "Two fish", "brain coral"]
    gts = ["coral", ["fish", "two fish"], "staghorn coral"]
    assert vqa_acc(preds, gts) == pytest.approx(2 / 3, abs=TOL)


def test_vqa_acc_multiple_choice_matches_letter() -> None:
    preds = ["B) staghorn", "a"]
    gts = ["staghorn coral", "brain coral"]
    letters = ["b", "a"]
    assert vqa_acc(preds, gts, option_letters=letters) == pytest.approx(1.0, abs=TOL)


def test_ece_hand_computed_two_bins() -> None:
    # 4 samples, perfectly calibrated within two coarse bins.
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.4, 0.6]])
    y_true = np.array([0, 1, 0, 1])
    got = ece(probs, y_true, n_bins=15)
    # Hand roll the same 15-equal-width-bin, L1-weighted computation independently.
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    acc = (preds == y_true).astype(float)
    edges = np.linspace(0, 1, 16)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        last = hi == edges[-1]
        mask = (confidences >= lo) & (confidences <= hi if last else confidences < hi)
        if mask.sum() == 0:
            continue
        total += (mask.sum() / len(y_true)) * abs(acc[mask].mean() - confidences[mask].mean())
    assert got == pytest.approx(total, abs=TOL)


def test_ece_perfect_calibration_is_zero() -> None:
    # All confidences land in one bin and accuracy matches it exactly.
    probs = np.array([[0.5, 0.5]] * 4)
    y_true = np.array([0, 1, 0, 1])
    assert ece(probs, y_true) == pytest.approx(0.0, abs=TOL)
