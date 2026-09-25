"""Metric registry (design doc §4.2).

``METRICS`` maps a metric name to a :class:`Metric` describing which task types it
applies to, the callable that computes it, and whether higher is better. Every
function here takes plain ``numpy`` arrays (or, for ``coco_map``, COCO-shaped dicts)
so that :mod:`marinedata.eval.bootstrap` can call them on arbitrary row subsets without
knowing anything about the eval harness's file formats.

Golden-value tests (``tests/test_eval_metrics.py``) check each function against
``sklearn`` / ``pycocotools`` (or a hand-computed fixture where no reference library
applies) to within 1e-6.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Metric:
    """One entry in :data:`METRICS`."""

    task_types: tuple[str, ...]
    fn: Callable[..., Any]
    higher_is_better: bool
    unit: str = "pt"


# ---------------------------------------------------------------------------
# cls: macro-F1
# ---------------------------------------------------------------------------


def macro_f1(
    y_true: Sequence[Any], y_pred: Sequence[Any], *, labels: Sequence[Any] | None = None
) -> float:
    """Unweighted mean F1 over classes with >= 1 GT sample in the evaluated split.

    A prediction naming a class absent from ``y_true`` (``labels``) never becomes a
    false positive for any class in ``labels`` — it just costs the true class its
    recall — matching ``sklearn.metrics.f1_score(y_true, y_pred, average="macro",
    labels=present)`` exactly (see design §4.2).
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true_arr.tolist()))
    if not labels:
        return float("nan")
    scores = []
    for cls in labels:
        tp = int(np.sum((y_true_arr == cls) & (y_pred_arr == cls)))
        fn = int(np.sum((y_true_arr == cls) & (y_pred_arr != cls)))
        fp = int(np.sum((y_true_arr != cls) & (y_pred_arr == cls)))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# sem-seg: mIoU
# ---------------------------------------------------------------------------


def miou(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int = 255,
) -> float:
    """Dataset-level mIoU: one confusion matrix accumulated over every pixel passed in
    (concatenate all images first — this is *not* a per-image mean), classes whose
    union of GT+pred pixels is 0 dropped from the average.
    """
    y_true_arr = np.asarray(y_true).reshape(-1)
    y_pred_arr = np.asarray(y_pred).reshape(-1)
    keep = y_true_arr != ignore_index
    y_true_arr = y_true_arr[keep]
    y_pred_arr = y_pred_arr[keep]
    valid_pred = (y_pred_arr >= 0) & (y_pred_arr < num_classes)
    y_true_arr = y_true_arr[valid_pred]
    y_pred_arr = y_pred_arr[valid_pred]
    idx = y_true_arr.astype(np.int64) * num_classes + y_pred_arr.astype(np.int64)
    cm = np.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    ious = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        union = tp + fp + fn
        if union == 0:
            continue
        ious.append(tp / union)
    return float(np.mean(ious)) if ious else float("nan")


# ---------------------------------------------------------------------------
# det / inst-seg: COCO mAP@[.5:.95]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CocoMapResult:
    map: float
    map50: float


def coco_map(
    gt_coco: dict[str, Any],
    pred_results: list[dict[str, Any]],
    *,
    iou_type: str = "bbox",
) -> CocoMapResult:
    """COCO mAP@[.5:.95] (+ mAP@.5 alongside), 101-point interpolation, maxDets 100 —
    a thin wrapper over ``pycocotools.cocoeval.COCOeval`` (design §4.2): ``stats[0]``
    is AP@[.5:.95] maxDets=100, ``stats[1]`` is AP@.5.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = gt_coco
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(pred_results) if pred_results else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return CocoMapResult(map=float(ev.stats[0]), map50=float(ev.stats[1]))


# ---------------------------------------------------------------------------
# points: micro top-1 accuracy (+ macro recall alongside)
# ---------------------------------------------------------------------------


def point_acc(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    """Micro top-1 accuracy over points."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if len(y_true_arr) == 0:
        return float("nan")
    return float(np.mean(y_true_arr == y_pred_arr))


def point_macro_recall(
    y_true: Sequence[Any], y_pred: Sequence[Any], *, labels: Sequence[Any] | None = None
) -> float:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if labels is None:
        labels = sorted(set(y_true_arr.tolist()))
    if not labels:
        return float("nan")
    recalls = []
    for cls in labels:
        total = int(np.sum(y_true_arr == cls))
        tp = int(np.sum((y_true_arr == cls) & (y_pred_arr == cls)))
        recalls.append((tp / total) if total else 0.0)
    return float(np.mean(recalls))


# ---------------------------------------------------------------------------
# vqa: exact-match accuracy after normalisation
# ---------------------------------------------------------------------------

_ARTICLES = frozenset({"a", "an", "the"})
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalise_vqa_answer(answer: str) -> str:
    """Lowercase, strip punctuation and articles, number words -> digits (design §4.2)."""
    text = _PUNCT_RE.sub("", answer.lower().strip())
    tokens = [t for t in text.split() if t not in _ARTICLES]
    tokens = [_NUMBER_WORDS.get(t, t) for t in tokens]
    return " ".join(tokens)


def vqa_acc(
    pred_answers: Sequence[str],
    gt_answers: Sequence[str | Sequence[str]],
    *,
    option_letters: Sequence[str | None] | None = None,
) -> float:
    """Exact match after normalisation; a multiple-choice item matches on the option
    letter instead (``option_letters[i]`` set, compared against the first character of
    ``pred_answers[i]``).
    """
    n = len(pred_answers)
    if n == 0:
        return float("nan")
    correct = 0
    for i in range(n):
        refs = gt_answers[i]
        refs_seq = [refs] if isinstance(refs, str) else list(refs)
        if option_letters is not None and option_letters[i] is not None:
            letter = str(pred_answers[i]).strip().lower()[:1]
            if letter == str(option_letters[i]).strip().lower():
                correct += 1
                continue
        pred_norm = normalise_vqa_answer(str(pred_answers[i]))
        refs_norm = {normalise_vqa_answer(str(r)) for r in refs_seq}
        if pred_norm in refs_norm:
            correct += 1
    return correct / n


# ---------------------------------------------------------------------------
# cls / points: Expected Calibration Error
# ---------------------------------------------------------------------------


def ece(
    probs: Sequence[Sequence[float]] | np.ndarray, y_true: Sequence[Any], *, n_bins: int = 15
) -> float:
    """15 equal-width bins on max-softmax confidence, L1-weighted by bin occupancy
    (design §4.2)."""
    probs_arr = np.asarray(probs, dtype=np.float64)
    y_true_arr = np.asarray(y_true)
    if probs_arr.ndim != 2 or len(probs_arr) == 0:
        return float("nan")
    confidences = probs_arr.max(axis=1)
    predictions = probs_arr.argmax(axis=1)
    accuracies = (predictions == y_true_arr).astype(np.float64)
    n = len(y_true_arr)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in itertools.pairwise(edges):
        last_bin = hi == edges[-1]
        in_bin = (confidences >= lo) & (confidences <= hi if last_bin else confidences < hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        bin_acc = float(accuracies[in_bin].mean())
        bin_conf = float(confidences[in_bin].mean())
        total += (count / n) * abs(bin_acc - bin_conf)
    return float(total)


METRICS: dict[str, Metric] = {
    "macro_f1": Metric(task_types=("cls",), fn=macro_f1, higher_is_better=True),
    "miou": Metric(task_types=("sem-seg",), fn=miou, higher_is_better=True),
    "map": Metric(task_types=("det", "inst-seg"), fn=coco_map, higher_is_better=True),
    "point_acc": Metric(task_types=("points",), fn=point_acc, higher_is_better=True),
    "point_macro_recall": Metric(
        task_types=("points",), fn=point_macro_recall, higher_is_better=True
    ),
    "vqa_acc": Metric(task_types=("vqa",), fn=vqa_acc, higher_is_better=True),
    "ece": Metric(task_types=("cls", "points"), fn=ece, higher_is_better=False),
}
