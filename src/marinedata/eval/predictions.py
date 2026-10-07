"""Prediction readers (design §4.1).

``eval score --pred <preds.parquet>`` accepts one of five shapes, keyed by the task's
``task_type``:

- ``cls`` / ``points``: parquet with ``image_sha256`` + ``pred_label`` and/or
  ``pred_probs`` (list<float>, one row per class); ``points`` rows also carry
  ``point_id`` (one point per row, several points per image).
- ``sem-seg``: parquet with ``image_sha256`` + ``pred_mask_png`` (a path to a
  single-channel PNG the same H×W as the ground-truth mask).
- ``det`` / ``inst-seg``: a COCO results JSON — a flat list of
  ``{"image_id", "category_id", "score", "bbox"|"segmentation"}`` rows, exactly the
  shape ``pycocotools.coco.COCO.loadRes`` expects.
- ``vqa``: parquet with ``image_sha256`` + ``pred_answer`` (a free-text string).

Readers here only *load* the file into a plain, task-typed structure; joining against
ground truth and computing metrics is :mod:`marinedata.eval.cli`'s job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLS_TASK_TYPES = ("cls",)
POINTS_TASK_TYPES = ("points",)
SEG_TASK_TYPES = ("sem-seg",)
COCO_TASK_TYPES = ("det", "inst-seg")
VQA_TASK_TYPES = ("vqa",)


class PredictionFormatError(ValueError):
    """A predictions file is missing a column its task type requires."""


def _require_columns(df: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise PredictionFormatError(f"{path}: missing column(s) {missing}")


@dataclass(frozen=True)
class ClsPredictions:
    """One row per (image, [point]). ``pred_probs`` is ``(n, k)`` or ``None`` if the
    file only carries hard labels."""

    image_sha256: np.ndarray
    pred_label: np.ndarray
    pred_probs: np.ndarray | None
    point_id: np.ndarray | None = None


def read_cls_predictions(path: str | Path, *, task_type: str = "cls") -> ClsPredictions:
    """Read ``cls`` or ``points`` predictions. ``points`` requires ``point_id``."""
    path = Path(path)
    df = pd.read_parquet(path)
    _require_columns(df, ("image_sha256",), path)
    if "pred_label" not in df.columns and "pred_probs" not in df.columns:
        raise PredictionFormatError(f"{path}: needs pred_label and/or pred_probs")
    pred_probs = None
    if "pred_probs" in df.columns:
        pred_probs = np.stack([np.asarray(row, dtype=np.float64) for row in df["pred_probs"]])
    if "pred_label" in df.columns:
        pred_label = df["pred_label"].to_numpy()
    else:
        pred_label = pred_probs.argmax(axis=1)
    point_id = None
    if task_type in POINTS_TASK_TYPES:
        _require_columns(df, ("point_id",), path)
        point_id = df["point_id"].to_numpy()
    return ClsPredictions(
        image_sha256=df["image_sha256"].to_numpy(),
        pred_label=pred_label,
        pred_probs=pred_probs,
        point_id=point_id,
    )


@dataclass(frozen=True)
class SegPredictions:
    """``image_sha256 -> pred_mask_png`` path, one row per image."""

    image_sha256: tuple[str, ...]
    pred_mask_png: tuple[str, ...]

    def as_dict(self) -> dict[str, str]:
        return dict(zip(self.image_sha256, self.pred_mask_png, strict=True))


def read_seg_predictions(path: str | Path) -> SegPredictions:
    path = Path(path)
    df = pd.read_parquet(path)
    _require_columns(df, ("image_sha256", "pred_mask_png"), path)
    return SegPredictions(
        image_sha256=tuple(df["image_sha256"].astype(str)),
        pred_mask_png=tuple(df["pred_mask_png"].astype(str)),
    )


def read_coco_predictions(path: str | Path) -> list[dict]:
    """A COCO results JSON: a flat list of per-detection dicts, ``pycocotools``-ready."""
    path = Path(path)
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise PredictionFormatError(f"{path}: COCO results must be a JSON list of dicts")
    required = ("image_id", "category_id", "score")
    for i, row in enumerate(data):
        missing = [c for c in required if c not in row]
        if missing:
            raise PredictionFormatError(f"{path}: row {i} missing {missing}")
    return data


@dataclass(frozen=True)
class VqaPredictions:
    image_sha256: tuple[str, ...]
    pred_answer: tuple[str, ...]


def read_vqa_predictions(path: str | Path) -> VqaPredictions:
    path = Path(path)
    df = pd.read_parquet(path)
    _require_columns(df, ("image_sha256", "pred_answer"), path)
    return VqaPredictions(
        image_sha256=tuple(df["image_sha256"].astype(str)),
        pred_answer=tuple(df["pred_answer"].astype(str)),
    )


def read_predictions(
    path: str | Path, task_type: str
) -> ClsPredictions | SegPredictions | list[dict] | VqaPredictions:
    """Dispatch on ``task_type`` to the reader whose shape matches (design §4.1)."""
    if task_type in CLS_TASK_TYPES or task_type in POINTS_TASK_TYPES:
        return read_cls_predictions(path, task_type=task_type)
    if task_type in SEG_TASK_TYPES:
        return read_seg_predictions(path)
    if task_type in COCO_TASK_TYPES:
        return read_coco_predictions(path)
    if task_type in VQA_TASK_TYPES:
        return read_vqa_predictions(path)
    raise PredictionFormatError(f"unknown task_type {task_type!r}")
