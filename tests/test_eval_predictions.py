"""Prediction readers (design §4.1) for each declared shape."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marinedata.eval.predictions import (
    PredictionFormatError,
    read_cls_predictions,
    read_coco_predictions,
    read_predictions,
    read_seg_predictions,
    read_vqa_predictions,
)


def test_read_cls_predictions_hard_labels(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame({"image_sha256": ["a", "b"], "pred_label": [0, 1]}).to_parquet(path)
    preds = read_cls_predictions(path)
    assert list(preds.image_sha256) == ["a", "b"]
    assert list(preds.pred_label) == [0, 1]
    assert preds.pred_probs is None
    assert preds.point_id is None


def test_read_cls_predictions_probs_only_derives_label(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame(
        {"image_sha256": ["a", "b"], "pred_probs": [[0.1, 0.9], [0.8, 0.2]]}
    ).to_parquet(path)
    preds = read_cls_predictions(path)
    assert list(preds.pred_label) == [1, 0]
    assert preds.pred_probs.shape == (2, 2)


def test_read_points_predictions_requires_point_id(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame({"image_sha256": ["a"], "pred_label": [1]}).to_parquet(path)
    with pytest.raises(PredictionFormatError):
        read_cls_predictions(path, task_type="points")


def test_read_points_predictions_ok(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame(
        {"image_sha256": ["a", "a"], "point_id": [0, 1], "pred_label": [1, 2]}
    ).to_parquet(path)
    preds = read_cls_predictions(path, task_type="points")
    assert list(preds.point_id) == [0, 1]


def test_read_seg_predictions(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame(
        {"image_sha256": ["a", "b"], "pred_mask_png": ["a.png", "b.png"]}
    ).to_parquet(path)
    preds = read_seg_predictions(path)
    assert preds.as_dict() == {"a": "a.png", "b": "b.png"}


def test_read_seg_predictions_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame({"image_sha256": ["a"]}).to_parquet(path)
    with pytest.raises(PredictionFormatError):
        read_seg_predictions(path)


def test_read_coco_predictions(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    rows = [{"image_id": 1, "category_id": 1, "score": 0.9, "bbox": [0, 0, 1, 1]}]
    path.write_text(json.dumps(rows))
    assert read_coco_predictions(path) == rows


def test_read_coco_predictions_rejects_non_list(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(PredictionFormatError):
        read_coco_predictions(path)


def test_read_coco_predictions_rejects_missing_fields(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(json.dumps([{"image_id": 1}]))
    with pytest.raises(PredictionFormatError):
        read_coco_predictions(path)


def test_read_vqa_predictions(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame({"image_sha256": ["a"], "pred_answer": ["coral"]}).to_parquet(path)
    preds = read_vqa_predictions(path)
    assert preds.pred_answer == ("coral",)


def test_read_predictions_dispatch(tmp_path: Path) -> None:
    cls_path = tmp_path / "cls.parquet"
    pd.DataFrame({"image_sha256": ["a"], "pred_label": [0]}).to_parquet(cls_path)
    assert read_predictions(cls_path, "cls").pred_label[0] == 0

    vqa_path = tmp_path / "vqa.parquet"
    pd.DataFrame({"image_sha256": ["a"], "pred_answer": ["x"]}).to_parquet(vqa_path)
    assert read_predictions(vqa_path, "vqa").pred_answer == ("x",)

    with pytest.raises(PredictionFormatError):
        read_predictions(cls_path, "not-a-task-type")


def test_read_cls_predictions_probs_dtype(tmp_path: Path) -> None:
    path = tmp_path / "preds.parquet"
    pd.DataFrame(
        {"image_sha256": ["a"], "pred_probs": [np.array([0.2, 0.8], dtype=np.float32)]}
    ).to_parquet(path)
    preds = read_cls_predictions(path)
    assert preds.pred_probs.dtype == np.float64
