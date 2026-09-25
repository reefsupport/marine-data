"""``marinedata eval score`` end to end for the ``cls`` task type — proves the one-line
wiring into :mod:`marinedata.cli` (design §4.1, P4 acceptance)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from marinedata.cli import main


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    gt = pd.DataFrame(
        {
            "image_sha256": [f"img{i}" for i in range(12)],
            "split_group": [f"g{i % 4}" for i in range(12)],
            "split": ["test"] * 12,
            "label": [i % 2 for i in range(12)],
        }
    )
    gt_path = tmp_path / "gt.parquet"
    gt.to_parquet(gt_path)

    pred = pd.DataFrame(
        {
            "image_sha256": [f"img{i}" for i in range(12)],
            "pred_label": [i % 2 for i in range(12)],
            "pred_probs": [[0.1, 0.9] if i % 2 else [0.9, 0.1] for i in range(12)],
        }
    )
    pred_path = tmp_path / "pred.parquet"
    pred.to_parquet(pred_path)
    return gt_path, pred_path


def test_eval_score_cls_end_to_end(tmp_path: Path) -> None:
    gt_path, pred_path = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = main(
        [
            "eval",
            "score",
            "--release",
            "v2-fixture",
            "--task",
            "coral-health-binary",
            "--task-type",
            "cls",
            "--split",
            "test",
            "--pred",
            str(pred_path),
            "--gt",
            str(gt_path),
            "--boot",
            "50",
            "--out",
            str(out_dir),
        ]
    )
    assert rc == 0
    metrics = json.loads((out_dir / "metrics.json").read_text())
    assert metrics["n_images"] == 12
    assert metrics["metrics"]["macro_f1"]["point"] == 1.0
    # every prediction is correct but only 90% confident -> mis-calibrated, ECE = 0.1
    assert metrics["metrics"]["ece"]["point"] == pytest.approx(0.1)
    assert (out_dir / "scores.parquet").exists()
