"""``marinedata eval`` sub-command (design §4.1). Split out of :mod:`marinedata.cli`
the same way :mod:`marinedata.cli_ingest` is — ``cli.py`` imports and wires
:func:`add_eval_subparser` from here (its one call line, disclosed in the P4 report).

Only ``score`` is implemented here; ``baseline``, ``reproduce`` and ``table`` are P5.

**Ground-truth contract.** P3 (split-v2) and P1 (benchmark manifests) are not merged
yet, so ``--gt`` is an explicit flag rather than derived from ``--release``: a parquet
(cls/points/sem-seg/vqa) or COCO JSON (det/inst-seg) with the same ``image_sha256``
key as the prediction formats in :mod:`marinedata.eval.predictions`, plus
``split_group`` (the bootstrap's cluster key — see :mod:`marinedata.splitmap`) and
``split``. ``--release`` is accepted for output bookkeeping (recorded in
``metrics.json``) and is where a future P3/P5 wiring would source ``--gt`` from
instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import cluster_bootstrap
from .metrics import METRICS
from .predictions import (
    COCO_TASK_TYPES,
    SEG_TASK_TYPES,
    VQA_TASK_TYPES,
    read_predictions,
)

TASK_TYPE_METRICS: dict[str, tuple[str, ...]] = {
    "cls": ("macro_f1", "ece"),
    "points": ("point_acc", "point_macro_recall", "ece"),
    "sem-seg": ("miou",),
    "det": ("map",),
    "inst-seg": ("map",),
    "vqa": ("vqa_acc",),
}


def _read_gt_table(path: Path, task_type: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["image_sha256", "split_group", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: ground truth missing column(s) {missing}")
    return df


def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all":
        return df
    return df[df["split"] == split]


def _score_cls_or_points(
    gt_path: Path,
    pred_path: Path,
    task_type: str,
    split: str,
    metric_names: tuple[str, ...],
    *,
    n_boot: int,
    seed: int,
) -> tuple[dict[str, dict], np.ndarray, pd.DataFrame]:
    gt = _read_gt_table(gt_path, task_type)
    gt = _filter_split(gt, split)
    preds = read_predictions(pred_path, task_type)

    key_cols = ["image_sha256"] + (["point_id"] if task_type == "points" else [])
    pred_df = pd.DataFrame({"image_sha256": preds.image_sha256, "pred_label": preds.pred_label})
    if task_type == "points":
        pred_df["point_id"] = preds.point_id
    if preds.pred_probs is not None:
        pred_df["pred_probs"] = list(preds.pred_probs)

    merged = gt.merge(pred_df, on=key_cols, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("no rows joined between ground truth and predictions")

    y_true = merged["label"].to_numpy()
    y_pred = merged["pred_label"].to_numpy()
    probs = np.stack(merged["pred_probs"].to_numpy()) if "pred_probs" in merged.columns else None
    labels = sorted(set(y_true.tolist()))
    group_ids = merged["split_group"].to_numpy()

    results: dict[str, dict] = {}
    per_sample_correct = (y_true == y_pred).astype(int)

    def _make_statistic(metric_name: str):
        metric = METRICS[metric_name]
        if metric_name == "ece":

            def statistic(idx: np.ndarray) -> float:
                return metric.fn(probs[idx], y_true[idx])

        else:

            def statistic(idx: np.ndarray) -> float:
                return metric.fn(y_true[idx], y_pred[idx], labels=labels)

        return statistic

    for name in metric_names:
        if name == "ece" and probs is None:
            continue
        boot = cluster_bootstrap(group_ids, _make_statistic(name), n_boot=n_boot, seed=seed)
        results[name] = {**boot.to_dict(), "n_images": len(merged)}
    return results, per_sample_correct, merged


def _cmd_eval_score(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_names = tuple(args.metric) if args.metric else TASK_TYPE_METRICS[args.task_type]

    unwired = SEG_TASK_TYPES + COCO_TASK_TYPES + VQA_TASK_TYPES
    if args.task_type in unwired:
        raise NotImplementedError(
            f"eval score: task_type {args.task_type!r} readers exist "
            "(marinedata.eval.predictions) but the score wiring for this shape is P5 work"
        )

    results, _per_sample, merged = _score_cls_or_points(
        Path(args.gt),
        Path(args.pred),
        args.task_type,
        args.split,
        metric_names,
        n_boot=args.boot,
        seed=args.seed,
    )

    metrics_out = {
        "release": args.release,
        "task": args.task,
        "task_type": args.task_type,
        "split": args.split,
        "n_images": len(merged),
        "n_groups": int(merged["split_group"].nunique()),  # numpy int64 -> plain int
        "metrics": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, sort_keys=True))
    scores_path = out_dir / "scores.parquet"
    merged[["image_sha256", "split_group"]].assign(
        correct=(merged["label"] == merged["pred_label"]).astype(int)
    ).to_parquet(scores_path, index=False)
    print(json.dumps(metrics_out, indent=2, sort_keys=True))
    return 0


def add_eval_subparser(sub: argparse._SubParsersAction) -> None:
    p_eval = sub.add_parser("eval", help="Score model predictions against ground truth")
    eval_sub = p_eval.add_subparsers(dest="eval_command", required=True)

    p_score = eval_sub.add_parser(
        "score", help="Compute metrics + cluster-bootstrap CIs for one task/split"
    )
    p_score.add_argument("--release", required=True, help="Release dir or tag (recorded, not read)")
    p_score.add_argument("--task", required=True)
    p_score.add_argument(
        "--task-type",
        dest="task_type",
        required=True,
        choices=sorted(TASK_TYPE_METRICS),
    )
    p_score.add_argument("--split", required=True, help="test|val|ood-<name>|all")
    p_score.add_argument("--pred", required=True, help="Predictions file (format per §4.1)")
    p_score.add_argument("--gt", required=True, help="Ground-truth file, same key + split_group")
    p_score.add_argument(
        "--metric", action="append", help="Metric name(s); default = task-type set"
    )
    p_score.add_argument("--boot", type=int, default=1000)
    p_score.add_argument("--seed", type=int, default=20260925)
    p_score.add_argument("--out", required=True)
    p_score.set_defaults(func=_cmd_eval_score)
