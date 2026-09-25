"""Card results table (design §4.5, `results/<release>/baselines.md`).

Renders the one committed markdown table from a list of ``eval reproduce`` /
``eval baseline`` ``metrics.json`` dicts. Each task gets one row per
``(split in {test} + every ood-*) x method``, followed by an ID→OOD gap line.
"""

from __future__ import annotations

import subprocess

HEADER = (
    "| task | split | metric | method | backbone@rev | score | 95% CI | seeds "
    "| n img / groups | config sha | harness commit |"
)
SEP = "|---|---|---|---|---|---|---|---|---|---|---|"


def _harness_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:  # pragma: no cover - no git / detached env
        return "unknown"


def _fmt_ci(cell: dict) -> str:
    lo, hi = cell.get("ci_lo"), cell.get("ci_hi")
    if lo is None or hi is None:
        return "n/a"
    return f"[{lo:.1f}, {hi:.1f}]"


def render_card_table(
    metrics_rows: list[dict], *, method: str = "probe", metric: str = "macro_f1"
) -> str:
    """``metrics_rows``: one dict per (task, backbone) run, shaped like
    ``eval reproduce``'s ``metrics.json`` (``task``, ``backbone``, ``revision``,
    ``splits: {split_name: {macro_f1, n_images, [n_groups], [ci_lo], [ci_hi]}}``).
    """
    commit = _harness_commit()
    lines = [HEADER, SEP]
    for row in metrics_rows:
        task = row.get("task", "?")
        backbone_rev = f"{row.get('backbone', '?')}@{row.get('revision', '?')[:7]}"
        config_sha = row.get("config_sha", row.get("config", "n/a"))
        splits = row.get("splits", {})
        id_score = None
        for split_name, cell in sorted(splits.items()):
            score = cell.get(metric, cell.get("point"))
            n_images = cell.get("n_images", "n/a")
            n_groups = cell.get("n_groups", "n/a")
            seeds = cell.get("seeds", 0)
            lines.append(
                f"| {task} | {split_name} | {metric} | {method} | {backbone_rev} | "
                f"{score:.1f} | {_fmt_ci(cell)} | {seeds} | {n_images} / {n_groups} | "
                f"{config_sha} | {commit} |"
            )
            if split_name == "test":
                id_score = score
        ood_scores = [
            cell.get(metric, cell.get("point"))
            for name, cell in splits.items()
            if name.startswith("ood-")
        ]
        if id_score is not None and ood_scores:
            gap = id_score - (sum(ood_scores) / len(ood_scores))
            lines.append(
                f"| {task} | id-vs-ood-gap | {metric} | {method} | {backbone_rev} | "
                f"{gap:+.1f} | n/a | n/a | n/a | {config_sha} | {commit} |"
            )
    return "\n".join(lines) + "\n"
