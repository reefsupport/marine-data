"""Frozen linear probe on cached features (design §4.4.A, cls/points tasks).

``LogisticRegression(lbfgs, multinomial)``, ``C`` chosen on the val split by macro-F1
over a small deterministic grid. Train is capped at 100k images per task via a seeded
group sample (a group — ``split_group_id`` — never straddles the cap boundary, so the
cap cannot leak a group's images across the sample/non-sample line). Everything here is
CPU-only and deterministic given fixed features: no randomness beyond the seeded
sampler and lbfgs, which is itself deterministic for a fixed input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
DEFAULT_TRAIN_CAP = 100_000


@dataclass(frozen=True)
class ProbeResult:
    model: LogisticRegression
    best_C: float
    val_macro_f1: float
    labels: tuple[str, ...]


def _capped_group_sample(n: int, group_ids: np.ndarray, cap: int, seed: int) -> np.ndarray:
    """Seeded whole-group sample of row indices, at most ``cap`` rows total."""
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    groups = np.unique(group_ids)
    rng.shuffle(groups)
    picked_mask = np.zeros(n, dtype=bool)
    count = 0
    for g in groups:
        if count >= cap:
            break
        idx = np.flatnonzero(group_ids == g)
        picked_mask[idx] = True
        count += len(idx)
    return np.flatnonzero(picked_mask)


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    group_ids: np.ndarray | None = None,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    cap: int = DEFAULT_TRAIN_CAP,
    seed: int = 0,
) -> ProbeResult:
    """Deterministic given ``(x_train, y_train, x_val, y_val, group_ids, seed)``."""
    x_train = np.asarray(x_train, dtype=np.float32)
    if group_ids is None:
        group_ids = np.arange(len(x_train))
    keep = _capped_group_sample(len(x_train), np.asarray(group_ids), cap, seed)
    xt, yt = x_train[keep], np.asarray(y_train)[keep]
    labels = tuple(sorted(set(yt.tolist())))

    best: tuple[float, float, LogisticRegression] | None = None
    for c in c_grid:
        # sklearn >= 1.7 dropped `multi_class`: lbfgs is multinomial by default now,
        # which is exactly design §4.4.A's "LogisticRegression(lbfgs, multinomial)".
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=seed)
        clf.fit(xt, yt)
        val_pred = clf.predict(np.asarray(x_val, dtype=np.float32))
        score = f1_score(y_val, val_pred, average="macro", labels=labels, zero_division=0)
        if best is None or score > best[0]:
            best = (score, c, clf)

    assert best is not None
    score, c, clf = best
    return ProbeResult(model=clf, best_C=c, val_macro_f1=float(score), labels=labels)


def score_probe(result: ProbeResult, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Macro-F1 and raw predictions of a fitted probe on a held-out split."""
    pred = result.model.predict(np.asarray(x, dtype=np.float32))
    f1 = f1_score(y, pred, average="macro", labels=result.labels, zero_division=0)
    return float(f1), pred
