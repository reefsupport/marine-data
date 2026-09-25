"""Confident learning over out-of-fold probe probabilities (WP-9).

* :func:`group_folds` — K folds with whole ``split_group_id`` groups kept together, so a
  near-duplicate of a held-out image is never in its training fold (WP-10 groups contain
  every dHash/SSCD duplicate cluster).
* :func:`fit_logreg` / :func:`oof_probs` — an L2 logistic probe on frozen features.
  Binary tasks use exact Newton/IRLS; K > 2 uses one-vs-rest IRLS, renormalised. Pure numpy,
  deterministic, no sklearn dependency.
* :func:`confident_joint` etc. — the cleanlab recipe (Northcutt et al. 2021): per-class
  thresholds ``t_j`` = mean ``p_j`` over samples labelled ``j``; a sample counts toward
  ``C[given, j*]`` where ``j*`` is the most probable class among those with ``p_j ≥ t_j``;
  rows are calibrated to the given-label counts. Off-diagonal samples are the label issues.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def group_folds(groups: Sequence[str], k: int = 5, seed: int = 20260925) -> np.ndarray:
    """Fold id per sample; whole groups per fold, largest groups placed first.

    Greedy: groups sorted by size (desc), ties broken by a seeded hash of the id, each put
    in the currently smallest fold. Deterministic, and balanced to within one group.
    """
    sizes: dict[str, int] = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1

    def tie(g: str) -> str:
        return hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()

    fold_of: dict[str, int] = {}
    load = [0] * k
    for g in sorted(sizes, key=lambda g: (-sizes[g], tie(g))):
        f = min(range(k), key=lambda i: (load[i], i))
        fold_of[g] = f
        load[f] += sizes[g]
    return np.array([fold_of[g] for g in groups], dtype=np.int64)


def _irls(x: np.ndarray, y: np.ndarray, lam: float, iters: int = 25) -> np.ndarray:
    """Binary L2 logistic regression by Newton's method; ``x`` has a bias column last."""
    n, d = x.shape
    w = np.zeros(d)
    reg = np.full(d, lam)
    reg[-1] = 0.0  # no penalty on the bias
    for _ in range(iters):
        z = np.clip(x @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = x.T @ (p - y) / n + reg * w
        hess = (x * (p * (1 - p))[:, None]).T @ x / n + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-6:
            break
    return w


@dataclass
class LogReg:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray  # (k or 1, d + 1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean) / self.scale
        z = np.hstack([z, np.ones((len(z), 1))])
        logits = np.clip(z @ self.weights.T, -30, 30)
        p = 1.0 / (1.0 + np.exp(-logits))
        if self.weights.shape[0] == 1:
            return np.hstack([1 - p, p])
        return p / p.sum(axis=1, keepdims=True)


def fit_logreg(x: np.ndarray, y: np.ndarray, n_classes: int, c: float = 1.0) -> LogReg:
    """Standardise, then fit an L2 probe with penalty ``1 / (c · n)`` per sample-mean loss."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-6
    z = np.hstack([(x - mean) / scale, np.ones((len(x), 1))])
    lam = 1.0 / (c * len(x))
    if n_classes == 2:
        w = _irls(z, (y == 1).astype(float), lam)[None, :]
    else:
        w = np.stack([_irls(z, (y == j).astype(float), lam) for j in range(n_classes)])
    return LogReg(mean, scale, w)


def oof_probs(
    x: np.ndarray, y: np.ndarray, folds: np.ndarray, n_classes: int, c: float = 1.0
) -> np.ndarray:
    """Out-of-fold class probabilities: each fold predicted by a probe fit on the others."""
    out = np.zeros((len(y), n_classes))
    for f in np.unique(folds):
        test = folds == f
        model = fit_logreg(x[~test], y[~test], n_classes, c)
        out[test] = model.predict_proba(x[test])
    return out


def per_class_thresholds(probs: np.ndarray, given: np.ndarray) -> np.ndarray:
    k = probs.shape[1]
    return np.array([probs[given == j, j].mean() if np.any(given == j) else 1.0 for j in range(k)])


def confident_assign(probs: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """``j*`` per sample: argmax over classes with ``p_j ≥ t_j``; ``-1`` when none clears."""
    above = probs >= thresholds[None, :]
    masked = np.where(above, probs, -np.inf)
    j = masked.argmax(axis=1)
    return np.where(above.any(axis=1), j, -1)


def confident_joint(given: np.ndarray, assigned: np.ndarray, k: int) -> np.ndarray:
    """Calibrated confident joint ``C[given, true]`` (rows sum to the given-label counts)."""
    cj = np.zeros((k, k))
    ok = assigned >= 0
    np.add.at(cj, (given[ok], assigned[ok]), 1)
    counts = np.bincount(given, minlength=k).astype(float)
    rows = cj.sum(axis=1, keepdims=True)
    return np.where(rows > 0, cj / np.where(rows == 0, 1, rows) * counts[:, None], 0.0)


def noise_rate(given: np.ndarray, assigned: np.ndarray, k: int) -> float:
    """Estimated fraction of wrong given labels: off-diagonal mass of the calibrated joint."""
    cj = confident_joint(given, assigned, k)
    total = cj.sum()
    return float((total - np.trace(cj)) / total) if total else float("nan")


def label_issues(given: np.ndarray, assigned: np.ndarray) -> np.ndarray:
    """cleanlab ``filter_by='confident_learning'``: confidently assigned to another class."""
    return (assigned >= 0) & (assigned != given)


def group_bootstrap_noise(
    given: np.ndarray,
    assigned: np.ndarray,
    groups: Sequence[str],
    k: int,
    *,
    n: int = 1000,
    seed: int = 20260925,
) -> tuple[float, float]:
    """95% percentile CI of :func:`noise_rate`, resampling whole groups."""
    garr = np.asarray(groups)
    uniq, inv = np.unique(garr, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv, minlength=len(uniq)))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[i] for i in pick])
        vals.append(noise_rate(given[idx], assigned[idx], k))
    lo, hi = np.nanpercentile(vals, [2.5, 97.5])
    return float(lo), float(hi)
