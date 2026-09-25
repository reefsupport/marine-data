"""Cluster bootstrap CIs (design §4.3).

Images in the same ``split_group`` are correlated (near-duplicates, same dive/site),
so the bootstrap resamples *groups* with replacement — never rows — n = 1000, a fixed
default seed, 95% **percentile** interval. BCa was rejected in the design (its
jackknife needs one metric recompute per group; non-standard for a cluster design).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_SEED = 20260925
DEFAULT_N_BOOT = 1000
DEFAULT_CI = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate (computed on every row, unresampled) plus a percentile CI over
    ``n_boot`` cluster resamples."""

    point: float
    ci_lo: float
    ci_hi: float
    n_boot: int
    n_groups: int
    seed: int
    ci: float
    boot_values: np.ndarray

    def to_dict(self) -> dict[str, float | int]:
        return {
            "point": self.point,
            "ci_lo": self.ci_lo,
            "ci_hi": self.ci_hi,
            "n_boot": self.n_boot,
            "n_groups": self.n_groups,
            "seed": self.seed,
            "ci": self.ci,
        }


def cluster_bootstrap(
    group_ids: Sequence[Hashable] | np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> BootstrapResult:
    """Percentile cluster bootstrap over ``group_ids``.

    ``statistic(row_indices)`` must compute the metric using only the rows named by
    ``row_indices`` into the arrays the caller closed over — including repeats, since a
    group sampled twice contributes its rows twice. Called once unresampled (the point
    estimate) then ``n_boot`` times over resampled group membership.

    Seed-stable: the same ``seed`` always drives the same
    ``numpy.random.Generator.choice`` sequence, so repeat calls are bit-identical.
    """
    group_arr = np.asarray(group_ids)
    if len(group_arr) == 0:
        raise ValueError("cluster_bootstrap: group_ids is empty")
    unique_groups, inverse = np.unique(group_arr, return_inverse=True)
    n_groups = len(unique_groups)
    group_to_indices = [np.flatnonzero(inverse == g) for g in range(n_groups)]

    point = float(statistic(np.arange(len(group_arr))))

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([group_to_indices[g] for g in sampled])
        boot_values[b] = statistic(idx)

    alpha = (1.0 - ci) / 2.0
    ci_lo = float(np.quantile(boot_values, alpha))
    ci_hi = float(np.quantile(boot_values, 1.0 - alpha))
    return BootstrapResult(
        point=point,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        n_boot=n_boot,
        n_groups=n_groups,
        seed=seed,
        ci=ci,
        boot_values=boot_values,
    )
