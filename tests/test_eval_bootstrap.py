""":mod:`marinedata.eval.bootstrap` (design §4.3): cluster (not row) resampling,
seed-stability, and the percentile CI."""

from __future__ import annotations

import numpy as np
import pytest

from marinedata.eval.bootstrap import cluster_bootstrap


def _mean_statistic(values: np.ndarray):
    def statistic(idx: np.ndarray) -> float:
        return float(values[idx].mean())

    return statistic


def test_point_estimate_is_the_unresampled_mean() -> None:
    values = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    result = cluster_bootstrap(groups, _mean_statistic(values))
    assert result.point == pytest.approx(0.5)
    assert result.n_groups == 2
    assert result.n_boot == 1000


def test_group_resampling_not_row_resampling() -> None:
    """Group A is all-1, group B is all-0. A *cluster* bootstrap over 2 groups can only
    ever land on {0.0, 0.5, 1.0} (BB, AB/BA, AA) — a row-level bootstrap over 6 rows
    would produce many more distinct values (e.g. 1/6, 2/6, 1/3, ...). This is the
    fixture proving the resampling unit is the group, not the row.
    """
    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    groups = np.array(["a", "a", "a", "b", "b", "b"])
    result = cluster_bootstrap(groups, _mean_statistic(values), n_boot=2000, seed=1)
    distinct = set(np.round(result.boot_values, 6).tolist())
    assert distinct <= {0.0, 0.5, 1.0}
    # both non-trivial outcomes must actually occur across 2000 draws
    assert 0.0 in distinct and 1.0 in distinct


def test_group_resampling_uneven_group_sizes() -> None:
    """A bigger group must still resample as one whole unit, its size and all — the
    mixed draw (one of each group) means over 12 rows (10 + 2), not two group-means
    averaged at 0.5. Only 2 groups exist, so exactly 3 outcomes are possible.
    """
    values = np.concatenate([np.ones(10), np.zeros(2)])
    groups = np.array(["big"] * 10 + ["small"] * 2)
    result = cluster_bootstrap(groups, _mean_statistic(values), n_boot=500, seed=3)
    expected = {round(v, 6) for v in (0.0, 10.0 / 12.0, 1.0)}
    distinct = set(np.round(result.boot_values, 6).tolist())
    assert distinct <= expected
    assert round(10.0 / 12.0, 6) in distinct  # the mixed (one big, one small) draw


def test_seed_stability_identical_seed_identical_result() -> None:
    rng = np.random.default_rng(0)
    values = rng.random(200)
    groups = rng.integers(0, 20, size=200)
    r1 = cluster_bootstrap(groups, _mean_statistic(values), seed=42, n_boot=300)
    r2 = cluster_bootstrap(groups, _mean_statistic(values), seed=42, n_boot=300)
    assert r1.ci_lo == r2.ci_lo
    assert r1.ci_hi == r2.ci_hi
    assert np.array_equal(r1.boot_values, r2.boot_values)


def test_seed_stability_different_seed_differs() -> None:
    rng = np.random.default_rng(0)
    values = rng.random(200)
    groups = rng.integers(0, 20, size=200)
    r1 = cluster_bootstrap(groups, _mean_statistic(values), seed=1, n_boot=300)
    r2 = cluster_bootstrap(groups, _mean_statistic(values), seed=2, n_boot=300)
    assert not np.array_equal(r1.boot_values, r2.boot_values)


def test_percentile_ci_matches_manual_quantiles() -> None:
    rng = np.random.default_rng(7)
    values = rng.random(500)
    groups = rng.integers(0, 50, size=500)
    result = cluster_bootstrap(groups, _mean_statistic(values), seed=99, n_boot=1000, ci=0.95)
    assert result.ci_lo == pytest.approx(float(np.quantile(result.boot_values, 0.025)))
    assert result.ci_hi == pytest.approx(float(np.quantile(result.boot_values, 0.975)))
    assert result.ci_lo <= result.point <= result.ci_hi


def test_empty_group_ids_raises() -> None:
    with pytest.raises(ValueError):
        cluster_bootstrap(np.array([]), _mean_statistic(np.array([])))
