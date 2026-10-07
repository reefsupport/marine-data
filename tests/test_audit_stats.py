"""Tests for the D-I audit stats helpers (WP-5d). No heavy deps, always run."""

from __future__ import annotations

import pytest

from marinedata.audit_stats import extrapolate_stratified, wilson_ci


def test_wilson_ci_point_estimate_matches_raw_proportion() -> None:
    ci = wilson_ci(18, 150)
    assert ci.point == pytest.approx(18 / 150)


def test_wilson_ci_bounds_are_ordered_and_within_unit_interval() -> None:
    ci = wilson_ci(18, 150)
    assert 0.0 <= ci.low <= ci.point <= ci.high <= 1.0


def test_wilson_ci_zero_successes_has_nonzero_upper_bound() -> None:
    # matches the brief's negatives check: 0/100 -> a real, non-degenerate upper bound
    ci = wilson_ci(0, 100)
    assert ci.point == 0.0
    assert ci.low == 0.0
    assert ci.high == pytest.approx(0.0369, abs=0.001)


def test_wilson_ci_is_narrower_at_larger_n_for_same_proportion() -> None:
    small = wilson_ci(3, 10)
    large = wilson_ci(30, 100)
    assert (large.high - large.low) < (small.high - small.low)


def test_wilson_ci_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        wilson_ci(1, 0)
    with pytest.raises(ValueError):
        wilson_ci(5, 3)


def test_extrapolate_stratified_matches_hand_computed_point_estimate() -> None:
    point, low, high = extrapolate_stratified(
        stratum_hits={"face": 15, "person_only": 1, "negative": 0},
        stratum_sample_sizes={"face": 150, "person_only": 86, "negative": 100},
        stratum_population_sizes={"face": 3387, "person_only": 821, "negative": 65392},
    )
    expected_point = (15 / 150) * 3387 + (1 / 86) * 821 + (0 / 100) * 65392
    assert point == pytest.approx(expected_point)
    assert low <= point <= high


def test_extrapolate_stratified_skips_empty_sample_strata() -> None:
    point, _low, _high = extrapolate_stratified(
        stratum_hits={"a": 5},
        stratum_sample_sizes={"a": 10, "b": 0},
        stratum_population_sizes={"a": 100, "b": 500},
    )
    # stratum "b" has 0 samples, contributes nothing rather than dividing by zero
    assert point == pytest.approx(0.5 * 100)
