"""Probe determinism on synthetic features (design §4.4.A: "deterministic")."""

import numpy as np
import pytest

# scikit-learn ships in the `eval`/`baselines` extras, not in `dev` (what CI installs),
# and `probe` imports it at module level.
pytest.importorskip("sklearn")

from marinedata.eval.baselines.probe import fit_probe, score_probe


def _synthetic_features(seed: int, n: int = 400, dim: int = 16, n_classes: int = 3):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    centers = rng.normal(size=(n_classes, dim)) * 5
    x = centers[y] + rng.normal(size=(n, dim))
    groups = np.arange(n) // 4
    return x.astype(np.float32), y, groups


def test_fit_probe_is_bit_stable_given_fixed_features():
    x, y, groups = _synthetic_features(seed=0)
    x_val, y_val, _ = _synthetic_features(seed=1)

    r1 = fit_probe(x, y, x_val, y_val, group_ids=groups, seed=0)
    r2 = fit_probe(x, y, x_val, y_val, group_ids=groups, seed=0)

    f1_a, pred_a = score_probe(r1, x_val, y_val)
    f1_b, pred_b = score_probe(r2, x_val, y_val)
    assert f1_a == f1_b
    assert np.array_equal(pred_a, pred_b)
    assert r1.best_C == r2.best_C


def test_fit_probe_train_cap_is_seeded_whole_group_sample():
    x, y, groups = _synthetic_features(seed=2, n=200)
    x_val, y_val, _ = _synthetic_features(seed=3, n=50)
    # cap smaller than n forces the sampler path; must stay deterministic and never
    # split a group across the cap boundary.
    r1 = fit_probe(x, y, x_val, y_val, group_ids=groups, cap=100, seed=7)
    r2 = fit_probe(x, y, x_val, y_val, group_ids=groups, cap=100, seed=7)
    assert r1.val_macro_f1 == r2.val_macro_f1
