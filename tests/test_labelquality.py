"""WP-9 label-quality: agreement metrics, fold grouping, confident learning, features."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from marinedata.labelquality import agreement as A
from marinedata.labelquality import confident as CL
from marinedata.labelquality.features import preprocess, resize_shape


def test_cohen_kappa_textbook_value() -> None:
    # 2x2 table [[20, 5], [10, 15]] → po = 0.7, pe = 0.5 → κ = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert A.cohen_kappa(a, b) == pytest.approx(0.4)
    assert A.cohen_kappa(a, a) == pytest.approx(1.0)
    assert math.isnan(A.cohen_kappa(["y"] * 5, ["y"] * 5))  # chance agreement is 1


def test_scott_pi_is_order_free_and_below_kappa_on_skewed_marginals() -> None:
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert A.scott_pi(a, b) == pytest.approx(A.scott_pi(b, a))
    assert A.scott_pi(a, b) <= A.cohen_kappa(a, b) + 1e-12


def _units() -> dict[str, list[A.Unit]]:
    return {
        "s1": [A.Unit("s1", "src-a", "H"), A.Unit("s1", "src-b", "U")],
        "s2": [A.Unit("s2", "src-a", "H")],
        "s3": [A.Unit("s3", "src-c", "H")],
    }


def test_unit_pairs_orient_and_scope() -> None:
    units = _units()
    pairs = A.unit_pairs({"c1": ["s2", "s3"], "sha:s1": ["s1"]}, units)
    assert len(pairs) == 2
    by = {p.cluster: p for p in pairs}
    assert by["c1"].a.source_id == "src-a" and by["c1"].b.source_id == "src-c"
    assert by["c1"].agree and by["c1"].cross_source
    assert not by["sha:s1"].agree
    assert list(A.conflicts(u for us in units.values() for u in us)) == ["s1"]


def test_dhash_clusters_only_multi() -> None:
    got = A.dhash_clusters({"a": "ff", "b": "ff", "c": "01"})
    assert got == {"dh:ff": ["a", "b"]}


def test_summary_and_cluster_bootstrap_is_deterministic() -> None:
    units = {f"s{i}": [A.Unit(f"s{i}", "x", "H" if i % 3 else "U")] for i in range(60)}
    units.update({f"t{i}": [A.Unit(f"t{i}", "y", "H" if i % 3 else "U")] for i in range(60)})
    units["t0"] = [A.Unit("t0", "y", "H")]  # one disagreement
    clusters = {f"c{i}": [f"s{i}", f"t{i}"] for i in range(60)}
    s = A.summarise(A.unit_pairs(clusters, units), n_boot=200)
    assert s.n_pairs == 60 and s.agreement == pytest.approx(59 / 60)
    assert s.kappa_ci[0] <= s.kappa <= s.kappa_ci[1]
    again = A.summarise(A.unit_pairs(clusters, units), n_boot=200)
    assert again.kappa_ci == s.kappa_ci
    assert s.confusion == {"H|H": 40, "H|U": 1, "U|U": 19}


def test_group_folds_never_split_a_group_and_balance() -> None:
    groups = [f"g{i // 3}" for i in range(300)] + ["big"] * 40
    folds = CL.group_folds(groups, k=5)
    seen: dict[str, set[int]] = {}
    for g, f in zip(groups, folds, strict=True):
        seen.setdefault(g, set()).add(int(f))
    assert all(len(v) == 1 for v in seen.values())
    sizes = np.bincount(folds, minlength=5)
    assert sizes.max() - sizes.min() <= 40
    assert np.array_equal(folds, CL.group_folds(groups, k=5))


def _noisy_blobs(n: int = 1200, flip: float = 0.1, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, n)
    x = rng.normal(size=(n, 8)) + np.where(y_true[:, None] == 1, 2.0, -2.0) * np.eye(8)[0]
    flipped = rng.random(n) < flip
    return x, np.where(flipped, 1 - y_true, y_true), flipped


def test_confident_learning_recovers_injected_noise() -> None:
    x, given, flipped = _noisy_blobs()
    groups = [f"g{i}" for i in range(len(given))]
    probs = CL.oof_probs(x, given, CL.group_folds(groups, 5), 2, c=1.0)
    assigned = CL.confident_assign(probs, CL.per_class_thresholds(probs, given))
    flags = CL.label_issues(given, assigned)
    precision = flipped[flags].mean()
    recall = flags[flipped].mean()
    assert precision > 0.8 and recall > 0.8
    assert CL.noise_rate(given, assigned, 2) == pytest.approx(flipped.mean(), abs=0.04)
    lo, hi = CL.group_bootstrap_noise(given, assigned, groups, 2, n=200)
    assert lo <= flipped.mean() <= hi


def test_confident_joint_rows_calibrate_to_given_counts() -> None:
    given = np.array([0, 0, 0, 1, 1, 1, 1])
    assigned = np.array([0, 1, -1, 1, 1, 0, 1])
    cj = CL.confident_joint(given, assigned, 2)
    assert cj.sum(axis=1) == pytest.approx([3, 4])


def test_multiclass_probe_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    x = rng.normal(size=(300, 4)) + np.eye(4)[y] * 3
    p = CL.fit_logreg(x, y, 3).predict_proba(x)
    assert p.shape == (300, 3) and np.allclose(p.sum(axis=1), 1)
    assert (p.argmax(axis=1) == y).mean() > 0.9


def test_preprocess_resize_rule_and_shape() -> None:
    assert resize_shape(640, 480, 256) == (341, 256)
    assert resize_shape(480, 640, 256) == (256, 341)
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (124, 116, 104)).save(buf, format="PNG")
    arr = preprocess(buf.getvalue())
    assert arr.shape == (3, 224, 224) and arr.dtype == np.float32
    assert abs(float(arr.mean())) < 0.05  # the ImageNet mean colour normalises to ~0


def test_cli_wires_labelquality() -> None:
    from marinedata.cli import build_parser

    args = build_parser().parse_args(
        ["labelquality", "agreement", "--hf", "h", "--dhash-db", "d", "--out", "o"]
    )
    assert args.labelquality_command == "agreement"


def test_label_origin_covers_every_v1_source_with_a_valid_origin() -> None:
    from pathlib import Path

    import yaml

    doc = yaml.safe_load((Path(__file__).parents[1] / "registry/label-origin.yaml").read_text())
    allowed = {"human_expert", "human_crowd", "pseudo_model", "derived_rule", "unknown"}
    assert all(e["origin"] in allowed and e["evidence"] for e in doc["entries"])
    v1 = {
        "coralscop-masks-rs", "noaa-pifsc-bleaching", "reef-support-benthic-own",
        "reef-support-bleaching", "roboflow-coral-bleaching-final-v6i",
        "roboflow-coral-bleaching-general-v1-yolov8s",
        "roboflow-coral-classification-copy-changed-v13i",
        "roboflow-coral-reef-bleach-detection-v2i", "roboflow-coral-reef-classification-v3i",
    }  # fmt: skip
    assert v1 <= {e["source"] for e in doc["entries"]}


def test_expert_sheet_allocation_and_weights() -> None:
    import pandas as pd

    from marinedata.labelquality.protocol import EXPERT_COLUMNS, allocate, expert_sheet

    alloc = allocate({("a",): 1000, ("b",): 30, ("c",): 5}, 100)
    assert sum(alloc.values()) == 100 and alloc[("c",)] == 5 and alloc[("b",)] >= 30 // 2
    rows = [(f"s{i}", "src" if i % 4 else "other", "H" if i % 3 else "U") for i in range(800)]
    labels = pd.DataFrame(rows, columns=["image_sha256", "source_id", "label"])
    sheet = expert_sheet(labels, {("s1", "src"), ("s2", "src")}, n=60)
    assert len(sheet) == 60 and sheet.audit_id.is_unique
    assert set(EXPERT_COLUMNS) <= set(sheet.columns)
    assert sheet.cl_flag.sum() == 2  # tiny flag strata are taken whole
    # Horvitz-Thompson weights recover the population size per stratum
    assert sheet.groupby("stratum").weight.sum().sum() == pytest.approx(800)
