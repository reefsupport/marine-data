"""WP-9 label-quality: agreement metrics, fold grouping, confident learning, features."""

from __future__ import annotations

import io
import math

import numpy as np
import pytest

from marinedata.labelquality import agreement as A
from marinedata.labelquality import confident as CL
from marinedata.labelquality import quality as Q
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


def _png(pixels: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(pixels.astype(np.uint8), "RGB").save(buf, format="PNG")
    return buf.getvalue()


def test_quality_features_separate_sharp_from_blurred() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    blurred = np.full((64, 64, 3), 128, dtype=np.uint8)
    f_sharp = Q.compute_features(_png(sharp))
    f_blur = Q.compute_features(_png(blurred))
    assert f_sharp.blur > f_blur.blur
    assert f_blur.blur == pytest.approx(0.0, abs=1e-6)


def test_quality_features_flag_colour_cast_and_saturation() -> None:
    balanced = np.full((32, 32, 3), 128, dtype=np.uint8)
    green_cast = np.zeros((32, 32, 3), dtype=np.uint8)
    green_cast[..., 1] = 220  # pure green channel: maximum cast and saturation
    f_balanced = Q.compute_features(_png(balanced))
    f_cast = Q.compute_features(_png(green_cast))
    assert f_balanced.color_cast == pytest.approx(0.0, abs=1e-6)
    assert f_cast.color_cast > 0.5
    assert f_cast.saturation_p50 == pytest.approx(1.0, abs=1e-6)
    assert f_balanced.saturation_p50 == pytest.approx(0.0, abs=1e-6)


def test_bleach_gate_passes_requires_every_threshold_and_margin() -> None:
    gate = Q.BleachGate(
        blur_min=10.0, color_cast_max=0.3, luminance_min=0.2, luminance_max=0.8, margin_min=0.5
    )
    good = Q.QualityFeatures(blur=50.0, color_cast=0.1, luminance=0.5, saturation_p50=0.3)
    assert gate.passes(good, margin=0.6) is True
    assert gate.passes(good, margin=0.4) is False  # margin too low
    too_blurry = Q.QualityFeatures(blur=1.0, color_cast=0.1, luminance=0.5, saturation_p50=0.3)
    assert gate.passes(too_blurry, margin=0.9) is False
    assert gate.enabled is False  # never on by default


def test_tuned_gate_is_shipped_off_by_default() -> None:
    assert Q.TUNED_GATE.enabled is False


def test_classify_conflict_categories() -> None:
    from marinedata.labelquality.pipeline import V1_YOLOV8S, V3I, V6I, V13I, classify_conflict

    # within-source duplicate (same source twice, different labels) -> conflict
    assert classify_conflict([("src-a", "H"), ("src-a", "U")]) == "conflict"
    # v3i disagreeing with anyone is a concept mismatch, not an error -> ok
    assert classify_conflict([(V3I, "Unhealthy"), (V1_YOLOV8S, "Healthy")]) == "ok"
    # v13i vs v1-yolov8s alone (v6i absent) -> conflict (the relabelling fork)
    assert classify_conflict([(V13I, "Bleached"), (V1_YOLOV8S, "Healthy")]) == "conflict"
    # v1-yolov8s vs v6i+v13i together -> ambiguous (expert call, D-I)
    assert (
        classify_conflict([(V1_YOLOV8S, "Healthy"), (V6I, "Bleached"), (V13I, "Bleached")])
        == "ambiguous"
    )


def _write_label_status_fixture(tmp_path):  # type: ignore[no-untyped-def]
    """Minimal hf_root + labelquality_dir + model-audit.tsv for ``run_label_status`` (D-U2).

    Four sha256: ``healthy`` (CL flag toward HEALTHY, precision 1.0 -> flagged_hard),
    ``bleach`` (CL flag toward BLEACHED, precision 0.0 -> stays ok), ``conflict`` (an
    identical-sha conflict that is *also* a toward-HEALTHY CL flag -> conflict wins),
    ``plain`` (no flag, no conflict -> ok, cl_flag null).
    """
    import pandas as pd

    hf_root = tmp_path / "hf"
    (hf_root / "data" / "coral-health-binary").mkdir(parents=True)
    (hf_root / "data" / "bleaching-condition").mkdir(parents=True)
    cols = ["image_sha256", "source_id", "sample_key", "label", "native_label", "label_reason"]
    labels = pd.DataFrame(
        [
            ("healthy", "src-a", "k1", "UNHEALTHY", "Unhealthy", None),
            ("bleach", "src-a", "k2", "HEALTHY", "Healthy", None),
            ("conflict", "src-a", "k3", "UNHEALTHY", "Unhealthy", None),
            ("conflict", "src-a", "k3", "HEALTHY", "Healthy", None),
            ("plain", "src-a", "k4", "HEALTHY", "Healthy", None),
        ],
        columns=cols,
    )
    labels.to_parquet(hf_root / "data" / "coral-health-binary" / "train-0.parquet", index=False)
    labels.iloc[:0].to_parquet(
        hf_root / "data" / "bleaching-condition" / "train-0.parquet", index=False
    )

    labelquality_dir = tmp_path / "lq"
    labelquality_dir.mkdir()
    pd.DataFrame(
        [("conflict", "src-a", "UNHEALTHY"), ("conflict", "src-a", "HEALTHY")],
        columns=["image_sha256", "source_id", "label"],
    ).to_csv(labelquality_dir / "conflicts-coral-health-binary.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            ("healthy", "HEALTHY"),
            ("bleach", "BLEACHED"),
            ("conflict", "HEALTHY"),
        ],
        columns=["image_sha256", "suggested"],
    ).to_parquet(labelquality_dir / "label_issues.parquet", index=False)

    model_audit_tsv = tmp_path / "model-audit.tsv"
    pd.DataFrame(
        [
            ("m1", "HEALTHY", "H", 1),
            ("m2", "HEALTHY", "H", 1),
            ("m3", "BLEACHED", "U", 0),
            ("m4", "BLEACHED", "U", 0),
            ("m5", "BLEACHED", None, 1),  # undecided (verdict_class null) -> excluded
        ],
        columns=["image_sha256", "suggested", "verdict_class", "flag_correct"],
    ).to_csv(model_audit_tsv, sep="\t", index=False)

    return hf_root, labelquality_dir, model_audit_tsv


def test_run_label_status_direction_threshold(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from marinedata.labelquality.pipeline import run_label_status

    hf_root, labelquality_dir, model_audit_tsv = _write_label_status_fixture(tmp_path)
    out = tmp_path / "label_status.parquet"
    counts = run_label_status(hf_root, labelquality_dir, model_audit_tsv, out)

    import pandas as pd

    table = pd.read_parquet(out).set_index("sha256")
    # toward-healthy precision is 1.0 (>= 0.60 floor) -> the flag becomes flagged_hard
    assert table.loc["healthy", "label_status"] == "flagged_hard"
    # toward-bleached/unhealthy precision is 0.0 (< 0.60 floor) -> stays ok
    assert table.loc["bleach", "label_status"] == "ok"
    assert counts["flagged_hard"] == 1
    assert counts["ok"] == 2  # "bleach" (below the floor) + "plain" (never flagged)


def test_run_label_status_metadata_carries_precisions(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    import pyarrow.parquet as pq

    from marinedata.labelquality.pipeline import run_label_status

    hf_root, labelquality_dir, model_audit_tsv = _write_label_status_fixture(tmp_path)
    out = tmp_path / "label_status.parquet"
    run_label_status(hf_root, labelquality_dir, model_audit_tsv, out)

    meta = pq.read_schema(out).metadata
    precision = json.loads(meta[b"label_quality.direction_precision"])
    assert precision == {"toward_healthy": 1.0, "toward_bleached_unhealthy": 0.0}
    assert meta[b"label_quality.hard_flag_min_precision"] == b"0.6"


def test_run_label_status_toward_bleached_flag_stays_ok_with_cl_flag(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    import pandas as pd

    from marinedata.labelquality.pipeline import run_label_status

    hf_root, labelquality_dir, model_audit_tsv = _write_label_status_fixture(tmp_path)
    out = tmp_path / "label_status.parquet"
    run_label_status(hf_root, labelquality_dir, model_audit_tsv, out)

    row = pd.read_parquet(out).set_index("sha256").loc["bleach"]
    assert row.label_status == "ok"
    assert row.cl_flag is True
    assert row.cl_flag_direction == "toward_bleached_unhealthy"
    # a never-flagged row carries nulls in both columns
    plain = pd.read_parquet(out).set_index("sha256").loc["plain"]
    assert pd.isna(plain.cl_flag) and pd.isna(plain.cl_flag_direction)


def test_run_label_status_conflict_beats_a_qualifying_cl_flag(
    tmp_path,  # type: ignore[no-untyped-def]
) -> None:
    """D-U2 must not touch precedence: conflict/ambiguous stay unchanged (WP-9 D-U)."""
    import pandas as pd

    from marinedata.labelquality.pipeline import run_label_status

    hf_root, labelquality_dir, model_audit_tsv = _write_label_status_fixture(tmp_path)
    out = tmp_path / "label_status.parquet"
    run_label_status(hf_root, labelquality_dir, model_audit_tsv, out)

    row = pd.read_parquet(out).set_index("sha256").loc["conflict"]
    # "conflict" is a within-source duplicate (conflicts.tsv) AND a toward-healthy CL
    # flag (which alone would qualify for flagged_hard) -> conflict wins either way.
    assert row.label_status == "conflict"
    assert row.cl_flag is True  # the columns are still populated for a CL flag row
    assert row.cl_flag_direction == "toward_healthy"


def test_grid_search_respects_recall_floor() -> None:
    rng = np.random.default_rng(3)
    n = 40
    labels = (rng.random(n) > 0.5).astype(int)
    # A feature that perfectly separates the classes, plus noise features.
    blur = np.where(labels == 1, 500.0, 50.0) + rng.normal(scale=1.0, size=n)
    feats = [
        Q.QualityFeatures(blur=b, color_cast=0.1, luminance=0.5, saturation_p50=0.3) for b in blur
    ]
    margins = np.full(n, 0.9)
    _gate, precision, recall = Q.grid_search(feats, margins, labels, min_recall=0.75)
    assert recall >= 0.75
    assert precision > labels.mean()  # gate beats the un-gated baseline here


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
