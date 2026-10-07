"""WP-U11: the unified ``pairs`` table, its producers and the pairs config."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

from _depth_pairs_fixtures import fake_tree, sha_of

from marinedata.annotation_schema import validate_row, validate_rows
from marinedata.task_layers import configs
from marinedata.task_layers import depth_table as dt
from marinedata.task_layers import pairs_table as pt


def _run(sid, names, **kw):
    spec = dt.DEPTH_PAIR_SOURCES[sid]
    lister, fetch = fake_tree(spec, names, checksums=kw.pop("checksums", True))
    return spec, dt.staged_depth_pairs(spec, lister=lister, fetch=fetch, **kw)


def test_lsui_input_is_anchor_and_gt_the_enhanced_reference():
    names = ["LSUI_input_1.jpg", "LSUI_GT_1.jpg", "LSUI_input_2.jpg", "LSUI_GT_3.jpg"]
    _, res = _run("lsui", names)
    assert res.pairs.seen == 2 and len(res.pairs.rows) == 1
    assert res.pairs.skipped["no GT reference for the input"] == 1
    row = res.pairs.rows[0]
    assert row["image_sha256"] == sha_of("LSUI_input_1.jpg")
    assert row["ref_image_sha256"] == sha_of("LSUI_GT_1.jpg") and row["pair_role"] == "enhanced"
    attrs = json.loads(row["attrs"])
    assert attrs["pair_kind"] == "enhancement" and attrs["licence_class"] == "internal-only"
    assert attrs["ref_key"].endswith("/images/LSUI_GT_1.jpg")
    validate_rows("pairs", res.pairs.rows)


def test_sunboat_camera_sonar_pair_by_prefix_and_index():
    p = "Sunboat_03-09-2023_zip_Sunboat_03-09-2023_2023-09-03-07-58-37"
    _, res = _run(
        "auv-flc-fls-sunboat",
        [f"{p}_camera_00001.png", f"{p}_sonar_00001.png", f"{p}_camera_00002.png"],
    )
    assert len(res.pairs.rows) == 1 and res.pairs.rows[0]["pair_role"] == "sonar"
    assert json.loads(res.pairs.rows[0]["attrs"])["licence_class"] == "open"
    assert res.pairs.skipped["no sonar frame for the camera frame"] == 1


def test_euvp_builds_no_row_and_logs_the_gap():
    names = [f"data_train-00000-of-00001_parquet_{n}.jpg" for n in range(5)]
    _, res = _run("euvp", names)
    assert not res.pairs.rows and not res.pairs.pending and res.pairs.seen == 5
    assert len(res.gaps) == 1 and "U7 gap" in res.gaps[0] and "contiguous" in res.gaps[0]


def test_unchecksummed_pairs_are_pending_with_both_keys(tmp_path):
    names = [f"3d_models_HabCam2019_dataset1_samples_x-{k}.tif" for k in ("left", "right", "disp")]
    spec, res = _run("viame-public", names, checksums=False)
    assert not res.pairs.rows and len(res.pairs.pending) == 1
    pend = res.pairs.pending[0]
    assert pend["pair_role"] == "stereo_right" and pend["image_key"].endswith("-left")
    assert pend["ref_image_key"].endswith("-right")
    assert not {"image_sha256", "ref_image_sha256"} & set(pend)
    assert pt.validate_pending_pairs(res.pairs.pending) == []
    path = pt.pending_pair_path(tmp_path, spec.source_id, spec.version)
    assert pt.write_pending_pairs(path, res.pairs.pending) == 1
    import pyarrow.parquet as pq

    cols = pq.read_table(path).column_names
    assert {"image_key", "ref_image_key", "attrs"} <= set(cols)
    assert not {"image_sha256", "ref_image_sha256"} & set(cols)


def test_pending_pair_without_key_is_rejected():
    _, res = _run(
        "viame-public",
        [f"3d_models_HabCam2019_dataset1_samples_x-{k}.tif" for k in ("left", "right")],
        checksums=False,
    )
    bad = [{**res.pairs.pending[0], "ref_image_key": None}]
    assert any("ref_image_key missing" in e for e in pt.validate_pending_pairs(bad))


def test_pair_row_and_validator_reject_bad_enums():
    spec = dt.DEPTH_PAIR_SOURCES["lsui"]
    with pytest.raises(ValueError, match="pair_kind"):
        pt.pair_row(
            spec=spec,
            ordinal=0,
            sha=sha_of("a"),
            ref_sha=sha_of("b"),
            pair_role="enhanced",
            pair_kind="morph",
        )
    row = pt.pair_row(
        spec=spec,
        ordinal=0,
        sha=sha_of("a"),
        ref_sha=sha_of("b"),
        pair_role="enhanced",
        pair_kind="enhancement",
    )
    assert validate_row("pairs", row) == []
    assert validate_row("pairs", {**row, "pair_role": "nonsense"})
    assert validate_row("pairs", {**row, "ref_image_sha256": "xyz"})


def test_pairs_config_reads_unified_rows(tmp_path):
    spec, res = _run("lsui", ["LSUI_input_1.jpg", "LSUI_GT_1.jpg"])
    pt.write_pairs(tmp_path, spec.source_id, spec.version, res.pairs.rows)
    out = configs.build_pairs_config(tmp_path)
    assert out.config_id == "pairs" and len(out.rows) == 1
    assert out.rows[0]["licence_class"] == "internal-only" and out.rows[0]["sha256"] == sha_of(
        "LSUI_input_1.jpg"
    )
    assert "lsui" in configs.PAIRS_CONFIG_SOURCES and "usod10k" not in configs.PAIRS_CONFIG_SOURCES
