"""WP-U11: the unified ``depth`` table, its producers and the depth config."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

from _depth_pairs_fixtures import fake_tree, sha_of

from marinedata.annotation_schema import TABLES, read_annotations, validate_rows
from marinedata.task_layers import configs
from marinedata.task_layers import depth_table as dt
from marinedata.task_layers.vqa_table import (
    pending_path,
    validate_pending,
    write_pending,
)

USOD = dt.DEPTH_PAIR_SOURCES["usod10k"]
NAMES = [
    "USOD10k_USOD10k_TR_RGB_00001.png", "USOD10k_USOD10k_TR_GT_00001.png",
    "USOD10k_USOD10k_TR_depth_00001.png", "USOD10k_USOD10k_TR_Boundary_00001_edge.png",
    "USOD10k_USOD10k_TR_RGB_00002.png",  # no depth for frame 2
    "USOD10k_USOD10k_VAL_RGB_00003.png", "USOD10k_USOD10k_VAL_depth_00003.png",
]  # fmt: skip


def _run(spec, names, **kw):
    lister, fetch = fake_tree(spec, names, checksums=kw.pop("checksums", True))
    return dt.staged_depth_pairs(spec, lister=lister, fetch=fetch, **kw)


def test_every_task_table_carries_nullable_attrs():
    for name, spec in TABLES.items():
        col = next(c for c in spec.columns if c.name == "attrs")
        assert col.nullable, name


def test_usod10k_pairs_rgb_with_depth_inside_images():
    res = _run(USOD, NAMES)
    assert res.depth.seen == 3 and len(res.depth.rows) == 2 and not res.depth.pending
    assert res.depth.skipped["no depth file for the RGB frame"] == 1
    validate_rows("depth", res.depth.rows)
    row = res.depth.rows[0]
    assert row["image_sha256"] == sha_of("USOD10k_USOD10k_TR_RGB_00001.png")
    assert row["depth_ref"].endswith("/images/USOD10k_USOD10k_TR_depth_00001.png")
    assert (row["units"], row["gt_type"], row["upstream_split"]) == (
        "relative",
        "estimated",
        "train",
    )
    assert row["annotator_type"] == "pseudo"
    attrs = json.loads(row["attrs"])
    assert attrs["licence_class"] == "internal-only" and attrs["depth_kind"] == "pseudo"
    assert attrs["non_image"] is True and attrs["depth_sha256"] == sha_of(
        "USOD10k_USOD10k_TR_depth_00001.png"
    )
    assert attrs["sibling_non_image"] == {
        "GT": "USOD10k_USOD10k_TR_GT_00001", "Boundary": "USOD10k_USOD10k_TR_Boundary_00001_edge"
    }  # fmt: skip
    assert res.depth.rows[1]["upstream_split"] == "val"


def test_limit_keeps_ordinals_of_the_full_set():
    names = [f"USOD10k_USOD10k_TR_{k}_{n:05d}.png" for n in range(10) for k in ("RGB", "depth")]
    full = _run(USOD, names)
    part = _run(USOD, names, limit=3)
    assert len(part.depth.rows) == 3
    by_id = {r["ann_id"]: r for r in full.depth.rows}
    assert all(by_id[r["ann_id"]]["image_sha256"] == r["image_sha256"] for r in part.depth.rows)


def test_without_checksums_every_row_is_pending(tmp_path):
    spec = dt.DEPTH_PAIR_SOURCES["viame-public"]
    names = [
        f"3d_models_HabCam2019_dataset1_samples_20190628_1-{k}.tif"
        for k in ("left", "right", "disp", "rectified")
    ]
    names.append("3d_models_HabCam2019_dataset1_flounder_201901_20190628_204947394_15978.tif")
    res = _run(spec, names, checksums=False)
    assert not res.depth.rows and len(res.depth.pending) == 1 and res.depth.seen == 1
    pend = res.depth.pending[0]
    assert pend["image_key"].endswith("-left") and "image_sha256" not in pend
    assert (pend["units"], pend["gt_type"]) == ("disparity_px", "stereo")
    assert validate_pending("depth", res.depth.pending) == []
    path = pending_path(tmp_path, "depth", spec.source_id, spec.version)
    write_pending("depth", path, res.depth.pending)
    import pyarrow.parquet as pq

    cols = pq.read_table(path).column_names
    assert "image_key" in cols and "image_sha256" not in cols and "attrs" in cols


def test_sonarsweep_frame_gives_depth_and_three_pairs():
    spec = dt.DEPTH_PAIR_SOURCES["sonarsweep"]
    f = "vfov60hfov90_blue_water_visual_degraded_1_2"
    sfx = (
        "cam_left",
        "cam_right",
        "enhanced_gray_cam_left",
        "depth_left_visualize",
        "sonar",
        "sonar_rect",
        "cropped_cam_left",
    )
    res = _run(spec, [f"{f}_{s}.png" for s in sfx] + ["LIAS_OCEAN_textures_Water_Normal.png"])
    assert res.depth.seen == 1 and len(res.depth.rows) == 1 and len(res.pairs.rows) == 3
    assert res.depth.rows[0]["gt_type"] == "synthetic"
    assert {r["pair_role"] for r in res.pairs.rows} == {"stereo_right", "enhanced", "sonar"}
    validate_rows("depth", res.depth.rows)
    validate_rows("pairs", res.pairs.rows)
    assert len({r["ann_id"] for r in res.depth.rows + res.pairs.rows}) == 4


def test_depth_row_rejects_unknown_kind():
    with pytest.raises(ValueError, match="depth_kind"):
        dt.depth_row(
            spec=USOD,
            ordinal=0,
            sha=sha_of("a"),
            depth_ref="k",
            units="m",
            gt_type="sensor",
            depth_kind="lidar",
        )


def test_gaps_name_sources_with_no_producer():
    assert not set(dt.DEPTH_PAIR_GAPS) & set(dt.DEPTH_PAIR_SOURCES)
    assert {"flsea", "koi-rgb-sonar", "uw-stereodepth-40k"} <= set(dt.DEPTH_PAIR_GAPS)


def test_depth_config_reads_unified_rows_and_skips_pending(tmp_path):
    res = _run(USOD, NAMES)
    dt.write_depth(tmp_path, USOD.source_id, USOD.version, res.depth.rows)
    assert (
        len(
            read_annotations(
                tmp_path / "_annotations/depth/usod10k" / f"{USOD.version}.parquet", "depth"
            )
        )
        == 2
    )
    out = configs.build_depth_config(tmp_path)
    assert out.config_id == "depth" and len(out.rows) == 2
    assert {r["licence_class"] for r in out.rows} == {"internal-only"}
    assert "depth" in configs.CONFIG_IDS and "pairs" in configs.CONFIG_IDS
    assert "usod10k" in configs.DEPTH_CONFIG_SOURCES and "lsui" not in configs.DEPTH_CONFIG_SOURCES
