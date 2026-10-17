# ruff: noqa: E501
from __future__ import annotations

import json
from pathlib import Path

import pytest
from _gates_build import (
    MASK_COLS,
    META_COLS,
    Img,
    default_images,
    make_build,
    mask_rows,
    meta_row,
    write_config,
)

from marinedata.gates.runner import GateOptions, run_gate
from marinedata.gates.shards import config_shards, source_split_counts
from marinedata.gates.splits import load_split_targets, run_splits_gate


def _has(result, kind: str) -> bool:
    return any(f"{kind}:" in line for line in result.failures)


def _imgs(train: int, validation: int, test: int) -> list[Img]:
    plan = ["train"] * train + ["validation"] * validation + ["test"] * test
    return [Img(i, s) for i, s in enumerate(plan)]


def test_clean_build_passes(tmp_path):
    result = run_splits_gate(make_build(tmp_path))
    assert result.passed, result.failures
    assert result.counts["split_groups"] == 20


def test_group_straddling_two_splits_fails(tmp_path):
    build = make_build(tmp_path, images=[*default_images(), Img(30, "test", group="g0")])
    assert _has(run_splits_gate(build), "group_straddles_splits")


def test_ratio_off_target_fails(tmp_path):
    result = run_splits_gate(make_build(tmp_path, images=_imgs(18, 1, 1)))
    assert not result.passed and _has(result, "ratio_off_target")


def test_target_comes_from_release_record_and_registry_overrides_it(tmp_path):
    images = _imgs(14, 3, 3)  # 70/15/15, the registry default since v1.1.0
    assert run_splits_gate(make_build(tmp_path / "a", images=images, ratio=None)).passed
    off = _imgs(16, 2, 2)  # 80/10/10 is off the default target
    assert _has(
        run_splits_gate(make_build(tmp_path / "a2", images=off, ratio=None)), "ratio_off_target"
    )
    ratio = {"source": "split_map", "target_pct": {"train": 70.0, "validation": 15.0, "test": 15.0}}
    assert run_splits_gate(make_build(tmp_path / "b", images=images, ratio=ratio)).passed
    registry = tmp_path / "reg"
    (registry / "registry").mkdir(parents=True)
    text = "default: {train: 80, validation: 10, test: 10}\nconfigs:\n  images: {train: 50, validation: 25, test: 25}\n"
    for folder in (registry, registry / "registry"):
        (folder / "split_ratio.yaml").write_text(text)
    build = make_build(tmp_path / "c", images=images, ratio=ratio)
    assert _has(run_splits_gate(build, registry_root=registry), "ratio_off_target")


def test_validation_image_near_a_train_image_leaks(tmp_path):
    build = make_build(
        tmp_path, images=[*default_images(), Img(0, "validation", shift=6, group="gv")]
    )
    assert _has(run_splits_gate(build), "near_dup_leak")


def test_missing_split_group_column_fails(tmp_path):
    assert _has(run_splits_gate(make_build(tmp_path, group_column=False)), "no_split_group")


def test_config_without_all_three_splits_skips_the_ratio(tmp_path):
    images = [*(Img(i, "train") for i in range(18)), Img(18, "validation"), Img(19, "validation")]
    result = run_splits_gate(make_build(tmp_path, images=images))
    assert result.passed, result.failures
    assert result.counts["ratio_skipped_configs"] >= 1


def test_registry_default_is_70_15_15_and_v1_configs_keep_their_measured_shares():
    default, per_config = load_split_targets()
    assert default == {"train": 70.0, "validation": 15.0, "test": 15.0}
    assert set(per_config) == {"coral-masks", "instance-masks", "scene-masks", "fish-boxes"}
    assert per_config["scene-masks"] == {"train": 73.9, "validation": 10.2, "test": 15.9}


# ratio_off_target counts the NON-pinned sources only (D-RATIO option A): a source the split map
# pins to its upstream test manifest has a structural share, so it is reported and never fails.


def _pins_block(*pinned: str) -> dict:
    """A real SPLIT_MAP.json ``pins`` block. ``manifest_rows`` also lists ``src-a``, which has no
    pinned image in the release (it counts manifest digests, held or not): it must stay unpinned."""
    held = {s: {"test": 5, "validation": 0} for s in pinned}
    return {
        "pins": {
            "rule": "route-to-our-test manifests",
            "benchmarks": held,
            "manifest_rows": {"src-a": {"test": 3, "validation": 0}, **held},
            "components": {"test": 5, "validation": 0},
            "skipped_components": {"ood": 0, "earlier_rule": 0, "withheld": 0},
            "v1_continuity_overridden": {"images": 0, "groups": 0, "by_source": {}},
        }
    }


def _write_map(folder: Path, data: object) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "SPLIT_MAP.json").write_text(data if isinstance(data, str) else json.dumps(data))
    return folder


def _pinned_images(free=(14, 3, 3), pinned_test: int = 20) -> list[Img]:
    """``src-a`` split ``free`` (train, validation, test), plus ``src-p``: all of it in test."""
    plan = ["train"] * free[0] + ["validation"] * free[1] + ["test"] * free[2]
    return [
        *(Img(i, s, "src-a") for i, s in enumerate(plan)),
        *(Img(100 + i, "test", "src-p") for i in range(pinned_test)),
    ]


def test_pinned_source_that_is_all_test_does_not_trip_the_ratio(tmp_path):
    images = _pinned_images()  # every row 35/7.5/57.5; the non-pinned src-a alone 70/15/15
    plain = run_splits_gate(make_build(tmp_path / "a", images=images, ratio=None))
    assert not plain.passed and _has(plain, "ratio_off_target")
    assert plain.counts["ratio_pins"] == {"map": None, "sources": []}
    assert plain.counts["ratio_pinned_sources"] == {}

    build = make_build(tmp_path / "b", images=images, ratio=None)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build)
    assert result.passed, result.failures
    held = {"src-p": {"train": 0, "validation": 0, "test": 20}}
    assert result.counts["ratio_pinned_sources"] == {"images": held, "metadata": held}
    free = {"rows": 20, "train": 70.0, "validation": 15.0, "test": 15.0}
    assert result.counts["ratio_excluding_pinned"] == {"images": free, "metadata": free}
    assert result.counts["ratio_pins"] == {"map": "build_dir", "sources": ["src-p"]}


def test_non_pinned_table_off_target_still_fails_next_to_a_pinned_source(tmp_path):
    build = make_build(tmp_path, images=_pinned_images(free=(18, 1, 1)), ratio=None)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build)
    assert not result.passed
    line = next(f for f in result.failures if f.startswith("images/test ratio_off_target"))
    assert "5.0% of 20 non-pinned rows (excl. 20 rows of pinned src-p), target 15%" in line
    free = {"rows": 20, "train": 90.0, "validation": 5.0, "test": 5.0}
    assert result.counts["ratio_excluding_pinned"]["images"] == free
    assert result.counts["ratio_pinned_sources"]["images"]["src-p"]["test"] == 20
    assert result.counts["failed_ratio_off_target"] == 6  # three splits in each of the two tables


def test_ratio_finding_reads_as_before_when_no_row_was_left_out(tmp_path):
    result = run_splits_gate(make_build(tmp_path, images=_imgs(18, 1, 1)))
    line = next(f for f in result.failures if f.startswith("images/train ratio_off_target"))
    assert line.startswith("images/train ratio_off_target: 90.0% of 20 rows, target 80% +/- 5 (")


def test_split_map_comes_from_release_dir_else_build_dir(tmp_path):
    build = make_build(tmp_path / "x", images=_pinned_images(), ratio=None)
    _write_map(build, _pins_block("src-gone"))  # a pin the build holds no row of changes nothing
    release = _write_map(tmp_path / "rel", _pins_block("src-p"))
    assert not run_splits_gate(build).passed
    result = run_splits_gate(build, release_dir=release)
    assert result.passed, result.failures
    assert result.counts["ratio_pins"] == {"map": "release_dir", "sources": ["src-p"]}
    elsewhere = run_gate("splits", build, GateOptions(release_dir=release))  # `--release-dir`
    assert elsewhere.passed, elsewhere.failures
    (tmp_path / "bare").mkdir()
    bare = run_splits_gate(build, release_dir=tmp_path / "bare")  # no map there: the build's
    assert bare.counts["ratio_pins"] == {"map": "build_dir", "sources": ["src-gone"]}


@pytest.mark.parametrize(
    ("content", "where"),
    [
        (None, None),
        ("{not json", "unreadable"),
        ("[]", "build_dir"),
        ({}, "build_dir"),
        ({"pins": {}}, "build_dir"),
        ({"pins": {"benchmarks": ["src-p"]}}, "build_dir"),
        ({"pins": {"manifest_rows": {"src-p": {"test": 5}}}}, "build_dir"),
    ],
)
def test_no_usable_pins_keeps_the_ratio_over_every_row(tmp_path, content, where):
    build = make_build(tmp_path, images=_pinned_images(), ratio=None)
    if content is not None:
        _write_map(build, content)
    result = run_splits_gate(build)
    assert not result.passed and _has(result, "ratio_off_target")
    assert result.counts["ratio_pins"] == {"map": where, "sources": []}
    assert result.counts["ratio_pinned_sources"] == {}


def test_config_without_a_source_column_keeps_every_row(tmp_path):
    build = make_build(tmp_path, images=_pinned_images(), ratio=None)
    _write_map(build, _pins_block("src-p"))
    cols = {k: v for k, v in META_COLS.items() if k != "source_id"}
    plan = {"train": 18, "validation": 1, "test": 1}
    rows = {s: [meta_row(f"{s}{i}".ljust(64, "0")) for i in range(n)] for s, n in plan.items()}
    write_config(build, "boxes", rows, cols)
    result = run_splits_gate(build)
    assert result.counts["ratio_unsourced_configs"] == 1
    assert "boxes" not in result.counts["ratio_pinned_sources"]
    assert result.counts["ratio_excluding_pinned"]["boxes"]["train"] == 90.0
    assert any(
        f.startswith("boxes/train ratio_off_target: 90.0% of 20 rows,") for f in result.failures
    )
    assert not any(f.startswith(("images/", "metadata/")) for f in result.failures)


def test_config_whose_rows_are_all_pinned_is_skipped_not_divided_by_zero(tmp_path):
    registry = tmp_path / "reg"
    (registry / "registry").mkdir(parents=True)
    text = "default: {train: 70, validation: 15, test: 15}\nconfigs:\n  images: {train: 70, validation: 15, test: 15}\n"
    for folder in (registry, registry / "registry"):
        (folder / "split_ratio.yaml").write_text(text)
    plan = ["train"] * 14 + ["validation"] * 3 + ["test"] * 3
    images = [Img(i, s, "src-p") for i, s in enumerate(plan)]
    build = make_build(tmp_path / "b", images=images, ratio=None, masks=False)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build, registry_root=registry)
    assert result.passed, result.failures
    assert result.counts["ratio_all_pinned_configs"] == 1  # images: named by the registry
    assert result.counts["ratio_skipped_configs"] == 1  # metadata: nothing left, no split left
    assert result.counts["ratio_excluding_pinned"] == {}
    assert set(result.counts["ratio_pinned_sources"]) == {"images", "metadata"}


def test_table_whose_non_pinned_rows_lack_a_split_is_not_ratio_checked(tmp_path):
    images = _pinned_images(free=(18, 2, 0))  # every row 45/5/50, but src-a has no test row
    plain = make_build(tmp_path / "a", images=images, ratio=None, masks=False)
    assert not run_splits_gate(plain).passed
    build = make_build(tmp_path / "b", images=images, ratio=None, masks=False)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build)
    assert result.passed, result.failures
    assert result.counts["ratio_skipped_configs"] == 2  # images and metadata: no test row left
    assert result.counts["ratio_excluding_pinned"] == {}


def test_source_split_counts_reads_source_id_else_source_else_none(tmp_path):
    by_id = {
        "train": [meta_row(source="s1"), meta_row(source="s2")],
        "test": [meta_row(source="s1")],
    }
    write_config(tmp_path, "by-id", by_id, META_COLS)
    write_config(tmp_path, "by-source", {"train": mask_rows(n=3)}, MASK_COLS)
    write_config(
        tmp_path, "neither", {"train": [meta_row()]}, {"image_sha256": META_COLS["image_sha256"]}
    )
    shards = config_shards(tmp_path)
    assert source_split_counts(shards["by-id"]) == {
        "s1": {"test": 1, "train": 1},
        "s2": {"train": 1},
    }
    assert source_split_counts(shards["by-source"]) == {"src-a": {"train": 3}}
    assert source_split_counts(shards["neither"]) is None
