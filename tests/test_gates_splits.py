# ruff: noqa: E501
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from _gates_build import (
    IMAGE_COLS,
    MASK_COLS,
    META_COLS,
    SPLITS,
    Img,
    default_images,
    make_build,
    mask_rows,
    meta_row,
    write_config,
)

from marinedata import licence_class
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
    result = run_splits_gate(make_build(tmp_path, images=_imgs(18, 1, 1)), ratio_min_images=10)
    assert not result.passed and _has(result, "ratio_off_target")


def test_target_comes_from_release_record_and_registry_overrides_it(tmp_path):
    images = _imgs(14, 3, 3)  # 70/15/15, the registry default since v1.1.0
    floor = {"ratio_min_images": 10}  # the fixtures hold 20 images
    assert run_splits_gate(make_build(tmp_path / "a", images=images, ratio=None), **floor).passed
    off = _imgs(16, 2, 2)  # 80/10/10 is off the default target
    assert _has(
        run_splits_gate(make_build(tmp_path / "a2", images=off, ratio=None), **floor),
        "ratio_off_target",
    )
    ratio = {"source": "split_map", "target_pct": {"train": 70.0, "validation": 15.0, "test": 15.0}}
    assert run_splits_gate(make_build(tmp_path / "b", images=images, ratio=ratio), **floor).passed
    registry = tmp_path / "reg"
    (registry / "registry").mkdir(parents=True)
    text = "default: {train: 80, validation: 10, test: 10}\nconfigs:\n  images: {train: 50, validation: 25, test: 25}\n"
    for folder in (registry, registry / "registry"):
        (folder / "split_ratio.yaml").write_text(text)
    build = make_build(tmp_path / "c", images=images, ratio=ratio)
    assert _has(run_splits_gate(build, registry_root=registry, **floor), "ratio_off_target")


def test_validation_image_near_a_train_image_leaks(tmp_path):
    build = make_build(
        tmp_path, images=[*default_images(), Img(0, "validation", shift=6, group="gv")]
    )
    assert _has(run_splits_gate(build), "near_dup_leak")


def test_missing_split_group_column_fails(tmp_path):
    assert _has(run_splits_gate(make_build(tmp_path, group_column=False)), "no_split_group")


def test_config_without_all_three_splits_skips_the_ratio(tmp_path):
    images = [*(Img(i, "train") for i in range(18)), Img(18, "validation"), Img(19, "validation")]
    result = run_splits_gate(make_build(tmp_path, images=images), ratio_min_images=10)
    assert result.passed, result.failures
    assert result.counts["ratio_skipped_configs"] == 1  # images: no test split, none named


def test_registry_default_is_70_15_15_and_v1_configs_keep_their_measured_shares():
    default, per_config = load_split_targets()
    assert default == {"train": 70.0, "validation": 15.0, "test": 15.0}
    assert set(per_config) == {"coral-masks", "instance-masks", "scene-masks", "fish-boxes"}
    assert per_config["scene-masks"] == {"train": 73.9, "validation": 10.2, "test": 15.9}


# ratio_off_target judges the strata of the `images` config and leaves the pinned sources out
# (D-RATIO option A): a source the split map pins to its upstream test manifest has a structural
# share, so it is reported and never fails. The PNG fixtures hold 20 images a stratum: floor 10.

FLOOR = {"ratio_min_images": 10}


def _share(n: int, train: float, validation: float, test: float) -> dict:
    return {"n": n, "train": train, "validation": validation, "test": test}


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
    images = _pinned_images()  # every image 35/7.5/57.5; the non-pinned src-a alone 70/15/15
    plain = run_splits_gate(make_build(tmp_path / "a", images=images, ratio=None), **FLOOR)
    assert not plain.passed and _has(plain, "ratio_off_target")
    assert plain.counts["ratio_pins"] == {"map": None, "sources": [], "absent": 0}
    assert plain.counts["ratio_pinned_sources"] == {}

    build = make_build(tmp_path / "b", images=images, ratio=None)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build, **FLOOR)
    assert result.passed, result.failures
    held = {"src-p": {"train": 0, "validation": 0, "test": 20}}
    assert result.counts["ratio_pinned_sources"] == {"images": held, "metadata": held}
    assert result.counts["ratio_strata"] == {"src-a": _share(20, 70.0, 15.0, 15.0)}
    assert result.counts["ratio_left_out_images"] == {"pinned": 20}
    assert result.counts["ratio_pins"] == {"map": "build_dir", "sources": ["src-p"], "absent": 0}
    whole = {"rows": _share(40, 35.0, 7.5, 57.5), "images": _share(40, 35.0, 7.5, 57.5)}
    assert result.counts["ratio_by_config"]["images"] == whole  # informative: pinned rows included


def test_non_pinned_stratum_off_target_still_fails_next_to_a_pinned_source(tmp_path):
    build = make_build(tmp_path, images=_pinned_images(free=(18, 1, 1)), ratio=None)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build, **FLOOR)
    assert not result.passed
    line = next(f for f in result.failures if f.startswith("images/test ratio_off_target"))
    assert "5.0% of 20 images of src-a, target 15%" in line
    assert result.counts["ratio_strata"] == {"src-a": _share(20, 90.0, 5.0, 5.0)}
    assert result.counts["ratio_pinned_sources"]["images"]["src-p"]["test"] == 20
    assert result.counts["failed_ratio_off_target"] == 3  # train, validation and test of src-a


def test_ratio_finding_names_the_stratum_and_its_image_count(tmp_path):
    result = run_splits_gate(make_build(tmp_path, images=_imgs(18, 1, 1)), **FLOOR)
    line = next(f for f in result.failures if f.startswith("images/train ratio_off_target"))
    assert line.startswith(
        "images/train ratio_off_target: 90.0% of 20 images of src-a, target 80% +/- 5 ("
    )


def test_split_map_comes_from_release_dir_else_build_dir(tmp_path):
    build = make_build(tmp_path / "x", images=_pinned_images(), ratio=None)
    _write_map(build, _pins_block("src-gone"))  # a pin the build holds no row of changes nothing
    release = _write_map(tmp_path / "rel", _pins_block("src-p"))
    assert not run_splits_gate(build, **FLOOR).passed
    result = run_splits_gate(build, release_dir=release, **FLOOR)
    assert result.passed, result.failures
    assert result.counts["ratio_pins"] == {"map": "release_dir", "sources": ["src-p"], "absent": 0}
    elsewhere = run_gate("splits", build, GateOptions(release_dir=release))  # `--release-dir`
    assert elsewhere.counts["ratio_pins"]["map"] == "release_dir"
    (tmp_path / "bare").mkdir()
    bare = run_splits_gate(
        build, release_dir=tmp_path / "bare", **FLOOR
    )  # no map there: the build's
    assert bare.counts["ratio_pins"] == {"map": "build_dir", "sources": [], "absent": 1}


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
def test_no_usable_pins_keeps_every_image_in_its_stratum(tmp_path, content, where):
    build = make_build(tmp_path, images=_pinned_images(), ratio=None)
    if content is not None:
        _write_map(build, content)
    result = run_splits_gate(build, **FLOOR)
    assert not result.passed and _has(result, "ratio_off_target")
    assert result.counts["ratio_pins"] == {"map": where, "sources": [], "absent": 0}
    assert result.counts["ratio_pinned_sources"] == {}


def test_config_without_a_source_column_is_reported_and_not_ratio_checked(tmp_path):
    build = make_build(tmp_path, images=_pinned_images(), ratio=None)
    _write_map(build, _pins_block("src-p"))
    cols = {k: v for k, v in META_COLS.items() if k != "source_id"}
    plan = {"train": 18, "validation": 1, "test": 1}
    rows = {s: [meta_row(f"{s}{i}".rjust(64, "0")) for i in range(n)] for s, n in plan.items()}
    write_config(build, "boxes", rows, cols)
    result = run_splits_gate(build, **FLOOR)
    assert result.passed, result.failures  # a 90/5/5 table is shown and never held to a target
    assert result.counts["ratio_unsourced_configs"] == 1
    assert "boxes" not in result.counts["ratio_pinned_sources"]
    shares = _share(20, 90.0, 5.0, 5.0)
    assert result.counts["ratio_by_config"]["boxes"] == {"rows": shares, "images": shares}


def test_config_whose_images_are_all_pinned_is_skipped_not_divided_by_zero(tmp_path):
    plan = ["train"] * 14 + ["validation"] * 3 + ["test"] * 3
    images = [Img(i, s, "src-p") for i, s in enumerate(plan)]
    build = make_build(tmp_path, images=images, ratio=None, masks=False)
    _write_map(build, _pins_block("src-p"))
    result = run_splits_gate(build, **FLOOR)
    assert result.passed, result.failures
    assert result.counts["ratio_all_excluded_configs"] == 1  # images: every image is pinned
    assert result.counts["ratio_strata"] == {} and "ratio_skipped_configs" not in result.counts
    assert set(result.counts["ratio_pinned_sources"]) == {"images", "metadata"}


def test_stratum_without_a_split_fails_once_it_is_large_enough(tmp_path):
    images = _pinned_images(free=(18, 2, 0))  # src-a is 90/10/0; pinned src-p holds the test split
    build = make_build(tmp_path, images=images, ratio=None, masks=False)
    _write_map(build, _pins_block("src-p"))
    assert run_splits_gate(build).passed  # 20 images: under the default floor
    result = run_splits_gate(build, **FLOOR)
    assert any(
        f.startswith("images/test ratio_off_target: 0.0% of 20 images of src-a")
        for f in result.failures
    ), result.failures


# -- the stratum rule, on builds without pixels (one `images` row per image, no near-dup scan) --


def _sha(source: str, split: str, i: int) -> str:
    return hashlib.sha256(f"{source}/{split}/{i}".encode()).hexdigest()


def _strata_build(root: Path, plan: dict[str, tuple[int, int, int]]) -> Path:
    """``images`` holds ``plan[source] = (train, validation, test)`` images of each source."""
    build = make_build(root, images=[], masks=False, card=False, ratio=None)  # target: 70/15/15
    by_split: dict[str, list[dict]] = {}
    for source, sizes in plan.items():
        for split, n in zip(SPLITS, sizes, strict=True):
            by_split.setdefault(split, []).extend(
                {
                    "image_sha256": _sha(source, split, i),
                    "source_id": source,
                    "split_group": f"{source}/{split}/{i}",
                }
                for i in range(n)
            )
    write_config(build, "images", by_split, {k: v for k, v in IMAGE_COLS.items() if k != "image"})
    return build


def test_stratum_under_the_floor_is_reported_and_never_fails(tmp_path):
    build = _strata_build(tmp_path, {"big": (70, 15, 15), "tiny": (9, 0, 0)})  # tiny: all train
    result = run_splits_gate(build)
    assert result.passed, result.failures
    assert result.counts["ratio_min_images"] == 100
    assert result.counts["ratio_small_strata"] == {"tiny": 9}
    assert result.counts["ratio_strata"] == {"big": _share(100, 70.0, 15.0, 15.0)}
    assert _has(run_splits_gate(build, ratio_min_images=9), "ratio_off_target")


def test_dominated_fork_passes_pooled_and_would_fail_per_source(tmp_path):
    build = _strata_build(tmp_path, {"fork-a": (700, 150, 150), "fork-b": (48, 36, 36)})
    alone = run_splits_gate(build)  # no family: each source is a stratum, fork-b is 40/30/30
    assert alone.counts["failed_ratio_off_target"] == 3
    assert all("images of fork-b," in f for f in alone.failures), alone.failures
    pooled = run_splits_gate(build, families={"fork-a": "fam", "fork-b": "fam"})
    assert pooled.passed, pooled.failures
    assert pooled.counts["ratio_strata"] == {"fam": _share(1120, 66.8, 16.6, 16.6)}
    assert pooled.counts["ratio_small_strata"] == {}


def test_runner_hands_the_registry_family_map_to_the_splits_gate(tmp_path, monkeypatch):
    build = _strata_build(tmp_path, {"fork-a": (700, 150, 150), "fork-b": (48, 36, 36)})
    assert not run_gate("splits", build).passed  # fork-a and fork-b belong to no registry family
    family = {"fork-a": "fam", "fork-b": "fam"}
    monkeypatch.setattr(licence_class, "source_families", lambda root=None: family)
    result = run_gate("splits", build)
    assert result.passed, result.failures
    assert list(result.counts["ratio_strata"]) == ["fam"]


def test_images_with_an_upstream_split_label_are_left_out(tmp_path):
    build = _strata_build(tmp_path, {"mixed": (70, 15, 15), "upstream": (120, 0, 0)})
    assert _has(run_splits_gate(build), "ratio_off_target")  # 120 train images in one split
    cols = {"image_sha256": META_COLS["image_sha256"], "upstream_split": META_COLS["source_id"]}
    rows = [
        {"image_sha256": _sha("upstream", "train", i), "upstream_split": "train"}
        for i in range(120)
    ]
    rows += [  # an empty or missing label is no upstream partition
        {"image_sha256": _sha("mixed", "train", i), "upstream_split": label}
        for i, label in enumerate(["", None])
    ]
    write_config(build, "points", {"train": rows}, cols)
    result = run_splits_gate(build)
    assert result.passed, result.failures
    assert result.counts["ratio_left_out_images"] == {"upstream_split": 120}
    assert result.counts["ratio_strata"] == {"mixed": _share(100, 70.0, 15.0, 15.0)}


def test_ratio_by_config_reports_rows_and_distinct_images_and_never_fails(tmp_path):
    build = _strata_build(tmp_path, {"s": (70, 15, 15)})
    rows = {  # annotation rows: one per train and validation image, twenty per test image
        "train": [{"image_sha256": _sha("s", "train", i)} for i in range(70)],
        "validation": [{"image_sha256": _sha("s", "validation", i)} for i in range(15)],
        "test": [{"image_sha256": _sha("s", "test", i % 15)} for i in range(300)],
    }
    write_config(build, "points", rows, {"image_sha256": META_COLS["image_sha256"]})
    result = run_splits_gate(build)
    assert result.passed, result.failures
    by_config = result.counts["ratio_by_config"]
    assert by_config["points"] == {
        "rows": _share(385, 18.2, 3.9, 77.9),  # 70 / 15 / 300 rows: not steerable, not held
        "images": _share(100, 70.0, 15.0, 15.0),
    }
    assert by_config["images"]["rows"] == _share(100, 70.0, 15.0, 15.0)


def test_build_without_an_images_config_reports_the_ratio_as_skipped(tmp_path):
    result = run_splits_gate(make_build(tmp_path, images=[]))  # only the fixture-masks config
    assert result.counts["ratio_check"] == "skipped: no images config"
    assert not _has(result, "ratio_off_target")


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
