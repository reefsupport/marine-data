# ruff: noqa: E501
from __future__ import annotations

from _gates_build import Img, default_images, make_build

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
    images = _imgs(14, 3, 3)  # 70/15/15, the v1.0 code default
    assert _has(run_splits_gate(make_build(tmp_path / "a", images=images)), "ratio_off_target")
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


def test_registry_default_is_80_10_10():
    default, per_config = load_split_targets()
    assert default == {"train": 80.0, "validation": 10.0, "test": 10.0} and per_config == {}
