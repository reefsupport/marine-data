"""Resolving a card's ``configs:`` patterns against the build and the v1.0 tree (no datasets)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from marinedata.cli_release_gate_v11 import wants_v11
from marinedata.gates.load_plan import (
    FROZEN_V10_CONFIGS,
    config_patterns,
    plan_configs,
    resolve_config,
)


def _touch(root: Path, *rels: str) -> None:
    for rel in rels:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")


def _entry(**splits: str) -> dict:
    return {"data_files": [{"split": s, "path": p} for s, p in splits.items()]}


def test_patterns_accept_every_data_files_shape():
    assert config_patterns({"data_files": "a/*.parquet"}) == {"train": ["a/*.parquet"]}
    assert config_patterns(_entry(train="a/t-*", test="a/x-*")) == {
        "train": ["a/t-*"],
        "test": ["a/x-*"],
    }
    listed = {
        "data_files": [
            {"split": "train", "path": ["a/1-*", "a/2-*"]},
            {"split": "train", "path": "a/3-*"},
        ]
    }
    assert config_patterns(listed) == {"train": ["a/1-*", "a/2-*", "a/3-*"]}
    assert config_patterns({"data_files": {"validation": ["v/*"]}}) == {"validation": ["v/*"]}
    assert config_patterns({"data_files": ["a/x", {"path": "a/y"}]}) == {"train": ["a/x", "a/y"]}
    assert config_patterns({}) == {} and config_patterns(None) == {}


def test_a_pattern_resolves_in_the_build_first_then_in_v1(tmp_path):
    build, v1 = tmp_path / "build", tmp_path / "v1"
    _touch(build, "data/images/train-00000-of-00001.parquet")
    _touch(
        v1, "data/images/train-00009-of-00009.parquet", "data/images/test-00000-of-00001.parquet"
    )
    entry = _entry(train="data/images/train-*.parquet", test="data/images/test-*.parquet")
    got = resolve_config("images", entry, build, v1)
    assert got.resolved and got.from_v1 == 1
    assert [p.name for p in got.splits["train"]] == ["train-00000-of-00001.parquet"]  # build wins
    assert got.splits["test"][0].parent == v1 / "data" / "images"


def test_a_pattern_matching_in_neither_tree_is_missing(tmp_path):
    build = tmp_path / "build"
    _touch(build, "data/images/train-00000-of-00001.parquet")
    entry = _entry(train="data/images/train-*.parquet", test="data/images/test-*.parquet")
    got = resolve_config("images", entry, build, None)
    assert not got.resolved and set(got.splits) == {"train"}
    assert got.missing == (("test", "data/images/test-*.parquet"),)


def test_patterns_cannot_leave_the_tree(tmp_path):
    (tmp_path / "secret.parquet").write_bytes(b"x")
    build = tmp_path / "build"
    build.mkdir()
    for bad in ("../secret.parquet", str(tmp_path / "secret.parquet")):
        assert not resolve_config("x", _entry(train=bad), build, build).resolved


def test_an_entry_without_data_files_is_unresolved(tmp_path):
    assert not resolve_config("x", {"config_name": "x"}, tmp_path, None).resolved


def test_plan_keeps_readme_order_and_flags_frozen_configs(tmp_path):
    build = tmp_path / "build"
    _touch(build, "data/images/train-00000-of-00001.parquet")
    cards = {
        "coral-masks": _entry(train="data/coral-masks/train-*.parquet"),
        "images": _entry(train="data/images/train-*.parquet"),
    }
    plan = plan_configs(cards, build)
    assert [p.config for p in plan] == ["coral-masks", "images"]
    assert plan[0].frozen and not plan[0].resolved
    assert plan[1].resolved and not plan[1].frozen
    assert {"coral-masks", "scene-masks", "instance-masks", "fish-boxes"} == FROZEN_V10_CONFIGS


def test_v1_dir_selects_the_v11_gates_only_without_a_v10_gate_flag():
    flags = {
        "v11": False, "only": None, "hidden_test": None, "benchmark": None, "nc_id": None,
        "taxonomy_version": None, "rehearsal": False, "build_dir": None, "v1_dir": "/v1",
    }  # fmt: skip
    assert wants_v11(Namespace(**flags), v10_selected=False)
    assert not wants_v11(Namespace(**flags), v10_selected=True)
