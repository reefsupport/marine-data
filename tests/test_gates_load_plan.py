"""Resolving a card's ``configs:`` patterns against the build and the v1.0 tree (no datasets)."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from marinedata.cli_release_gate_v11 import wants_v11
from marinedata.gates.load_plan import (
    FROZEN_V10_CONFIGS,
    ConfigFiles,
    config_patterns,
    inherited_phantoms,
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
        "v11": False,
        "only": None,
        "hidden_test": None,
        "benchmark": None,
        "nc_id": None,
        "privacy_review": None,
        "taxonomy_version": None,
        "rehearsal": False,
        "build_dir": None,
        "v1_dir": "/v1",
    }
    assert wants_v11(Namespace(**flags), v10_selected=False)
    assert not wants_v11(Namespace(**flags), v10_selected=True)


# --- inherited v1.0 phantoms: a card pattern the frozen v1.0 tree never held a file for ---------

PHANTOM = {
    "train": "data/legacy-masks/train-*.parquet",
    "validation": "data/legacy-masks/validation-*.parquet",  # in the v1.0 card, no v1.0 shard
    "test": "data/legacy-masks/test-*.parquet",
}
V1_SHARDS = (
    "data/legacy-masks/train-00000-of-00001.parquet",
    "data/legacy-masks/test-00000-of-00001.parquet",
)


def _write_card(root: Path, entries: dict[str, dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    configs = [{"config_name": name, **entry} for name, entry in entries.items()]
    (root / "README.md").write_text(f"---\n{yaml.safe_dump({'configs': configs})}---\n\n# card\n")


def _phantom_trees(
    tmp_path: Path, *, v1_entry=None, v1_files=V1_SHARDS, v1_card=True, build_files=()
):
    """A build whose card repeats the v1.0 entry of ``legacy-masks`` (its shards live in v1.0 only)
    and a v1.0 tree holding the train and test shards of the config, but no validation shard."""
    build, v1 = tmp_path / "build", tmp_path / "v1"
    entry = _entry(**PHANTOM)
    _write_card(build, {"legacy-masks": entry})
    _touch(build, *build_files)
    _touch(v1, *v1_files)
    v1.mkdir(exist_ok=True)
    if v1_files:
        (v1 / "data" / "legacy-masks").mkdir(parents=True, exist_ok=True)
    if v1_card:
        _write_card(v1, {"legacy-masks": entry if v1_entry is None else v1_entry})
    return build, {"legacy-masks": entry}, v1


def test_an_inherited_v1_phantom_is_counted_not_missing(tmp_path):
    build, cards, v1 = _phantom_trees(tmp_path)
    [got] = plan_configs(cards, build, v1)
    assert got.resolved and got.missing == () and got.from_v1 == 2
    assert got.inherited == (("validation", PHANTOM["validation"]),)
    assert set(got.splits) == {"train", "test"}  # the phantom split simply has no files


def test_without_v1_dir_a_phantom_pattern_is_missing_as_before(tmp_path):
    build, cards, _ = _phantom_trees(tmp_path)
    [got] = plan_configs(cards, build)
    assert not got.inherited and len(got.missing) == 3 and not got.resolved


@pytest.mark.parametrize(
    "case",
    [
        "config folder absent in v1.0",
        "v1.0 entry differs",
        "v1.0 card missing",
        "v1.0 card unreadable",
        "config resolves no split",
    ],
)
def test_a_phantom_is_not_inherited_unless_all_conditions_hold(tmp_path, case):
    differs = _entry(train=PHANTOM["train"], test=PHANTOM["test"])  # v1.0 never listed validation
    kwargs = {
        "config folder absent in v1.0": {"v1_files": (), "build_files": V1_SHARDS},
        "v1.0 entry differs": {"v1_entry": differs},
        "v1.0 card missing": {"v1_card": False},
        "v1.0 card unreadable": {},
        "config resolves no split": {"v1_files": ()},
    }[case]
    build, cards, v1 = _phantom_trees(tmp_path, **kwargs)
    if case == "v1.0 card unreadable":
        (v1 / "README.md").write_text("---\nconfigs: [unclosed\n---\n")
    if case == "config resolves no split":
        (v1 / "data" / "legacy-masks").mkdir(parents=True)  # a folder, but no shard at all
    [got] = plan_configs(cards, build, v1)
    assert got.inherited == () and ("validation", PHANTOM["validation"]) in got.missing
    assert not got.resolved


def test_a_card_pattern_the_v1_tree_matches_is_not_inherited(tmp_path):
    _, cards, v1 = _phantom_trees(tmp_path)
    entry = cards["legacy-masks"]
    item = ConfigFiles("legacy-masks", {"train": (Path("x"),)}, (("test", PHANTOM["test"]),), 0)
    assert inherited_phantoms(item, entry, v1, {"legacy-masks": entry}) == frozenset()  # v1 has it
    item = ConfigFiles("legacy-masks", item.splits, (("validation", PHANTOM["validation"]),), 0)
    assert inherited_phantoms(item, entry, v1, {"legacy-masks": entry}) == set(item.missing)


def test_a_config_name_cannot_point_out_of_the_v1_data_folder(tmp_path):
    _, cards, v1 = _phantom_trees(tmp_path)
    entry = cards["legacy-masks"]
    item = ConfigFiles("..", {"train": (Path("x"),)}, (("validation", PHANTOM["validation"]),), 0)
    assert (v1 / "data" / "..").is_dir()  # the folder exists, the name must still be refused
    assert inherited_phantoms(item, entry, v1, {"..": entry}) == frozenset()
