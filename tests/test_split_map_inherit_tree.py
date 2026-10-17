"""``release split-map --inherit-tree``: the published v1.0 tree is the prior, per image (PR-1).

The world is ``test_release_split_map_inherit._world``; its prior map becomes a v1.0 Hub tree
(``data/<config>/<split>-<source>-*.parquet`` with ``image_sha256``). Gate G6 then reads a build
whose metadata follows the generated map, with that tree as ``--v1-dir``."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from test_global_split_map_flavours import NOW
from test_release_split_map_inherit import _img, _world

from marinedata.benchmark_pins import BenchmarkPins
from marinedata.cli_release_inherit import add_inherit_args, inherit_kwargs
from marinedata.gates.splits_v11 import run_splits_v11_gate
from marinedata.release import generate_split_map
from marinedata.splitmap import SplitMapError, load_split_map
from marinedata.splitmap_tree import load_v1_tree

HUB = {"val": "validation"}


def _sha(tag: str) -> str:
    return hashlib.sha256(_img(tag)).hexdigest()


def _rows(world) -> list[tuple[str, str, str]]:  # type: ignore[no-untyped-def]
    """``(image_sha256, split_group, source)`` of every staged image (sha256 of its bytes)."""
    rows: set[tuple[str, str, str]] = set()
    for source, root in sorted(world.roots.items()):
        stems = {path.stem: path for path in Path(root, "images").rglob("*.img")}
        for record in pq.read_table(Path(root, "metadata.parquet")).to_pylist():
            stem = next(v for v in record.values() if isinstance(v, str) and v in stems)
            sha = hashlib.sha256(stems[stem].read_bytes()).hexdigest()
            rows.add((sha, record["split_group"], source))
    return sorted(rows)


def _tree(tmp_path: Path, world, rows) -> Path:  # type: ignore[no-untyped-def]
    """v1.0 as published: the prior map per image, old/g10 split across train and test (a group
    v1.0 itself straddled) and deepfish/d0 (upstream test) in train."""
    split_of: dict[str, str] = {}
    for sha, group, source in rows:
        if source == "src-old":
            split = world.prior.assignments[group]
            split_of[sha] = HUB.get(split, split)
        elif group == "deepfish/d0":
            split_of[sha] = "train"
    split_of[_sha("old-g10-b")] = "test"
    root = tmp_path / "v1.0-repo"
    for hub in sorted(set(split_of.values())):
        path = root / "data" / "coral-masks" / f"{hub}-src-old-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        shas = sorted(sha for sha, s in split_of.items() if s == hub)
        pq.write_table(pa.table({"image_sha256": shas}), path)
    return root


def _generate(world, build: Path, tree: Path, **kwargs):  # type: ignore[no-untyped-def]
    build.mkdir(parents=True, exist_ok=True)
    stats = generate_split_map(
        world.registry,
        out=build / "SPLIT_MAP.json",
        roots=world.roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=11,
        min_groups=3,
        near_dup=world.near,
        inherit_tree=load_v1_tree(tree),
        **kwargs,
    )
    return stats, load_split_map(build / "SPLIT_MAP.json")


def _gate(build: Path, tree: Path, rows, got):  # type: ignore[no-untyped-def]
    """G6 on a build whose metadata follows ``got`` (withheld groups ship nothing)."""
    (build / "task_layers").mkdir(exist_ok=True)
    report = {"unsplit_images": 0, "allow_unsplit": False}
    (build / "task_layers" / "V11_TABLES.json").write_text(json.dumps(report))
    by_split: dict[str, list[dict]] = {}
    for sha, group, source in rows:
        if (split := got.assignments.get(group)) is not None:
            hub = HUB.get(split, split)
            row = {"image_sha256": sha, "split": hub, "split_group": group, "source_id": source}
            by_split.setdefault(hub, []).append(row)
    for hub, records in sorted(by_split.items()):
        path = build / "data" / "metadata" / f"{hub}-00000-of-00001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(records), path)
    return run_splits_v11_gate(build, v1_dir=tree)


def _failed(result, kind: str) -> int:  # type: ignore[no-untyped-def]
    return result.counts.get(f"failed_{kind}", 0)


def test_every_v1_image_keeps_its_split_near_copies_follow_and_the_band_holds(
    tmp_path, monkeypatch
):
    world = _world(tmp_path, monkeypatch)
    rows = _rows(world)
    assert _sha("old-g00-a") in {sha for sha, _, _ in rows}
    tree = _tree(tmp_path, world, rows)
    _, got = _generate(world, tmp_path / "build", tree)
    a = got.assignments
    for group, split in world.prior.assignments.items():
        if group != "old/g10":
            assert a[group] == split, group  # a hash source: v1.0 kept, never withheld
    assert a["deepfish/d0"] == "train"  # an upstream-test group v1.0 trained on stays train
    assert a["new/h00"] == "val" and a["new/h01"] == "test"  # same bytes / dHash <= 4 copy
    assert a["new/h06"] == "train" and a["new/h07"] == "test"  # band 5-8: never-eval / -train
    assert {"new/h02", "new/h08", "old/g10"} <= set(got.withheld) and "old/g10" not in a
    straddles = got.inherits["v1_tree"]["straddles"]
    assert straddles == {
        "components": 1,
        "group_ids": ["old/g02", "old/g04"],
        "withheld_groups": 1,
        "withheld_group_ids": ["old/g10"],
    }
    assert "sha256" not in got.inherits  # no map prior
    result = _gate(tmp_path / "build", tree, rows, got)
    assert _failed(result, "reassigned_v1") == 0 and result.passed, result.failures


def test_a_pin_move_is_listed_a_held_out_v1_val_stays_and_a_band_train_pin_is_withheld(
    tmp_path, monkeypatch
):
    world = _world(tmp_path, monkeypatch)
    rows = _rows(world)
    tree = _tree(tmp_path, world, rows)
    pins = BenchmarkPins(
        by_sha={
            _sha("old-g03-a"): ("test", "bench"),  # v1.0 train: the pin moves it, listed
            _sha("old-g09-a"): ("validation", "bench"),  # v1.0 val, held-out: stays put
            _sha("new-h06-a"): ("test", "bench"),  # near v1.0 train: never-eval vs the pin
        }
    )
    stats, got = _generate(world, tmp_path / "build", tree, benchmark_pins=pins)
    a = got.assignments
    assert a["old/g03"] == "test" and a["old/g09"] == "val"
    assert "new/h06" in got.withheld and "new/h06" not in a
    over = got.pins["v1_continuity_overridden"]
    assert (over["groups"], over["group_ids"]) == (1, ["old/g03"])
    assert stats.pins["skipped_components"] == {"ood": 0, "earlier_rule": 0, "withheld": 1}
    result = _gate(tmp_path / "build", tree, rows, got)
    assert _failed(result, "reassigned_v1") == 0 and _failed(result, "override_unchanged") == 0
    assert result.passed, result.failures


def test_a_map_prior_applies_only_to_groups_the_tree_leaves_untouched(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    rows = _rows(world)
    tree = _tree(tmp_path, world, rows)
    extra = {"new/h15": "test", "old/g11": "val"}  # old/g11 is train in the v1.0 tree
    prior = replace(world.prior, assignments={**world.prior.assignments, **extra})
    _, got = _generate(world, tmp_path / "build", tree, inherit=prior, inherit_sha256="ab" * 32)
    assert got.assignments["old/g11"] == "train" and got.assignments["new/h15"] == "test"
    assert got.inherits["groups_kept"] == 1 and got.inherits["sha256"] == "ab" * 32
    old_images = {sha for sha, _, source in rows if source == "src-old"}
    assert got.inherits["v1_tree"]["images"] == len(old_images) + 2  # + deepfish/d0 (a, b)


def test_the_cli_reads_the_tree_and_refuses_a_directory_without_data(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    tree = _tree(tmp_path, world, _rows(world))
    parser = argparse.ArgumentParser()
    add_inherit_args(parser)
    args = parser.parse_args(["--inherit-tree", str(tree), "--no-ood"])
    kwargs = inherit_kwargs(args, world.registry)
    assert set(kwargs) == {"inherit_tree"} and kwargs["inherit_tree"].header["images"] > 0
    with pytest.raises(SplitMapError, match="no data/ directory"):
        load_v1_tree(tmp_path)
