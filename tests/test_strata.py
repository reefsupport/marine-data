"""Stratified allocation (S7l): every stratum hits the ratios, not only the union.

The synthetic corpus is shaped like the S7k dry run that exposed the bug: one source
of a few hundred-image sites plus ~30 stations, one source of ~850 near-singleton
groups. Allocated as one pool, the big groups fill ``train`` and the singletons fill
``val``/``test``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marinedata.cli import main
from marinedata.scan import assign_splits
from marinedata.splitmap import SplitMapError, load_split_map, resolve_splits
from marinedata.strata import achieved_by_stratum, assign_splits_stratified

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}
TOLERANCE = 0.03


def _corpus() -> dict[str, dict[str, int]]:
    own = {f"own/site{i}": n for i, n in enumerate((222, 169, 146, 121, 105, 68, 51, 48, 32))}
    own.update({f"own/station{i}": 11 + i % 6 for i in range(20)})
    points = {f"points/{i:04d}": 2 if i % 60 == 0 else 1 for i in range(856)}
    return {"own": own, "points": points}


def _union(strata: dict[str, dict[str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for groups in strata.values():
        for key, n in groups.items():
            counts[key] = max(counts.get(key, 0), n)
    return counts


def _shares(per: dict[str, int]) -> dict[str, float]:
    total = sum(per.values())
    return {split: per.get(split, 0) / total for split in RATIOS}


def test_unstratified_union_skews_each_stratum() -> None:
    """The defect, pinned: one pool balances the union and skews both sources."""
    strata = _corpus()
    achieved = achieved_by_stratum(strata, assign_splits(_union(strata), RATIOS, seed=0))
    assert _shares(achieved["own"])["train"] > 0.85
    assert _shares(achieved["points"])["train"] < 0.5


def test_stratified_ratio_within_tolerance_per_stratum(tmp_path: Path) -> None:
    strata = _corpus()
    got = resolve_splits(
        tmp_path / "SPLIT_MAP.json", _union(strata), RATIOS, by="group", now="t",
        strata=strata, stratify="source",
    )  # fmt: skip
    for name, per in achieved_by_stratum(strata, got).items():
        for split, share in _shares(per).items():
            assert share == pytest.approx(RATIOS[split], abs=TOLERANCE), (name, split, share)


def test_disjoint_strata_reproduce_per_stratum_allocation() -> None:
    strata = _corpus()
    got = assign_splits_stratified(strata, RATIOS, seed=0)
    for groups in strata.values():
        expected = assign_splits(groups, RATIOS, seed=0)
        assert {key: got[key] for key in groups} == expected


def test_shared_cross_stratum_group_lands_in_one_split(tmp_path: Path) -> None:
    strata = _corpus()
    strata["own"]["seaview/12345"] = 40
    strata["points"]["seaview/12345"] = 30
    counts = _union(strata)
    path = tmp_path / "SPLIT_MAP.json"
    got = resolve_splits(path, counts, RATIOS, by="group", strata=strata, stratify="source")
    persisted = load_split_map(path)
    assert persisted is not None and persisted.stratify == "source"
    assert list(persisted.assignments).count("seaview/12345") == 1
    assert got["seaview/12345"] == persisted.assignments["seaview/12345"]
    # Both strata see that one split for it, and both stay balanced.
    for per in achieved_by_stratum(strata, got).values():
        assert _shares(per)["test"] == pytest.approx(0.15, abs=TOLERANCE)


def test_stratum_below_min_groups_is_train_only() -> None:
    strata = _corpus()
    strata["tiny"] = {"tiny/a": 50, "tiny/b": 40}
    got = assign_splits_stratified(strata, RATIOS, seed=0, min_groups=3)
    assert got["tiny/a"] == got["tiny/b"] == "train"
    # A shared group is decided by the balanced stratum it also belongs to.
    strata["tiny"] = {"tiny/a": 50, "own/site1": 7}
    got = assign_splits_stratified(strata, RATIOS, seed=0, min_groups=3)
    assert got["own/site1"] == assign_splits_stratified(_corpus(), RATIOS, seed=0)["own/site1"]


def test_stratified_map_extension_is_append_only_and_checked(tmp_path: Path) -> None:
    strata = _corpus()
    path = tmp_path / "SPLIT_MAP.json"
    first = resolve_splits(path, _union(strata), RATIOS, by="group", strata=strata,
                           stratify="source")  # fmt: skip
    strata["points"]["points/new"] = 1
    second = resolve_splits(path, _union(strata), RATIOS, by="group", strata=strata,
                            stratify="source")  # fmt: skip
    assert {k: second[k] for k in first} == first
    strata["points"]["points/newer"] = 1
    with pytest.raises(SplitMapError, match="stratify"):
        resolve_splits(path, _union(strata), RATIOS, by="group")


def test_generate_stratified_is_byte_identical_and_reports_strata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = ["image_sha256\tsplit_group\tstratum"]
    for name, groups in _corpus().items():
        for key, n in groups.items():
            rows += [f"{key}#{i}\t{key}\t{name}" for i in range(n)]
    rows.append("own/site0#0\town/site0\tpoints")  # one image staged by both sources
    src = tmp_path / "images.tsv"
    src.write_text("\n".join(rows) + "\n")
    outs = [tmp_path / "a.json", tmp_path / "b.json"]
    for out in outs:
        argv = ["splitmap", "generate", "--release", "r1", "--in", str(src), "--now",
                "2026-09-23T00:00:00Z", "--stratify", "source", "--out", str(out)]  # fmt: skip
        assert main(argv) == 0
    assert outs[0].read_bytes() == outs[1].read_bytes()
    assert json.loads(outs[0].read_text())["stratify"] == "source"
    printed = capsys.readouterr().out
    assert "stratum own:" in printed and "stratum points:" in printed


# --- WP-R6: no empty val/test per source ------------------------------------------------


def test_source_with_three_groups_gets_a_group_in_every_split() -> None:
    strata = {"a": {"g1": 90, "g2": 6, "g3": 4}}
    got = assign_splits_stratified(strata, RATIOS, seed=0, min_groups=3)
    assert sorted(got.values()) == ["test", "train", "val"]


def test_val_is_taken_from_train_never_from_the_upstream_test() -> None:
    # The R5b case: 14 groups forced to test by the upstream split, 7 left to allocate
    # (7 train / 14 test / 0 val before WP-R6).
    groups = {f"g{i:02d}": 10 for i in range(21)}
    pinned = {f"g{i:02d}": "test" for i in range(14)}
    got = assign_splits_stratified({"suim": groups}, RATIOS, seed=0, pinned=pinned)
    assert set(got) == {f"g{i:02d}" for i in range(14, 21)}  # pinned groups are never moved
    split_of = {**pinned, **got}
    per = {s: sum(1 for v in split_of.values() if v == s) for s in ("train", "val", "test")}
    assert per["val"] >= 1 and per["train"] >= 1
    assert per["test"] == 14  # nothing was taken from the upstream test


def test_a_shared_group_is_not_moved_if_it_empties_another_strata_split() -> None:
    strata = {
        "a": {"g1": 60, "g2": 20, "g3": 20, "shared": 10},
        "b": {"shared": 10, "b1": 10, "b2": 10},
    }
    got = assign_splits_stratified(strata, RATIOS, seed=3)
    for name, groups in strata.items():
        splits = {got[g] for g in groups}
        assert splits == {"train", "val", "test"}, (name, got)


def test_small_strata_stay_train_only() -> None:
    got = assign_splits_stratified({"a": {"g1": 5, "g2": 5}}, RATIOS, seed=0, min_groups=3)
    assert set(got.values()) == {"train"}
