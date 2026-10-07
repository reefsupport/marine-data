"""SPLIT_MAP.json persistence — the guarantee `assign_splits` alone cannot make.

`assign_splits` re-sorts every group on every call, so a growing group can cross
another in the sort order and flip its assigned split even at a fixed seed. These tests
pin the fix: once a group's split is written to the map, it never changes, and the map
is shared by every task that scans the same registry.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marinedata.scan import assign_splits
from marinedata.splitmap import SplitMapError, load_split_map, resolve_splits

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Found by search against the real allocator: g7 growing 5->6 flips test->val and
# cascades into g6 (val->test) at seed=0 — the Phase 0 defect this map fixes.
PHASE0_BEFORE = {"g0": 17, "g1": 35, "g2": 39, "g3": 10, "g4": 20, "g5": 7, "g6": 5, "g7": 5}
PHASE0_AFTER = {**PHASE0_BEFORE, "g7": 6}


def test_split_map_absent_reproduces_current_allocation(tmp_path: Path) -> None:
    """No file yet: `resolve_splits` must match plain `assign_splits` exactly — a first
    run over an existing corpus does not silently re-split it."""
    counts = {f"site{i}": 100 * (i + 1) for i in range(20)}
    path = tmp_path / "SPLIT_MAP.json"
    assert not path.exists()

    resolved = resolve_splits(path, counts, RATIOS, seed=7)

    assert resolved == assign_splits(counts, RATIOS, seed=7)
    assert path.exists()


def test_split_map_growth_in_existing_group_never_reassigns(tmp_path: Path) -> None:
    """Phase 0 regression: growing an already-persisted group must not flip its split,
    even though the raw allocator (unchanged, and correctly so) still would."""
    raw_before = assign_splits(PHASE0_BEFORE, RATIOS, seed=0)
    raw_after = assign_splits(PHASE0_AFTER, RATIOS, seed=0)
    assert raw_before["g7"] != raw_after["g7"], "fixture must reproduce the raw flip"
    assert raw_before["g6"] != raw_after["g6"], "fixture must reproduce the cascade"

    path = tmp_path / "SPLIT_MAP.json"
    persisted_before = resolve_splits(path, PHASE0_BEFORE, RATIOS, seed=0)
    persisted_after = resolve_splits(path, PHASE0_AFTER, RATIOS, seed=0)

    assert persisted_after == persisted_before, "no key's assigned split may change"


def test_split_map_new_group_appends_without_moving_existing(tmp_path: Path) -> None:
    """A new group large enough to displace several existing groups under a full
    re-sort must still leave every persisted key untouched, and land against the
    remaining quota itself."""
    path = tmp_path / "SPLIT_MAP.json"
    first = resolve_splits(path, PHASE0_BEFORE, RATIOS, seed=0)

    grown = {**PHASE0_BEFORE, "g8": 200}
    full_resort = assign_splits(grown, RATIOS, seed=0)
    assert any(full_resort[k] != first[k] for k in PHASE0_BEFORE), (
        "fixture must actually cross rank order under a full re-sort"
    )

    second = resolve_splits(path, grown, RATIOS, seed=0)
    for key, split in first.items():
        assert second[key] == split, f"{key} moved from {split} to {second[key]}"
    assert second["g8"] in RATIOS
    assert set(second) == set(grown)


def test_split_map_shared_across_task_ids(tmp_path: Path) -> None:
    """The map is keyed by group only — no task id — so a site scanned under one task
    and again under a different one lands in the same split both times."""
    path = tmp_path / "SPLIT_MAP.json"
    shared_site = "reef_support/SEAFLOWER_BOLIVAR"

    benthic_segmentation = {shared_site: 900, "reef_support/OTHER_SITE": 4000, "coralscop/p1": 1500}
    pretrain = {shared_site: 900, "benthicnet/p2": 50_000, "benthicnet/p3": 20_000}

    from_benthic_task = resolve_splits(path, benthic_segmentation, RATIOS, seed=0)
    from_pretrain_task = resolve_splits(path, pretrain, RATIOS, seed=0)

    assert from_pretrain_task[shared_site] == from_benthic_task[shared_site]


def test_split_map_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    """Stable key order and a trailing newline: two runs on the same input give the
    same bytes, regardless of the order counts/ratios were passed in."""
    counts = {f"site{i}": 100 * (i + 1) for i in range(20)}
    reordered_counts = dict(reversed(counts.items()))
    reordered_ratios = {"test": 0.15, "val": 0.15, "train": 0.7}
    now = "2026-09-18T00:00:00+00:00"

    path_a = tmp_path / "a" / "SPLIT_MAP.json"
    path_b = tmp_path / "b" / "SPLIT_MAP.json"
    resolve_splits(path_a, counts, RATIOS, seed=3, now=now)
    resolve_splits(path_b, reordered_counts, reordered_ratios, seed=3, now=now)

    raw_a, raw_b = path_a.read_text(), path_b.read_text()
    assert raw_a == raw_b
    assert raw_a.endswith("\n") and not raw_a.endswith("\n\n")


def test_split_map_file_format_matches_schema(tmp_path: Path) -> None:
    path = tmp_path / "SPLIT_MAP.json"
    resolve_splits(path, {"g0": 5, "g1": 10}, RATIOS, seed=0, by="site", now="2026-09-18T00:00:00Z")

    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 1
    assert raw["by"] == "site"
    assert raw["seed"] == 0
    assert raw["ratios"] == RATIOS
    assert raw["generated_at"] == "2026-09-18T00:00:00Z"
    assert set(raw["assignments"]) == {"g0", "g1"}

    loaded = load_split_map(path)
    assert loaded is not None
    assert loaded.assignments == raw["assignments"]


def test_split_map_absent_file_loads_as_none(tmp_path: Path) -> None:
    assert load_split_map(tmp_path / "does-not-exist.json") is None


def test_site_map_rejected_for_group(tmp_path: Path) -> None:
    """A map generated `by="site"` cannot be extended by a `by="group"` request — mixing
    keying strategies under one file would silently reinterpret every existing key."""
    path = tmp_path / "SPLIT_MAP.json"
    resolve_splits(path, {"a/site0": 10, "a/site1": 20}, RATIOS, seed=0, by="site")

    with pytest.raises(SplitMapError, match="by="):
        resolve_splits(path, {"g0": 5}, RATIOS, seed=0, by="group")


def test_frozen_map_raises_on_new_group(tmp_path: Path) -> None:
    """`frozen=True` is the release-build contract: every group must already be in the
    map, and a call that finds one missing must write nothing — not even a partial
    append — so a release can never quietly grow the shared map."""
    path = tmp_path / "SPLIT_MAP.json"
    counts = {"g0": 10, "g1": 20, "g2": 5}
    resolve_splits(path, counts, RATIOS, seed=0)
    before = path.read_text()

    with pytest.raises(SplitMapError, match="frozen"):
        resolve_splits(path, {**counts, "g3": 3}, RATIOS, seed=0, frozen=True)

    assert path.read_text() == before, "a frozen call must not write anything, even on failure"

    # A frozen call over exactly the persisted groups succeeds and changes nothing.
    resolved = resolve_splits(path, counts, RATIOS, seed=0, frozen=True)
    assert resolved == resolve_splits(path, counts, RATIOS, seed=0)
    assert path.read_text() == before, "a frozen call must never write, even when it succeeds"


def test_frozen_map_raises_when_file_absent(tmp_path: Path) -> None:
    path = tmp_path / "SPLIT_MAP.json"
    with pytest.raises(SplitMapError, match="frozen"):
        resolve_splits(path, {"g0": 5}, RATIOS, seed=0, frozen=True)
    assert not path.exists()


def test_split_map_mismatched_ratios_raises(tmp_path: Path) -> None:
    """Extending a map under different ratios/seed/by would silently reinterpret a
    stale map, so it must raise instead — the same "raise, don't warn" contract as the
    rest of the gate."""
    path = tmp_path / "SPLIT_MAP.json"
    counts = {"g0": 10, "g1": 20, "g2": 5}
    resolve_splits(path, counts, RATIOS, seed=0)

    other_ratios = {"train": 0.5, "val": 0.25, "test": 0.25}
    with pytest.raises(SplitMapError, match="ratios"):
        resolve_splits(path, {**counts, "g3": 3}, other_ratios, seed=0)
