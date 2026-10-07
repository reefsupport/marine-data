"""Duplicate-content merge + never-eval (WS-D S15c).

The staged coralscop-masks-rs tree has 684 groups whose images share a sha256 with a
*different* ``split_group`` — `rows_to_counts` used to raise "maps to two different
split_group values" on this, crashing any release that includes it. These tests pin the
fix: such groups MERGE into one connected component and are assigned as a unit, a
conflict with a group a *prior* map already split apart still raises, and a
``never-eval``-tagged source can only ever land in ``train``.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from marinedata.splitmap import (
    SplitMap,
    SplitMapError,
    load_split_map,
    resolve_splits,
    rows_to_counts,
    save_split_map,
)

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def test_two_stems_same_digest_merge_into_one_component() -> None:
    """Two rows for the same image under two different split_group values (the coralscop
    shape: same content, different per-stem group) merge instead of raising."""
    rows = [("sha1", "g/a", None), ("sha1", "g/b", None), ("sha2", "g/c", None)]
    counts, _, merge_info = rows_to_counts(rows)
    # sha1 is one image: it counts once toward the merged group, same as the existing
    # cross-source-dedup rule this used to raise on. sha2 is a second, unrelated image.
    assert counts == {"g/a": 1, "g/c": 1}
    assert merge_info.canonical == {"g/a": "g/a", "g/b": "g/a", "g/c": "g/c"}
    assert merge_info.merged_components == 1


def test_cross_source_duplicate_merges(tmp_path: Path) -> None:
    """The same image staged under two different *sources* (different stratum, sharing
    a digest) still merges into one component and gets one split."""
    rows = [
        ("sha1", "src-a/g1", "src-a"),
        ("sha1", "src-b/g1", "src-b"),
        ("sha2", "src-a/g2", "src-a"),
        ("sha3", "src-b/g2", "src-b"),
    ]
    counts, strata, merge_info = rows_to_counts(rows, stratified=True)
    canon = merge_info.canonical["src-a/g1"]
    assert merge_info.canonical["src-b/g1"] == canon
    assert canon in counts

    out = tmp_path / "SPLIT_MAP.json"
    resolve_splits(
        out,
        counts,
        {"train": 1.0, "val": 0.0, "test": 0.0},
        seed=0,
        by="group",
        strata=strata,
        stratify="source",
        min_groups=1,
        merge_canonical=merge_info.canonical,
    )
    split_map = load_split_map(out)
    assert split_map is not None
    # Both original group ids resolve to the same split, not just the canonical one.
    assert split_map.assignments["src-a/g1"] == split_map.assignments["src-b/g1"]


def test_merge_conflicting_with_prior_map_raises(tmp_path: Path) -> None:
    """A merge that would join a component whose members a PRIOR map already split
    apart raises, naming both groups — never silently moving a released assignment."""
    out = tmp_path / "SPLIT_MAP.json"
    save_split_map(
        out,
        SplitMap(by="group", seed=0, ratios=RATIOS, assignments={"g/a": "train", "g/b": "test"}),
    )
    rows = [("sha1", "g/a", None), ("sha1", "g/b", None)]
    counts, _, merge_info = rows_to_counts(rows)

    with pytest.raises(SplitMapError, match=r"g/a.*g/b|g/b.*g/a"):
        resolve_splits(
            out, counts, RATIOS, seed=0, by="group", merge_canonical=merge_info.canonical
        )


def test_merge_agreeing_with_prior_map_inherits_split(tmp_path: Path) -> None:
    """If a prior map already agrees across every persisted member of a component, the
    whole (now-merged) component inherits that split rather than re-allocating."""
    out = tmp_path / "SPLIT_MAP.json"
    save_split_map(out, SplitMap(by="group", seed=0, ratios=RATIOS, assignments={"g/a": "test"}))
    rows = [("sha1", "g/a", None), ("sha1", "g/b", None), ("sha2", "g/c", None)]
    counts, _, merge_info = rows_to_counts(rows)

    resolve_splits(out, counts, RATIOS, seed=0, by="group", merge_canonical=merge_info.canonical)
    split_map = load_split_map(out)
    assert split_map is not None
    assert split_map.assignments["g/a"] == "test"
    assert split_map.assignments["g/b"] == "test"


def test_never_eval_only_component_forced_to_train(tmp_path: Path) -> None:
    """A component whose only contributing stratum is never-eval-tagged goes straight
    to train — no hashing lottery — even under ratios that would send it elsewhere."""
    strata = {"never-eval-src": {"g/a": 100}}
    counts = {"g/a": 100}
    out = tmp_path / "SPLIT_MAP.json"
    resolve_splits(
        out,
        counts,
        {"train": 0.0, "val": 0.5, "test": 0.5},
        seed=0,
        by="group",
        strata=strata,
        stratify="source",
        min_groups=1,
        forced={"g/a": "train"},
    )
    split_map = load_split_map(out)
    assert split_map is not None
    assert split_map.assignments["g/a"] == "train"


def test_mixed_component_in_test_excludes_never_eval_rows() -> None:
    """A mixed component (never-eval + real rows) sent to a non-train split keeps the
    real rows there; the never-eval rows are excluded and counted, never moved to train.

    Exercises the row-level filter `build_release` applies (WS-D S15c), inlined here
    since standing up a full registry/staged-tree fixture is out of this unit's scope
    (see test_release_never_eval.py for the end-to-end version).
    """
    never_eval_sources = {"never-eval-src"}
    rows = [
        ("real-src", "test", "sha-real-1"),
        ("real-src", "test", "sha-real-2"),
        ("never-eval-src", "test", "sha-pseudo-1"),
        ("never-eval-src", "train", "sha-pseudo-2"),
    ]
    kept: list[tuple[str, str]] = []
    excluded = 0
    for source_id, split_name, sha256 in rows:
        if split_name != "train" and source_id in never_eval_sources:
            excluded += 1
            continue
        kept.append((sha256, split_name))
    assert excluded == 1
    assert ("sha-real-1", "test") in kept
    assert ("sha-real-2", "test") in kept
    assert ("sha-pseudo-2", "train") in kept
    assert not any(sha == "sha-pseudo-1" for sha, _ in kept)


def test_merge_is_deterministic_under_row_shuffle() -> None:
    """Shuffling row order must not change which id a component canonicalises to, nor
    the merged component count — the DSU + lexicographic-min rule is order-independent."""
    base_rows = [
        ("sha1", "g/c", "src-a"),
        ("sha1", "g/a", "src-b"),
        ("sha2", "g/a", "src-b"),
        ("sha2", "g/b", "src-c"),
        ("sha3", "g/z", "src-a"),
    ]
    first_counts, first_strata, first_merge = rows_to_counts(list(base_rows), stratified=True)

    shuffled = list(base_rows)
    random.Random(42).shuffle(shuffled)
    second_counts, second_strata, second_merge = rows_to_counts(shuffled, stratified=True)

    assert first_counts == second_counts
    assert first_strata == second_strata
    assert first_merge.canonical == second_merge.canonical
    assert first_merge.merged_components == second_merge.merged_components == 1
