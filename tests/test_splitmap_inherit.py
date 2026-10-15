"""D1/D2: an inheriting split map embeds the prior map, never reassigns it, and routes OOD groups
to ``test_ood`` outside the 70/15/15 quota."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marinedata.splitmap import SplitMap, SplitMapError, load_split_map, save_split_map
from marinedata.splitmap_inherit import band_constraints, load_inherited, resolve_inherited_splits
from marinedata.strata import TEST_OOD

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
NOW = "2026-10-15T00:00:00+00:00"


def _prior(tmp_path: Path, assignments: dict[str, str]):  # type: ignore[no-untyped-def]
    path = tmp_path / "v1.json"
    save_split_map(
        path,
        SplitMap(
            by="group",
            seed=0,
            ratios=dict(RATIOS),
            assignments=assignments,
            generated_at=NOW,
            release="v1.0.0",
            stratify="source",
        ),
    )
    return load_inherited(path)


def _resolve(  # type: ignore[no-untyped-def]
    tmp_path: Path, prior_assignments, counts, *, merge=None, strata=None, **kwargs
):
    prior, digest = _prior(tmp_path, prior_assignments)
    out = tmp_path / "v11.json"
    result = resolve_inherited_splits(
        out,
        counts,
        RATIOS,
        prior=prior,
        prior_sha256=digest,
        strata=strata or {"s": counts},
        merge_canonical=merge or {},
        now=NOW,
        release="v1.1.0",
        seed=3,
        **kwargs,
    )
    return result, load_split_map(out), digest


def test_prior_splits_survive_a_near_dup_component_that_spans_them(tmp_path: Path) -> None:
    # component "a" = {a, b, n1}: the prior map put a in train and b in val
    counts = {"a": 3, "c": 2, "n2": 1}
    merge = {"a": "a", "b": "a", "n1": "a", "c": "c", "n2": "n2"}
    result, got, digest = _resolve(
        tmp_path, {"a": "train", "b": "val", "c": "test"}, counts, merge=merge
    )
    assert got is not None
    assert (got.assignments["a"], got.assignments["b"], got.assignments["c"]) == (
        "train",
        "val",
        "test",
    )
    assert "n1" not in got.assignments  # cannot honour both prior splits: withheld, not guessed
    assert "train and val" in got.withheld["n1"]
    assert result.conflict_components == 1 and result.inherited_groups == 3
    assert got.inherits == {"release": "v1.0.0", "seed": 0, "groups": 3, "sha256": digest}


def test_a_new_group_in_a_prior_component_takes_the_prior_split(tmp_path: Path) -> None:
    counts = {"a": 2, "c": 1}
    result, got, _ = _resolve(
        tmp_path,
        {"a": "val", "c": "train"},
        counts,
        merge={"a": "a", "n1": "a", "c": "c"},
    )
    assert got is not None and got.assignments["n1"] == "val"
    assert result.adopted_groups == 1 and not got.withheld


def test_every_member_of_a_new_near_dup_group_lands_in_one_split(tmp_path: Path) -> None:
    counts = {f"k{i:02d}a": 3 for i in range(40)}
    merge = {f"k{i:02d}{s}": f"k{i:02d}a" for i in range(40) for s in "abc"}
    _result, got, _ = _resolve(tmp_path, {"p": "train"}, {**counts, "p": 1}, merge=merge)
    assert got is not None
    for i in range(40):
        assert len({got.assignments[f"k{i:02d}{s}"] for s in "abc"}) == 1
    assert {"val", "test"} <= set(got.assignments.values())


def test_ood_components_go_to_test_ood_and_to_no_other_split(tmp_path: Path) -> None:
    counts = {f"g{i:02d}": 1 for i in range(40)}
    tags = ["ood-source-deepfish"]
    result, got, _ = _resolve(
        tmp_path,
        {"g11": "test"},
        counts,
        ood={"g03": tags, "g07": tags, "g09": tags, "g11": tags},
        forced={"g09": "train"},  # never-eval / band outrank an OOD match; so does a prior split
        ood_header={"rules_sha256": "x"},
    )
    assert got is not None
    routed = {g for g, s in got.assignments.items() if s == TEST_OOD}
    assert routed == {"g03", "g07"} == set(got.ood_rules)
    assert got.assignments["g09"] == "train" and got.assignments["g11"] == "test"
    assert result.ood_components == 2 and result.ood_blocked == 2
    assert got.ood == {"rules_sha256": "x"}
    assert all(s in RATIOS for g, s in got.assignments.items() if g not in routed)


def test_proportions_of_new_groups_are_within_five_points(tmp_path: Path) -> None:
    strata = {f"s{k}": {f"s{k}/g{i:02d}": 1 + (i % 3) for i in range(60)} for k in range(3)}
    strata["s0"]["old"] = 1
    counts = {g: n for groups in strata.values() for g, n in groups.items()}
    _, got, _ = _resolve(tmp_path, {"old": "train"}, counts, strata=strata)
    assert got is not None
    total = sum(counts.values())
    for split, target in RATIOS.items():
        share = sum(n for g, n in counts.items() if got.assignments[g] == split) / total
        assert abs(share - target) <= 0.05, (split, share)


def test_a_prior_test_ood_group_is_kept_and_takes_no_quota(tmp_path: Path) -> None:
    counts = {"o1": 2, **{f"n{i:02d}": 1 for i in range(30)}}
    _, got, _ = _resolve(tmp_path, {"o1": TEST_OOD, "t": "train"}, {**counts, "t": 1})
    assert got is not None and got.assignments["o1"] == TEST_OOD


def test_ratios_must_match_and_an_existing_map_is_never_overwritten(tmp_path: Path) -> None:
    prior, digest = _prior(tmp_path, {"a": "train"})
    kwargs = dict(prior=prior, prior_sha256=digest, strata={"s": {"a": 1}}, merge_canonical={})
    out = tmp_path / "v11.json"
    with pytest.raises(SplitMapError, match="ratios"):
        resolve_inherited_splits(out, {"a": 1}, {"train": 0.8, "val": 0.1, "test": 0.1}, **kwargs)
    resolve_inherited_splits(out, {"a": 1}, RATIOS, **kwargs)
    with pytest.raises(SplitMapError, match="already exists"):
        resolve_inherited_splits(out, {"a": 1}, RATIOS, **kwargs)


def test_band_neighbours_of_prior_images_are_never_eval_never_train_or_withheld() -> None:
    v1 = {"t1": "train", "v1": "val", "e1": "test"}
    component_of = {"n1": "A", "n2": "B", "n3": "C"}
    pairs = [("n1", "t1", 6), ("n2", "e1", 7), ("n2", "v1", 5), ("n3", "t1", 8), ("n3", "v1", 6)]
    forced, withheld = band_constraints(pairs, v1_split=v1, component_of=component_of)
    assert forced == {"A": "train", "B": "val"}  # B: the nearest evaluation image decides
    assert list(withheld) == ["C"]


def test_new_header_fields_round_trip_and_a_v1_map_keeps_its_exact_keys(tmp_path: Path) -> None:
    plain = SplitMap(by="group", seed=0, ratios=dict(RATIOS), assignments={"a": "train"})
    save_split_map(tmp_path / "plain.json", plain)
    assert not {"inherits", "ood", "ood_rules", "withheld"} & set(
        json.loads((tmp_path / "plain.json").read_text())
    )
    full = SplitMap(
        by="group",
        seed=1,
        ratios=dict(RATIOS),
        assignments={"a": TEST_OOD},
        inherits={"sha256": "ab"},
        ood={"holdouts": [{"name": "h"}]},
        ood_rules={"a": ["h"]},
        withheld={"b": "why"},
    )
    save_split_map(tmp_path / "full.json", full)
    assert load_split_map(tmp_path / "full.json") == full
