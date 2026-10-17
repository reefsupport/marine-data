"""Split-map run 8: a DCT pHash-64 pair is confirmed before it merges split groups, and one image of
every pHash-64 pair the assigned map still ships across splits is excluded (``phash_excluded``)."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from test_neardup import _config, _encode, _picture, _registry, _source, _stage

from marinedata.neardup import PHASH_UNION_RULE, NearDupChainError, UnionConfirm
from marinedata.neardup_phash import phash_confirm_shas, phash_exclusions, select_phash_links
from marinedata.release import generate_split_map
from marinedata.splitmap import (
    SCHEMA_VERSION,
    SplitMap,
    SplitMapError,
    load_split_map,
    save_split_map,
)

TRAIN, VAL, TEST = frozenset({"train"}), frozenset({"val"}), frozenset({"test"})


def _features(bits: int, entropy: float = 6.0) -> tuple[bytes, float]:
    """A synthetic pHash-256 with ``bits`` low bits set: a distance is the popcount of the XOR."""
    return ((1 << bits) - 1).to_bytes(32, "big"), entropy


def test_selection_unions_confirmed_or_single_image_pairs_never_two_prior_images() -> None:
    features = {"a": _features(0), "b": _features(10), "c": _features(40), "d": _features(0, 1.0)}
    pairs = [("a", "b", 4), ("a", "c", 6), ("a", "d", 2), ("p", "q", 0), ("s1", "s2", 8)]
    single, prior = frozenset({"s1", "s2", "p", "q"}), frozenset({"p", "q"})
    assert phash_confirm_shas(pairs, single=single, prior=prior) == {"a", "b", "c", "d"}
    got = select_phash_links(
        pairs, features=features, rule=UnionConfirm(), single=single, prior=prior
    )
    assert got.links == [("a", "b"), ("s1", "s2")]
    counts = (got.confirmed, got.single, got.prior_pairs, got.rejected_phash, got.rejected_entropy)
    assert counts == (1, 1, 1, 1, 1)
    assert got.pairs == pairs  # every pair stays for the exclusion after assignment


def test_exclusion_drops_the_busiest_unprotected_image_and_freezes_prior_pairs() -> None:
    splits = {"hub": TRAIN, "x": TEST, "y": VAL, "z": TRAIN, "p": TEST, "q": TRAIN, "w": VAL}
    pairs = [
        ("hub", "x", 6),
        ("hub", "y", 8),
        ("hub", "z", 2),  # one split: not a leak
        ("p", "q", 0),  # two prior images: frozen, never dropped
        ("p", "w", 4),  # the prior image keeps its place, w goes
        ("gone", "x", 0),  # gone ships nowhere (withheld group, band-excluded image)
    ]
    excluded, frozen = phash_exclusions(pairs, splits, protected=frozenset({"p", "q"}))
    assert excluded == {"hub": (6, "x", "test"), "w": (4, "p", "test")}
    assert frozen == 1
    kept = {sha for sha in splits if sha not in excluded}
    left = [(a, b) for a, b, _ in pairs if {a, b} <= kept and len(splits[a] | splits[b]) > 1]
    assert left == [("p", "q")]  # one pass: only the frozen pair still spans splits


def test_exclusion_breaks_ties_by_sha_and_spares_protected_images() -> None:
    splits = {"m": TRAIN, "n": TEST}
    assert phash_exclusions([("m", "n", 8)], splits) == ({"m": (8, "n", "test")}, 0)
    spared = phash_exclusions([("m", "n", 8)], splits, protected=frozenset({"m"}))
    assert spared == ({"n": (8, "m", "train")}, 0)
    both = {"m": TRAIN | TEST, "n": TRAIN}  # an image published in two splits
    got = phash_exclusions([("m", "n", 2)], both, protected=frozenset({"m"}))
    assert got == ({"n": (2, "m", "test")}, 0)


def test_phash_excluded_round_trips_and_needs_schema_2(tmp_path: Path) -> None:
    split_map = SplitMap(
        by="group",
        seed=0,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        assignments={"s/g": "train"},
        schema_version=SCHEMA_VERSION,
        phash_excluded={"ab12": (6, "cd34", "test")},
    )
    save_split_map(tmp_path / "m.json", split_map)
    got = load_split_map(tmp_path / "m.json")
    assert got is not None
    assert (got.phash_excluded, got.band_excluded) == ({"ab12": (6, "cd34", "test")}, {})
    with pytest.raises(SplitMapError, match="phash_excluded needs schema_version"):
        save_split_map(tmp_path / "v1.json", replace(split_map, schema_version=1))
    save_split_map(tmp_path / "plain.json", replace(split_map, phash_excluded={}))
    assert "phash_excluded" not in (tmp_path / "plain.json").read_text()
    plain = load_split_map(tmp_path / "plain.json")
    assert plain is not None and plain.phash_excluded == {}


def _site_and_twin(tmp_path: Path):  # type: ignore[no-untyped-def]
    """The twin fixture of test_neardup, but source a's twin sits in a 5-image site group."""
    twin = _picture(100)
    rows_a = [("t", "a/site", _encode(twin, "PNG"))]
    rows_a += [(f"s{i}", "a/site", _encode(_picture(i), "PNG")) for i in range(4)]
    rows_a += [(f"u{i}", f"a/g{i}", _encode(_picture(i), "PNG")) for i in range(4, 8)]
    rows_b = [("t", "b/twin", _encode(twin, "JPEG"))]  # same photo, different bytes
    rows_b += [(f"u{i}", f"b/g{i}", _encode(_picture(50 + i), "PNG")) for i in range(8)]
    roots = {"src-a": _stage(tmp_path / "a", rows_a), "src-b": _stage(tmp_path / "b", rows_b)}
    return _registry({"src-a": _source("src-a"), "src-b": _source("src-b")}), roots


def test_guard_still_refuses_a_confirmed_site_merge_and_an_unconfirmed_pair_drops_one_image(
    tmp_path: Path,
) -> None:
    registry, roots = _site_and_twin(tmp_path)
    kw = {"registry": registry, "roots": roots, "now": "2026-10-17T00:00:00Z", "min_groups": 1}
    loose = UnionConfirm(phash256_max=256, min_entropy_bits=0.0)  # confirms everything
    with pytest.raises(NearDupChainError):  # site + twin = 6 of 18 images > 20 %
        generate_split_map(
            out=tmp_path / "loose.json",
            near_dup=_config(tmp_path, union_confirm=loose, chain_fraction=0.2, phash_max=8),
            **kw,
        )
    flat = UnionConfirm(min_entropy_bits=99.0)  # rejects every pair: no rule-B link either
    group_of = {
        hashlib.sha256((roots[s] / "images/p/t.img").read_bytes()).hexdigest(): g
        for s, g in (("src-a", "a/site"), ("src-b", "b/twin"))
    }
    spans = 0
    for seed in range(12):
        out = tmp_path / f"m{seed}.json"
        stats = generate_split_map(
            out=out,
            seed=seed,
            near_dup=_config(tmp_path, union_confirm=flat, chain_fraction=0.2, phash_max=8),
            **kw,
        )
        got = (stats.near_dup_phash_pairs, stats.near_dup_phash_unioned, stats.near_dup_unions)
        assert got == (1, 0, 0)  # the site's pair is not unioned: no chain, the guard passes
        split_map = load_split_map(out)
        assert split_map is not None
        assert split_map.near_dup["phash_union"] == PHASH_UNION_RULE
        split_of = {sha: split_map.assignments[g] for sha, g in group_of.items()}
        if len(set(split_of.values())) == 1:
            assert (split_map.phash_excluded, stats.phash_excluded_images) == ({}, 0)
            continue
        spans += 1
        low, high = sorted(split_of)
        ((sha, hit),) = split_map.phash_excluded.items()
        assert (sha, hit[1], hit[2]) == (low, high, split_of[high]) and 0 <= hit[0] <= 8
        assert (split_map.schema_version, stats.phash_excluded_images) == (SCHEMA_VERSION, 1)
    assert spans  # some seed puts the site and the twin in different splits
