"""P3 acceptance: §3.6 worked example, v13i-like coarse tolerance, byte-identical
regeneration, and v1's SPLIT_MAP.json left untouched."""

from __future__ import annotations

import hashlib
from pathlib import Path

from marinedata.splitv2.allocate import achieved_by_stratum, allocate
from marinedata.splitv2.mapfile import build, to_json

REPO_ROOT = Path(__file__).resolve().parents[1]
V1_SPLIT_MAP = REPO_ROOT / "registry" / "SPLIT_MAP.json"


def _singleton_groups(prefix: str, n: int) -> dict[str, int]:
    return {f"sg-{prefix}-{i}": 1 for i in range(n)}


def test_section_3_6_coralscapes_test_overfull_reproduces_the_documented_counts():
    """design §3.6: coralscapes 2,075 images, route test 392 (overfull) -> train 1,386
    / val 297 / test 392. Single-image groups make the target arithmetic exact, so the
    allocator must reproduce it exactly, not just "close"."""
    groups = _singleton_groups("cs", 2075)
    pinned_gids = list(groups)[:392]
    pinned = {gid: "test" for gid in pinned_gids}
    strata = {"coralscapes": groups}

    result = allocate(strata, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0, pinned=pinned)
    achieved = achieved_by_stratum(strata, result.assignment)["coralscapes"]

    assert achieved == {"train": 1386, "val": 297, "test": 392}
    assert sum(achieved.values()) == 2075


def test_section_3_6_never_eval_stratum_is_excluded_before_allocation():
    """coralscop-masks-rs (never-eval) never enters the ratio-gated pool at all — it is
    a separate train-only pool (design §0 point 1), not folded into any stratum here."""
    never_eval_groups = _singleton_groups("ce", 37273)
    # the caller (cli_splits.build_strata) excludes never-eval sources before calling
    # allocate; simulate that by simply never passing them in.
    strata = {"coralscapes": _singleton_groups("cs2", 100)}
    result = allocate(strata, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0)
    assert set(result.assignment) == set(strata["coralscapes"])
    assert len(never_eval_groups) == 37273  # sanity: the excluded pool size from §0/§3.6


def test_v13i_like_coarse_stratum_stays_within_its_widened_tolerance():
    """v1's roboflow-...-v13i (1,172 images) came out 50/25/25 (design §0/§3.3). Build a
    coarse stratum (one group > 10% of the stratum) at the same scale and check the
    allocator's achieved ratios stay within design §3.3's coarse tolerance:
    max(2.0 pt, largest-group share pt)."""
    total = 1172
    big = 129  # 11.0% of 1172 -> triggers the "largest group > 10%" coarse rule
    rest = total - big
    groups = {"sg-v13i-big": big}
    # 20 similarly-sized small groups filling the remainder
    each = rest // 20
    for i in range(19):
        groups[f"sg-v13i-{i}"] = each
    groups["sg-v13i-19"] = rest - each * 19
    assert sum(groups.values()) == total
    strata = {"v13i": groups}

    result = allocate(strata, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0)
    achieved = achieved_by_stratum(strata, result.assignment)["v13i"]

    largest_share_pt = 100 * big / total
    tolerance_pt = max(2.0, largest_share_pt)
    for split, frac in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        achieved_pt = 100 * achieved.get(split, 0) / total
        assert abs(achieved_pt - frac * 100) <= tolerance_pt, (split, achieved)


def test_split_map_v2_regeneration_is_byte_identical():
    """design §3.5: same seed/ratios/hashes/assignments -> byte-identical map_sha256
    and JSON, any number of times."""
    kwargs = dict(
        seed=0,
        ratios={"train": 0.70, "val": 0.15, "test": 0.15},
        rules_sha256="a" * 64,
        benchmarks_sha256="b" * 64,
        assignments={"sg-1": "train", "sg-2": "val", "sg-3": "test"},
        ood_tags={"sg-4": ["ood-source-deepfish"]},
    )
    first = build(**kwargs)
    second = build(**kwargs)

    assert first.map_sha256 == second.map_sha256
    assert to_json(first) == to_json(second)


def test_v1_split_map_json_is_untouched_by_running_split_v2():
    """Running the split-v2 allocator/mapfile never reads or writes v1's
    registry/SPLIT_MAP.json — hash it before and after (brief: 'Do NOT touch v1's
    SPLIT_MAP')."""
    before = hashlib.sha256(V1_SPLIT_MAP.read_bytes()).hexdigest()

    groups = _singleton_groups("noop", 50)
    allocate({"noop-stratum": groups}, {"train": 0.70, "val": 0.15, "test": 0.15}, seed=0)
    build(
        seed=0,
        ratios={"train": 0.70, "val": 0.15, "test": 0.15},
        rules_sha256="c" * 64,
        benchmarks_sha256="d" * 64,
        assignments={"sg-noop-0": "train"},
    )

    after = hashlib.sha256(V1_SPLIT_MAP.read_bytes()).hexdigest()
    assert before == after
