"""Group-size-aware allocator with a repair pass (design §3.3).

Fixes v1's 85.7/7.2/7.1 drift (design §0): that drift was a reporting artefact of
folding the never-eval pool into the ratio gate, not a packing bug — but v1's packer
*did* let a big group overshoot val/test in a coarse stratum (§0 point 2). This module
is the fix for that: a free group only lands in a split if doing so keeps that split at
or under ``overshoot_cap`` (default 1.10) of every stratum target it belongs to; a
repair pass then swaps groups between ``train`` and a short split to close the
remaining gap, smallest groups first, deterministically, at most ``repair_max_swaps``
total.

A stratum here is normally a source; ``strata`` follows :mod:`marinedata.strata`'s
shape (``stratum -> group_id -> image count``) so a group shared by two strata is
still placed once. ``pinned`` groups (upstream routing, v1 continuity) are never
re-offered — they only count toward the quota, exactly as :mod:`marinedata.strata`
and :mod:`marinedata.splitmap` already do for v1.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from ..strata import DEFAULT_MIN_GROUPS, small_strata

SplitName = str
TRAIN: SplitName = "train"


@dataclass(frozen=True)
class StratumPlan:
    kind: str  # "normal" | "test-heavy" | "train-only"
    total: int
    targets: dict[SplitName, float]


@dataclass(frozen=True)
class AllocationResult:
    assignment: dict[str, SplitName]
    plans: dict[str, StratumPlan]
    filled: dict[str, dict[SplitName, int]]
    swaps: int


def _group_hash(seed: int, gid: str) -> str:
    return hashlib.sha256(f"{seed}:{gid}".encode()).hexdigest()


def _build_membership(
    strata: Mapping[str, Mapping[str, int]],
) -> dict[str, dict[str, int]]:
    membership: dict[str, dict[str, int]] = {}
    for stratum, groups in strata.items():
        for gid, count in groups.items():
            membership.setdefault(gid, {})[stratum] = count
    return membership


def allocate(
    strata: Mapping[str, Mapping[str, int]],
    ratios: Mapping[SplitName, float],
    *,
    seed: int = 0,
    pinned: Mapping[str, SplitName] | None = None,
    min_groups: int = DEFAULT_MIN_GROUPS,
    test_overfull_threshold: float | None = None,
    overshoot_cap: float = 1.10,
    repair_max_swaps: int = 200,
) -> AllocationResult:
    """Assign every group in ``strata``. Pinned groups keep their split; the rest are
    placed by the largest-normalised-deficit rule under the overshoot cap, then a
    repair pass narrows any stratum still outside tolerance."""
    if TRAIN not in ratios:
        raise ValueError(f"allocation needs a {TRAIN!r} split, got {sorted(ratios)}")
    pinned = dict(pinned or {})
    test_overfull_threshold = (
        ratios.get("test", 0.0) if test_overfull_threshold is None else test_overfull_threshold
    )
    small = set(small_strata(strata, min_groups))
    membership = _build_membership(strata)
    totals = {name: sum(groups.values()) for name, groups in strata.items()}

    filled: dict[str, dict[SplitName, int]] = {name: dict.fromkeys(ratios, 0) for name in strata}
    for gid, stratum_counts in membership.items():
        split = pinned.get(gid)
        if split is None:
            continue
        for stratum, count in stratum_counts.items():
            filled[stratum][split] = filled[stratum].get(split, 0) + count

    test_overfull = {
        name
        for name, total in totals.items()
        if name not in small
        and total > 0
        and filled[name].get("test", 0) / total >= test_overfull_threshold
    }

    plans: dict[str, StratumPlan] = {}
    for name, total in totals.items():
        if name in small:
            plans[name] = StratumPlan("train-only", total, dict.fromkeys(ratios, 0.0))
        elif name in test_overfull:
            non_test = total - filled[name].get("test", 0)
            train_frac = ratios[TRAIN] / (ratios[TRAIN] + ratios.get("val", 0.0))
            val_frac = 1.0 - train_frac
            targets = {TRAIN: train_frac * non_test, "val": val_frac * non_test}
            targets["test"] = float(filled[name].get("test", 0))
            plans[name] = StratumPlan("test-heavy", total, targets)
        else:
            plans[name] = StratumPlan("normal", total, {s: f * total for s, f in ratios.items()})

    assignment: dict[str, SplitName] = dict(pinned)
    _place_free_groups(membership, plans, filled, assignment, pinned, ratios, seed, overshoot_cap)
    _test_heavy_free_pass(strata, plans, filled, assignment, pinned, seed, overshoot_cap)
    swaps = _repair(strata, plans, filled, assignment, pinned, seed, repair_max_swaps)
    return AllocationResult(assignment=assignment, plans=plans, filled=filled, swaps=swaps)


def _place_free_groups(
    membership: Mapping[str, Mapping[str, int]],
    plans: Mapping[str, StratumPlan],
    filled: dict[str, dict[SplitName, int]],
    assignment: dict[str, SplitName],
    pinned: Mapping[str, SplitName],
    ratios: Mapping[SplitName, float],
    seed: int,
    overshoot_cap: float,
) -> None:
    def size(gid: str) -> int:
        return max(membership[gid].values())

    def test_heavy_only(gid: str) -> bool:
        kinds = {plans[s].kind for s in membership[gid]}
        return kinds == {"test-heavy"}

    free = [gid for gid in membership if gid not in pinned and not test_heavy_only(gid)]
    ordered = sorted(free, key=lambda g: (-size(g), _group_hash(seed, g)))

    for gid in ordered:
        active = {
            stratum: count
            for stratum, count in membership[gid].items()
            if plans[stratum].kind == "normal"
        }
        if not active:
            split = TRAIN
        else:
            best_split, best_score = None, float("-inf")
            for split_name in ratios:
                fits = all(
                    filled[s][split_name] + count <= plans[s].targets[split_name] * overshoot_cap
                    for s, count in active.items()
                )
                if not fits:
                    continue
                score = sum(
                    (plans[s].targets[split_name] - filled[s][split_name])
                    / plans[s].targets[split_name]
                    for s in active
                    if plans[s].targets[split_name] > 0
                )
                if score > best_score:
                    best_score, best_split = score, split_name
            split = best_split if best_split is not None else TRAIN
        assignment[gid] = split
        for stratum, count in membership[gid].items():
            filled[stratum][split] = filled[stratum].get(split, 0) + count


def _test_heavy_free_pass(
    strata: Mapping[str, Mapping[str, int]],
    plans: Mapping[str, StratumPlan],
    filled: dict[str, dict[SplitName, int]],
    assignment: dict[str, SplitName],
    pinned: Mapping[str, SplitName],
    seed: int,
    overshoot_cap: float,
) -> None:
    for name, plan in plans.items():
        if plan.kind != "test-heavy":
            continue
        free = [gid for gid in strata[name] if gid not in pinned and gid not in assignment]
        ordered = sorted(free, key=lambda g: (-strata[name][g], _group_hash(seed, g)))
        for gid in ordered:
            count = strata[name][gid]
            fits = [
                s
                for s in ("train", "val")
                if filled[name][s] + count <= plan.targets[s] * overshoot_cap
            ]
            deficits = {
                s: (plan.targets[s] - filled[name][s]) / max(plan.targets[s], 1) for s in fits
            }
            split = max(deficits, key=lambda s: deficits[s]) if fits else TRAIN
            assignment[gid] = split
            filled[name][split] = filled[name].get(split, 0) + count


def _l1_error(plan: StratumPlan, filled: Mapping[SplitName, int]) -> float:
    return sum(abs(filled.get(s, 0) - t) for s, t in plan.targets.items())


def _repair(
    strata: Mapping[str, Mapping[str, int]],
    plans: Mapping[str, StratumPlan],
    filled: dict[str, dict[SplitName, int]],
    assignment: dict[str, SplitName],
    pinned: Mapping[str, SplitName],
    seed: int,
    max_swaps: int,
) -> int:
    """Pairwise train<->short-split swaps, smallest groups first, deterministic order,
    accepted only when they strictly reduce a stratum's L1 error (design §3.3.4)."""
    swaps = 0
    for name in sorted(plans):
        plan = plans[name]
        if plan.kind != "normal" or max_swaps - swaps <= 0:
            continue
        own = dict(strata[name])
        for _ in range(max_swaps - swaps):
            deficits = {s: plan.targets[s] - filled[name].get(s, 0) for s in plan.targets}
            short = max(deficits, key=lambda s: deficits[s])
            if deficits[short] <= 0:
                break
            train_candidates = sorted(
                (gid for gid in own if assignment.get(gid) == TRAIN and gid not in pinned),
                key=lambda g: (own[g], _group_hash(seed, g)),
            )
            short_candidates = sorted(
                (gid for gid in own if assignment.get(gid) == short and gid not in pinned),
                key=lambda g: (own[g], _group_hash(seed, g)),
            )
            best_pair = None
            best_error = _l1_error(plan, filled[name])
            for g_train in train_candidates[:20]:
                for g_short in short_candidates[:20]:
                    trial = dict(filled[name])
                    trial[TRAIN] = trial[TRAIN] - own[g_train] + own[g_short]
                    trial[short] = trial[short] - own[g_short] + own[g_train]
                    error = _l1_error(plan, trial)
                    if error < best_error:
                        best_error, best_pair = error, (g_train, g_short)
            if best_pair is None:
                break
            g_train, g_short = best_pair
            assignment[g_train], assignment[g_short] = short, TRAIN
            filled[name][TRAIN] = filled[name][TRAIN] - own[g_train] + own[g_short]
            filled[name][short] = filled[name][short] - own[g_short] + own[g_train]
            swaps += 1
            if swaps >= max_swaps:
                return swaps
    return swaps


def achieved_by_stratum(
    strata: Mapping[str, Mapping[str, int]], assignment: Mapping[str, SplitName]
) -> dict[str, dict[SplitName, int]]:
    out: dict[str, dict[SplitName, int]] = {}
    for name, groups in sorted(strata.items()):
        per: dict[SplitName, int] = {}
        for gid, count in groups.items():
            per[assignment[gid]] = per.get(assignment[gid], 0) + count
        out[name] = per
    return out
