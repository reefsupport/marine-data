"""Stratified group allocation — every stratum gets the requested ratios, not just the union.

:func:`marinedata.scan.assign_splits` packs one pool of groups largest-first against one
set of targets. Run over the union of several sources, that pool is dominated by
whichever source has the biggest groups: they fill ``train`` while every split is still
empty, and the small groups of the other sources fill ``val``/``test``. The union totals
then look right while each source is skewed — in the S7k dry run all four
``rs-colombia/*`` sites (1,250 images) landed in ``train`` and reefolution came out
27/36/36 by image.

:func:`assign_splits_stratified` keeps one allocator and one shared assignment, but gives
every stratum (normally a source) its own targets. Each group is placed, largest share of
its stratum first, into the split with the greatest summed deficit across every stratum
it belongs to, each deficit normalised by that stratum's size. Consequences:

* A group in exactly one stratum is placed exactly as ``assign_splits`` would place it
  inside that stratum alone, so disjoint strata reproduce per-stratum allocation.
* A group shared by several strata (``seaview/<survey>`` across the SEAVIEW-family
  sources, ``rs-colombia/<site>`` across benthic-own and the bleaching set) is still
  placed once — the leakage invariant is untouched — and lands where the strata it
  belongs to are jointly furthest below target.
* A stratum with at least ``min_groups`` groups never ends with an empty split: a split left
  without a group takes one from ``train`` (WP-R6), never an upstream-test group (those are
  pinned).
* A stratum with fewer than ``min_groups`` groups cannot hold a group out per split. It
  takes no part in the deficit sum, and a group that belongs *only* to such strata goes
  to ``train``: a held-out split made of one group would measure that group, not the
  source, while looking like a metric.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from .scan import SplitName

TRAIN: SplitName = "train"
TEST: SplitName = "test"
DEFAULT_MIN_GROUPS = 3


def small_strata(strata: Mapping[str, Mapping[str, int]], min_groups: int) -> list[str]:
    """Strata with too few groups to be balanced — they are train-only by rule."""
    return sorted(name for name, groups in strata.items() if len(groups) < min_groups)


def assign_splits_stratified(
    strata: Mapping[str, Mapping[str, int]],
    ratios: Mapping[SplitName, float],
    *,
    seed: int = 0,
    pinned: Mapping[str, SplitName] | None = None,
    min_groups: int = DEFAULT_MIN_GROUPS,
) -> dict[str, SplitName]:
    """Assign every group in ``strata`` that is not ``pinned``; return only those.

    ``strata`` maps stratum -> group -> images of that group in that stratum. A group
    may appear in several strata. ``pinned`` groups (already recorded in a persisted
    map) are never re-offered: they count toward the filled quota of every stratum
    they belong to, exactly as ``assign_splits(filled=...)`` does for one pool.
    """
    pinned = dict(pinned or {})
    if TRAIN not in ratios:
        raise ValueError(f"stratified allocation needs a {TRAIN!r} split, got {sorted(ratios)}")
    small = set(small_strata(strata, min_groups))

    totals = {name: sum(groups.values()) for name, groups in strata.items()}
    targets = {
        name: {split: fraction * totals[name] for split, fraction in ratios.items()}
        for name in strata
    }
    filled = {name: dict.fromkeys(ratios, 0) for name in strata}
    membership: dict[str, dict[str, int]] = {}
    for name, groups in strata.items():
        for key, count in groups.items():
            membership.setdefault(key, {})[name] = count
            if key in pinned:
                split = pinned[key]
                filled[name][split] = filled[name].get(split, 0) + count

    def share(key: str) -> float:
        return max(
            (count / totals[name] for name, count in membership[key].items() if name not in small),
            default=0.0,
        )

    unpinned = [key for key in membership if key not in pinned]
    ordered = sorted(
        unpinned,
        key=lambda k: (-share(k), hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()),
    )

    assignment: dict[str, SplitName] = {}
    for key in ordered:
        balanced = [name for name in membership[key] if name not in small]
        if not balanced:
            split = TRAIN
        else:
            score = {
                s: sum((targets[n][s] - filled[n][s]) / totals[n] for n in balanced) for s in ratios
            }
            split = max(ratios, key=lambda s: (score[s], s))
        assignment[key] = split
        for name, count in membership[key].items():
            filled[name][split] += count
    return _ensure_every_split(
        assignment, pinned, membership, targets, ratios, small=small, seed=seed
    )


def _ensure_every_split(
    assignment: Mapping[str, SplitName],
    pinned: Mapping[str, SplitName],
    membership: Mapping[str, Mapping[str, int]],
    targets: Mapping[str, Mapping[SplitName, float]],
    ratios: Mapping[SplitName, float],
    *,
    small: set[str],
    seed: int,
) -> dict[str, SplitName]:
    """No empty val/test per source (WP-R6): a balanced stratum (>= ``min_groups`` groups) that
    ends up with no group in some split takes one from a split that keeps at least one.

    Only groups this call assigned may move; a ``pinned`` group (an upstream-test group forced to
    ``test``, a never-eval group forced to ``train``, a group persisted in an earlier map) never
    does, so ``val`` is taken from ``train`` and never from the upstream test. A donor must not
    empty a split of another balanced stratum it belongs to. ``train`` donates first, then the
    split holding the most groups; the group closest to the missing split's target size moves.
    """
    result = dict(assignment)
    current = {**pinned, **result}

    def groups_in(name: str, split: SplitName) -> list[str]:
        return [k for k in membership if name in membership[k] and current.get(k) == split]

    for name in sorted(_strata_of(membership)):
        if name in small:
            continue
        for split in sorted(ratios):
            if groups_in(name, split):
                continue
            donors = sorted(
                (s for s in ratios if s != split and len(groups_in(name, s)) >= 2),
                key=lambda s: (s != TRAIN, -len(groups_in(name, s)), s),
            )
            for donor in donors:
                movable = [
                    k
                    for k in groups_in(name, donor)
                    if k in result
                    and all(
                        len(groups_in(other, donor)) >= 2
                        for other in membership[k]
                        if other not in small
                    )
                ]
                if not movable:
                    continue
                pick = min(
                    movable,
                    key=lambda k: (
                        abs(membership[k][name] - targets[name][split]),
                        hashlib.sha256(f"{seed}:{k}".encode()).hexdigest(),
                    ),
                )
                result[pick] = split
                current[pick] = split
                break
    return result


def _strata_of(membership: Mapping[str, Mapping[str, int]]) -> set[str]:
    return {name for per in membership.values() for name in per}


def achieved_by_stratum(
    strata: Mapping[str, Mapping[str, int]], assignment: Mapping[str, SplitName]
) -> dict[str, dict[SplitName, int]]:
    """Images per split within each stratum, for the generate summary and the tests."""
    out: dict[str, dict[SplitName, int]] = {}
    for name, groups in sorted(strata.items()):
        per: dict[SplitName, int] = {}
        for key, count in groups.items():
            per[assignment[key]] = per.get(assignment[key], 0) + count
        out[name] = per
    return out
