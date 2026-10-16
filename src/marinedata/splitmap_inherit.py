"""Inheriting a prior release's split map (v1.1.0 from v1.0): its assignments are frozen.

``resolve_splits`` extends a map under the same seed and refuses to join a merged component whose
members a prior map already split apart. A new release needs a different contract:

* every group the prior map names keeps its split, verbatim, whatever a new near-duplicate link or
  shared digest says: the prior map is embedded in the new one and never rewritten;
* a new group in a component with prior groups takes their split when they agree, and is
  *withheld* (no entry, listed in ``SplitMap.withheld``) when they disagree: it cannot be placed
  without contradicting one of them;
* the rest is allocated around the frozen quota exactly as before
  (:func:`~marinedata.strata.assign_splits_stratified`), except that a component an
  out-of-distribution rule matched goes to ``test_ood`` and takes no part in the 70/15/15 quota.

Precedence for a component of new groups, first match wins: inherited (a prior group in the
component), the prior-release neighbour band (:func:`band_constraints`), forced (a never-eval-only
component is train), OOD (``test_ood``), upstream test (``test``), then the stratified group hash.
A benchmark pin outranks the inherited split (design 3.1 rule 3): a component named in ``override``
moves whole to the pinned split, its prior groups included. Those moved groups, and no others, may
differ from the prior map: the append-only guard raises for any other, and
:attr:`InheritResult.overridden` lists them.
The caller (:func:`marinedata.release.generate_split_map`) builds the inputs; this module decides.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .scan import SplitName
from .splitmap import SplitMap, SplitMapError, load_split_map, save_split_map
from .strata import DEFAULT_MIN_GROUPS, TEST_OOD, TRAIN, assign_splits_stratified

_RATIO_TOLERANCE = 1e-9


@dataclass(frozen=True)
class InheritResult:
    """What :func:`resolve_inherited_splits` decided (the map itself is on disk)."""

    assignments: dict[str, SplitName]
    withheld: dict[str, str]
    inherited_groups: int
    """Groups the prior map names; every one is in ``assignments`` unchanged."""
    adopted_groups: int = 0
    """New groups that took a prior split because a prior group shares their component."""
    conflict_components: int = 0
    """Components whose prior groups the prior map had split apart (new groups withheld)."""
    ood_components: int = 0
    ood_groups: int = 0
    forced_components: int = 0
    ood_blocked: int = 0
    """Components an OOD rule matched that precedence placed elsewhere (prior, band, never-eval)."""
    overridden: tuple[str, ...] = ()
    """Prior groups whose split a benchmark pin changed (``override``), sorted."""


def load_inherited(path: str | Path) -> tuple[SplitMap, str]:
    """The prior release's map and the sha256 of its bytes; raises if it does not exist."""
    p = Path(path)
    prior = load_split_map(p)
    if prior is None:
        raise SplitMapError(f"{p} does not exist: --inherit needs the prior release's split map")
    return prior, hashlib.sha256(p.read_bytes()).hexdigest()


def inherit_header(prior: SplitMap, sha256: str) -> dict[str, object]:
    """What the new map records about the map it embeds."""
    return {
        "release": prior.release,
        "seed": prior.seed,
        "groups": len(prior.assignments),
        "sha256": sha256,
    }


def band_constraints(
    pairs: Iterable[tuple[str, str, int]],
    *,
    v1_split: Mapping[str, SplitName],
    component_of: Mapping[str, str],
) -> tuple[dict[str, SplitName], dict[str, str]]:
    """Constraints from new images within the near-duplicate band of prior images.

    ``pairs`` are ``(new_sha, prior_sha, hamming)`` beyond the union radius but inside the band
    (:func:`marinedata.neardup.v1_band_pairs`). A component near a prior train image is
    never-eval (forced ``train``); near a prior evaluation image it is never-train (forced to the
    nearest such image's split); near both it is withheld. Returns ``(forced, withheld)`` keyed by
    component."""
    hits: dict[str, list[tuple[int, str, SplitName]]] = {}
    for new_sha, prior_sha, distance in pairs:
        hits.setdefault(component_of[new_sha], []).append(
            (distance, prior_sha, v1_split[prior_sha])
        )
    forced: dict[str, SplitName] = {}
    withheld: dict[str, str] = {}
    for canon, found in sorted(hits.items()):
        train = [hit for hit in found if hit[2] == TRAIN]
        evaluation = sorted(hit for hit in found if hit[2] != TRAIN)
        if train and evaluation:
            withheld[canon] = (
                "within the near-duplicate band of prior-release train and evaluation images: "
                "neither never-eval nor never-train can hold"
            )
        elif train:
            forced[canon] = TRAIN
        else:
            forced[canon] = evaluation[0][2]
    return forced, withheld


def _members(merge_canonical: Mapping[str, str]) -> dict[str, list[str]]:
    by_canonical: dict[str, list[str]] = {}
    for group, canon in merge_canonical.items():
        by_canonical.setdefault(canon, []).append(group)
    return {canon: sorted(members) for canon, members in by_canonical.items()}


def _check_compatible(prior: SplitMap, ratios: Mapping[SplitName, float]) -> None:
    if prior.by != "group" or prior.stratify != "source":
        raise SplitMapError(
            f"cannot inherit a map with by={prior.by!r} stratify={prior.stratify!r}: the quota "
            "arithmetic needs by='group' and stratify='source'"
        )
    same = set(prior.ratios) == set(ratios) and all(
        abs(prior.ratios[name] - ratios[name]) < _RATIO_TOLERANCE for name in ratios
    )
    if not same:
        raise SplitMapError(
            f"the prior map used ratios={prior.ratios}, this call {dict(ratios)}: new groups "
            "are allocated against the quota the prior groups already filled"
        )


def resolve_inherited_splits(
    path: str | Path,
    counts: Mapping[str, int],
    ratios: Mapping[SplitName, float],
    *,
    prior: SplitMap,
    prior_sha256: str,
    strata: Mapping[str, Mapping[str, int]],
    merge_canonical: Mapping[str, str],
    seed: int = 0,
    now: str | None = None,
    release: str | None = None,
    min_groups: int = DEFAULT_MIN_GROUPS,
    forced: Mapping[str, SplitName] | None = None,
    withhold: Mapping[str, str] | None = None,
    ood: Mapping[str, Sequence[str]] | None = None,
    ood_header: Mapping[str, object] | None = None,
    near_dup: Mapping[str, object] | None = None,
    override: Mapping[str, SplitName] | None = None,
) -> InheritResult:
    """Write a fresh map at ``path`` that embeds ``prior`` and assigns the new components.

    ``counts``/``strata``/``merge_canonical`` come from :func:`~marinedata.splitmap.rows_to_counts`
    over every enumerated tree (prior and new). ``forced``: component -> split for components
    with no prior group (never-eval, band, upstream test). ``withhold``: component -> reason, a new
    component that must get no entry. ``ood``: component -> OOD rule names; those components go to
    ``test_ood`` unless a prior group or a ``forced``/``withhold`` entry placed them first.
    ``override``: component -> split for components a benchmark pin moves off their prior split
    (:func:`marinedata.benchmark_pins.apply_benchmark_pins`): every group of the component, prior
    ones included, takes it. ``seed`` seeds the allocation of new groups only; the prior seed is
    kept in ``inherits``. Raises ``SplitMapError`` if ``path`` exists or the prior map cannot be
    the base."""
    if load_split_map(path) is not None:
        raise SplitMapError(f"{path} already exists: an inheriting map is generated fresh once")
    _check_compatible(prior, ratios)
    covered = {key for groups in strata.values() for key in groups}
    if covered != set(counts):
        raise SplitMapError(
            f"strata cover {len(covered)} groups but counts has {len(counts)}: every group must "
            "belong to at least one stratum"
        )
    members = _members(merge_canonical)
    persisted = prior.assignments
    forced = dict(forced or {})
    withhold = dict(withhold or {})
    override = {canon: split for canon, split in (override or {}).items() if canon in counts}

    adopted: dict[str, SplitName] = {}
    conflicts: dict[str, list[str]] = {}
    for canon in counts:
        if canon in override:
            continue
        group_members = members.get(canon, [canon])
        found = sorted({persisted[m] for m in group_members if m in persisted})
        if len(found) == 1:
            adopted[canon] = found[0]
        elif found:
            conflicts[canon] = group_members
    fresh = [
        canon
        for canon in counts
        if canon not in adopted and canon not in conflicts and canon not in override
    ]

    withheld: dict[str, str] = {}
    for _canon, group_members in sorted(conflicts.items()):
        splits = " and ".join(sorted({persisted[m] for m in group_members if m in persisted}))
        for member in group_members:
            if member not in persisted:
                withheld[member] = (
                    f"near-duplicate of prior-release images the prior map put in {splits}: "
                    "no split can honour both"
                )
    withheld_fresh = {canon for canon in fresh if canon in withhold}
    for canon in sorted(withheld_fresh):
        for member in members.get(canon, [canon]):
            withheld[member] = withhold[canon]
    ood_components = {
        canon: list(tags)
        for canon, tags in sorted((ood or {}).items())
        if canon in fresh and canon not in withheld_fresh and canon not in forced
    }
    aside = set(conflicts) | withheld_fresh | set(ood_components)
    forced_new = {
        canon: split for canon, split in forced.items() if canon in fresh and canon not in aside
    }

    fixed = {**adopted, **forced_new, **override}
    in_quota = {canon: split for canon, split in fixed.items() if split in ratios}
    outside = set(fixed) - set(in_quota)  # e.g. a prior test_ood group: not part of the quota
    alloc_strata = {
        name: kept
        for name, groups in strata.items()
        if (kept := {g: n for g, n in groups.items() if g not in aside and g not in outside})
    }
    alloc_groups = {group for groups in alloc_strata.values() for group in groups}
    allocated: dict[str, SplitName] = {}
    if alloc_strata:
        allocated = assign_splits_stratified(
            alloc_strata,
            ratios,
            seed=seed,
            pinned={canon: split for canon, split in in_quota.items() if canon in alloc_groups},
            min_groups=min_groups,
        )

    assignments: dict[str, SplitName] = dict(persisted)
    for canon in counts:
        if canon in aside:
            continue
        split = fixed[canon] if canon in fixed else allocated[canon]
        for member in members.get(canon, [canon]):
            if canon in override:
                assignments[member] = split  # a benchmark pin moves the whole component
            else:
                assignments.setdefault(member, split)
    ood_rules: dict[str, list[str]] = {}
    for canon, tags in ood_components.items():
        for member in members.get(canon, [canon]):
            assignments[member] = TEST_OOD  # no prior group is in an OOD component
            ood_rules[member] = tags
    overridden: list[str] = []
    for key, split in persisted.items():
        if assignments[key] == split:
            continue
        if merge_canonical.get(key, key) not in override:  # the append-only contract's guard
            raise SplitMapError(
                f"prior assignment moved: {key!r} {split!r} -> {assignments[key]!r}"
            )
        overridden.append(key)

    save_split_map(
        path,
        SplitMap(
            by=prior.by,
            seed=seed,
            ratios=dict(ratios),
            assignments=assignments,
            generated_at=now if now is not None else datetime.now(UTC).isoformat(),
            release=release or "",
            stratify=prior.stratify,
            near_dup=dict(near_dup or {}),
            inherits=inherit_header(prior, prior_sha256),
            ood=dict(ood_header or {}),
            ood_rules=ood_rules,
            withheld=withheld,
        ),
    )
    return InheritResult(
        assignments=assignments,
        withheld=withheld,
        inherited_groups=len(persisted),
        adopted_groups=sum(
            1 for canon in adopted for m in members.get(canon, [canon]) if m not in persisted
        ),
        conflict_components=len(conflicts),
        ood_components=len(ood_components),
        ood_groups=len(ood_rules),
        forced_components=len(forced_new),
        ood_blocked=len(ood or {}) - len(ood_components),
        overridden=tuple(sorted(overridden)),
    )
