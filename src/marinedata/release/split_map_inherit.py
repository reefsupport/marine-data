"""Row-level inputs of an inheriting split map (D1, D2).

:func:`marinedata.release.generate_split_map` hands its enumerated rows and dHashes here and gets
back the constraints :func:`marinedata.splitmap_inherit.resolve_inherited_splits` applies: the
prior-release neighbour band and the out-of-distribution holdouts. Kept apart so the generator
stays readable.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field

from ..neardup import NearDupConfig, v1_band_pairs
from ..scan import SplitName
from ..splitmap import SplitMap, SplitMapError
from ..splitmap_inherit import band_constraints
from ..splitv2.release_ood import plan_ood, release_samples
from ..splitv2.rules import RulesConfig


@dataclass(frozen=True)
class InheritPlan:
    forced: dict[str, SplitName] = field(default_factory=dict)
    """Component -> split, from the prior-release neighbour band (never-eval / never-train)."""
    withhold: dict[str, str] = field(default_factory=dict)
    """Component -> reason: it neighbours prior train and prior evaluation images alike."""
    band_pairs: int = 0
    ood: dict[str, list[str]] = field(default_factory=dict)
    """Component -> the holdout names that matched one of its groups."""
    ood_header: dict[str, object] = field(default_factory=dict)


def plan_inheritance(
    rows: Sequence[tuple[str, str, str]],
    *,
    inherit: SplitMap,
    never_eval_sources: Collection[str],
    canonical: Mapping[str, str],
    hashes: Mapping[str, int],
    near_dup: NearDupConfig | None,
    ood_rules: RulesConfig | None,
    ood_metadata: Mapping[str, Mapping[str, object]] | None,
) -> InheritPlan:
    """``rows`` are ``(image_sha256, split_group, source_id)``; ``hashes`` the dHash of every
    eval-capable image (empty without ``near_dup``); ``canonical`` maps a group to its component.

    Raises ``SplitMapError`` when a constructible OOD holdout exceeds the size ceiling: a human
    narrows the yaml rule, nothing subsamples a holdout."""
    prior = inherit.assignments
    prior_split: dict[str, SplitName] = {}
    for sha, group, source in sorted(rows):
        if group in prior and source not in never_eval_sources:
            prior_split.setdefault(sha, prior[group])
    forced: dict[str, SplitName] = {}
    withhold: dict[str, str] = {}
    band: list[tuple[str, str, int]] = []
    if near_dup is not None and prior_split:
        band = v1_band_pairs(
            {sha: h for sha, h in hashes.items() if sha not in prior_split},
            {sha: hashes[sha] for sha in prior_split},
            union_max=near_dup.union_max,
            band_max=near_dup.exclude_max,
        )
        component_of = {sha: canonical.get(group, group) for sha, group, _ in rows}
        forced, withhold = band_constraints(band, v1_split=prior_split, component_of=component_of)
    ood: dict[str, list[str]] = {}
    header: dict[str, object] = {}
    if ood_rules is not None:
        plan = plan_ood(release_samples(rows, never_eval_sources, ood_metadata), ood_rules)
        if plan.issues:
            raise SplitMapError(
                "OOD holdouts exceed the size guards (narrow the rule in splits/v2.yaml; nothing "
                "subsamples a holdout): " + "; ".join(plan.issues)
            )
        for group, tags in sorted(plan.group_tags.items()):
            names = ood.setdefault(canonical.get(group, group), [])
            names.extend(tag for tag in tags if tag not in names)
        header = plan.header()
    return InheritPlan(
        forced=forced, withhold=withhold, band_pairs=len(band), ood=ood, ood_header=header
    )
