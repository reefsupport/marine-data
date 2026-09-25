"""Group-level OOD holdout assignment (design §3.2) — the any-member-OOD rule.

A sample's ``split_group_id`` (WP-10) is the unit: if *any* member of a group matches a
holdout rule, the whole group is a candidate for that holdout. The **first** rule in
precedence order (``registry/splits/v2.yaml`` list order) that both matches a group and
is itself *constructible* (§3.2 size guard) wins that group's split; every constructible
rule the group matches is recorded in ``ood_tags``, not only the winner. A rule that
fails the size guard holds nothing out in this release — its candidates fall through to
the next rule, and eventually to ID, exactly as "missing metadata stays ID" does.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .rules import HoldoutRule, RulesConfig, matches

OOD_PREFIX = "ood-"


@dataclass(frozen=True)
class Sample:
    """One image's fields relevant to OOD holdout rules. Anything unset is ``None`` —
    "missing metadata never sends a sample OOD" (design §3.2)."""

    sha256: str
    split_group_id: str
    source_id: str
    meow_realm: str | None = None
    meow_province: str | None = None
    depth_m: float | None = None
    platform: str | None = None
    capture_datetime: dt.datetime | None = None


@dataclass(frozen=True)
class HoldoutOutcome:
    name: str
    constructible: bool
    n_images: int
    n_groups: int
    note: str = ""


@dataclass(frozen=True)
class OODResult:
    group_split: dict[str, str]  # split_group_id -> "ood-<name>", winners only
    group_tags: dict[str, list[str]]  # split_group_id -> every constructible match, in order
    outcomes: tuple[HoldoutOutcome, ...]  # one per rule, precedence order

    @property
    def ood_groups(self) -> set[str]:
        return set(self.group_split)


def _cutoff(rule: HoldoutRule) -> dt.datetime:
    value = rule.value
    if isinstance(value, dt.datetime):
        return value
    return dt.datetime.fromisoformat(str(value))


def bimodal_sources(samples: Sequence[Sample], rule: HoldoutRule) -> set[str]:
    """Sources with at least one group's sample before, and one at/after, the cutoff."""
    cutoff = _cutoff(rule)
    before: set[str] = set()
    after: set[str] = set()
    for s in samples:
        if s.capture_datetime is None:
            continue
        (after if s.capture_datetime >= cutoff else before).add(s.source_id)
    return before & after


def _group_matches(
    rule: HoldoutRule,
    config: RulesConfig,
    members: Sequence[Sample],
    bimodal: set[str],
    *,
    allowed: tuple[object, ...] | None = None,
) -> bool:
    for sample in members:
        if rule.requires_bimodal_source and sample.source_id not in bimodal:
            continue
        if matches(rule, config, sample, allowed=allowed):
            return True
    return False


def _constructible(images: int, groups: int, guards: Mapping[str, float]) -> bool:
    return images >= guards["min_images"] and groups >= guards["min_groups"]


def assign_ood(samples: Sequence[Sample], config: RulesConfig) -> OODResult:
    by_group: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_group[s.split_group_id].append(s)

    guards = config.size_guards
    matched: dict[str, set[str]] = {}
    outcomes: list[HoldoutOutcome] = []

    for rule in config.holdouts:
        bimodal = bimodal_sources(samples, rule) if rule.requires_bimodal_source else set()
        gids = {
            gid
            for gid, members in by_group.items()
            if _group_matches(rule, config, members, bimodal)
        }
        images = sum(len(by_group[g]) for g in gids)
        note = ""
        if not _constructible(images, len(gids), guards) and rule.widen_values:
            widened = {
                gid
                for gid, members in by_group.items()
                if _group_matches(rule, config, members, bimodal, allowed=rule.widen_values)
            }
            w_images = sum(len(by_group[g]) for g in widened)
            if _constructible(w_images, len(widened), guards):
                gids, images, note = widened, w_images, f"widened to {rule.widen_values}"
        ok = _constructible(images, len(gids), guards)
        if not ok:
            note = note or f"not constructible ({images} images, {len(gids)} groups)"
            matched[rule.name] = set()
        else:
            matched[rule.name] = gids
        outcomes.append(
            HoldoutOutcome(rule.name, ok, images if ok else 0, len(gids) if ok else 0, note)
        )

    group_split: dict[str, str] = {}
    group_tags: dict[str, list[str]] = defaultdict(list)
    for rule in config.holdouts:
        for gid in matched[rule.name]:
            group_tags[gid].append(rule.name)
            group_split.setdefault(gid, OOD_PREFIX + rule.name.removeprefix(OOD_PREFIX))

    return OODResult(group_split=group_split, group_tags=dict(group_tags), outcomes=tuple(outcomes))


def share_issues(
    outcomes: Sequence[HoldoutOutcome], eligible_total: int, guards: Mapping[str, float]
) -> list[str]:
    """CI-gate issues for the size ceiling (§3.2): a human must narrow the yaml rule,
    nothing here silently subsamples a holdout down to size."""
    issues: list[str] = []
    if eligible_total <= 0:
        return issues
    total_images = 0
    for outcome in outcomes:
        if not outcome.constructible:
            continue
        total_images += outcome.n_images
        share = outcome.n_images / eligible_total
        if share > guards["max_holdout_share"]:
            issues.append(
                f"{outcome.name}: {share:.1%} of the eligible corpus exceeds "
                f"{guards['max_holdout_share']:.0%}"
            )
    total_share = total_images / eligible_total
    max_total = guards["max_total_holdout_share"]
    if total_share > max_total:
        issues.append(f"all holdouts together: {total_share:.1%} exceeds {max_total:.0%}")
    return issues
