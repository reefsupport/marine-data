"""D-P tropical-province OOD holdout (5★ charter D-P(1), decided 2026-09-25, after
``docs/design/eval-decontamination-split-v2.md`` §3.2 was written — this overrides it).

The other six holdouts in ``registry/splits/v2.yaml`` are fully authored: their
``value`` is fixed at yaml-edit time. This seventh one is data-driven: the *rule*
(field, size floor, tropical-realm allow-list) is static, but *which* province it
fires on is picked at ``splits check`` time from the labelled pool. This module owns
that selection; once :func:`resolve_tropical_province_rule` has patched the yaml
rule's ``value``, :mod:`marinedata.splitv2.holdouts` matches it exactly like any
other ``field_eq`` rule.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .holdouts import Sample
from .rules import HoldoutRule, RulesConfig, effective_field

RULE_NAME = "ood-geo-tropical-province"
TARGET_SHARE = 0.05

# D-P(1) (charter): "ADD one tropical-reef PROVINCE holdout outside the Tropical
# Atlantic" — the Tropical Atlantic is our own operating realm and holds most of our
# own-labelled data, so it is excluded HERE, hard-coded, regardless of what
# `tropical_realms` in v2.yaml says. A future yaml edit that (by mistake) adds
# "Tropical Atlantic" to that list must never be able to route our own imagery OOD.
EXCLUDED_REALM = "Tropical Atlantic"

# Ad hoc lookup "rules" that reuse `rules.effective_field`'s tested fallback-filling
# logic to read a sample's realm/province — these are never matched against, only
# used to resolve a single field's effective value.
_REALM_FIELD = HoldoutRule(
    name="_dp_realm_lookup",
    order=0,
    kind="field_eq",
    field="meow_realm",
    value=None,
    fallback="geo_fallback",
    fallback_key="realm",
)
_PROVINCE_FIELD = HoldoutRule(
    name="_dp_province_lookup",
    order=0,
    kind="field_eq",
    field="meow_province",
    value=None,
    fallback="geo_fallback",
    fallback_key="province",
)


@dataclass(frozen=True)
class DPChoice:
    """Outcome of the D-P province selection. ``province is None`` means inert —
    "if none qualifies, document it" (charter D-P(1))."""

    province: str | None
    n_images: int
    labelled_total: int
    reason: str = ""

    @property
    def share(self) -> float:
        return self.n_images / self.labelled_total if self.labelled_total else 0.0

    def describe(self) -> str:
        """The exact line `splits check` prints (brief P3b)."""
        if self.province is None:
            return f"D-P holdout: none qualifies ({self.reason})"
        return f"D-P holdout: {self.province} ({self.n_images}, {100 * self.share:.2f}%)"


def labelled_total(samples: Sequence[Sample], never_eval: Sequence[str]) -> int:
    """Every image whose source is not tagged ``never_eval`` — pseudo-labelled/
    unlabelled pretrain sources never count toward the labelled pool (design §0's
    train-only pool is a separate line item, never part of this denominator)."""
    never_eval_set = set(never_eval)
    return sum(1 for s in samples if s.source_id not in never_eval_set)


def province_labelled_counts(
    samples: Sequence[Sample], config: RulesConfig, never_eval: Sequence[str]
) -> dict[str, tuple[str, int]]:
    """province -> (realm, count of labelled images), among labelled samples whose
    realm AND province both resolve (own field or per-source ``geo_fallback``)."""
    never_eval_set = set(never_eval)
    counts: dict[str, list[object]] = {}
    for s in samples:
        if s.source_id in never_eval_set:
            continue
        realm = effective_field(_REALM_FIELD, config, s)
        province = effective_field(_PROVINCE_FIELD, config, s)
        if realm is None or province is None:
            continue
        entry = counts.setdefault(province, [realm, 0])
        entry[1] += 1
    return {p: (r, n) for p, (r, n) in counts.items()}


def select_tropical_province(
    counts: Mapping[str, tuple[str, int]],
    *,
    tropical_realms: Sequence[str],
    labelled_total: int,
    min_images: int,
    target_share: float = TARGET_SHARE,
) -> DPChoice:
    """D-P(1): the tropical-realm province (never Tropical Atlantic) whose labelled
    count is closest to ``target_share`` of ``labelled_total``, with
    ``>= min_images``. Ties break by province name, ascending."""
    allowed_realms = {r for r in tropical_realms if r != EXCLUDED_REALM}
    candidates = {
        province: n
        for province, (realm, n) in counts.items()
        if realm in allowed_realms and n >= min_images
    }
    if not candidates:
        return DPChoice(
            None,
            0,
            labelled_total,
            reason=(
                f"no province in the tropical realms {sorted(allowed_realms)} "
                f"(excl. {EXCLUDED_REALM}) has >= {min_images} labelled images"
            ),
        )
    target = target_share * labelled_total
    province = min(candidates, key=lambda p: (abs(candidates[p] - target), p))
    return DPChoice(province, candidates[province], labelled_total)


def resolve_tropical_province_rule(config: RulesConfig, choice: DPChoice) -> RulesConfig:
    """Patch the data-driven holdout's ``value`` with the chosen province. Leaving it
    ``None`` when ``choice.province`` is ``None`` keeps the rule permanently inert —
    it can never match (``rules.matches`` never matches a rule whose value is
    ``None`` against a resolved sample value)."""
    holdouts = tuple(
        dataclasses.replace(rule, value=choice.province) if rule.data_driven else rule
        for rule in config.holdouts
    )
    return dataclasses.replace(config, holdouts=holdouts)
