"""Split v2 — pools, OOD holdouts, group-size-aware allocator (design §3, package P3).

``rules`` loads and evaluates ``registry/splits/v2.yaml``; ``holdouts`` does the
any-member-OOD group aggregation; ``dp_province`` resolves the data-driven D-P(1)
tropical-province holdout's value before ``holdouts`` runs; ``allocate`` is the
ratio-target allocator with its repair pass; ``mapfile`` is the SPLIT_MAP v2 schema
and append-only persistence. ``marinedata.cli_splits`` wires all of these into
``marinedata splits check``.
"""

from .allocate import AllocationResult, StratumPlan, achieved_by_stratum, allocate
from .dp_province import (
    DPChoice,
    labelled_total,
    province_labelled_counts,
    resolve_tropical_province_rule,
    select_tropical_province,
)
from .holdouts import HoldoutOutcome, OODResult, Sample, assign_ood, share_issues
from .mapfile import SplitMapV2, load, merge_append_only, save
from .rules import HoldoutRule, RulesConfig, load_rules, matches, rules_sha256

__all__ = [
    "AllocationResult",
    "DPChoice",
    "HoldoutOutcome",
    "HoldoutRule",
    "OODResult",
    "RulesConfig",
    "Sample",
    "SplitMapV2",
    "StratumPlan",
    "achieved_by_stratum",
    "allocate",
    "assign_ood",
    "labelled_total",
    "load",
    "load_rules",
    "matches",
    "merge_append_only",
    "province_labelled_counts",
    "resolve_tropical_province_rule",
    "rules_sha256",
    "save",
    "select_tropical_province",
    "share_issues",
]
