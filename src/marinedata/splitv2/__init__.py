"""Split v2 — pools, OOD holdouts, group-size-aware allocator (design §3, package P3).

``rules`` loads and evaluates ``registry/splits/v2.yaml``; ``holdouts`` does the
any-member-OOD group aggregation; ``allocate`` is the ratio-target allocator with its
repair pass; ``mapfile`` is the SPLIT_MAP v2 schema and append-only persistence.
``marinedata.cli_splits`` wires all four into ``marinedata splits check``.
"""

from .allocate import AllocationResult, StratumPlan, achieved_by_stratum, allocate
from .holdouts import HoldoutOutcome, OODResult, Sample, assign_ood, share_issues
from .mapfile import SplitMapV2, load, merge_append_only, save
from .rules import HoldoutRule, RulesConfig, load_rules, matches, rules_sha256

__all__ = [
    "AllocationResult",
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
    "load",
    "load_rules",
    "matches",
    "merge_append_only",
    "rules_sha256",
    "save",
    "share_issues",
]
