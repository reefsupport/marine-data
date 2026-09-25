"""Load and evaluate ``registry/splits/v2.yaml`` (design §3.2) — one sample at a time.

:mod:`marinedata.splitv2.holdouts` does the group-level any-member-OOD aggregation;
this module owns only the per-sample predicate: does *this* sample match *this* rule,
after filling a missing field from the yaml's per-source fallback tables. Missing
metadata with no fallback never matches — it is the caller's job to leave such a
sample in-distribution (ID), never to treat "no match" as an error.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

try:  # libyaml is faster and present in most environments
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover - pure-Python fallback
    from yaml import SafeLoader as _Loader

RuleKind = Literal["source_eq", "field_eq", "field_gte"]


class RulesError(Exception):
    """``registry/splits/v2.yaml`` is malformed or fails a load-time check."""


@dataclass(frozen=True)
class HoldoutRule:
    name: str
    order: int
    kind: RuleKind
    value: Any
    field: str | None = None
    fallback: str | None = None  # name of a lookup table in RulesConfig
    fallback_key: str | None = None  # sub-key within that table's per-source entry
    widen_values: tuple[Any, ...] | None = None
    requires_bimodal_source: bool = False


@dataclass(frozen=True)
class RulesConfig:
    raw: dict[str, Any]
    path: Path
    seed: int
    ratios: dict[str, float]
    min_groups: int
    tolerance: dict[str, float]
    size_guards: dict[str, float]
    allocator: dict[str, float]
    holdouts: tuple[HoldoutRule, ...]
    geo_fallback: dict[str, dict[str, str]]
    platform_fallback: dict[str, str]
    source_min_depth_m: dict[str, float]

    def lookup_table(self, name: str) -> dict[str, Any]:
        return {
            "geo_fallback": self.geo_fallback,
            "platform_fallback": self.platform_fallback,
            "source_min_depth_m": self.source_min_depth_m,
        }[name]


def load_rules(path: str | Path) -> RulesConfig:
    path = Path(path)
    raw = yaml.load(path.read_text(), Loader=_Loader)
    if raw.get("schema_version") != 2:
        raise RulesError(f"{path}: schema_version must be 2, got {raw.get('schema_version')!r}")
    holdouts = tuple(
        HoldoutRule(
            name=h["name"],
            order=h["order"],
            kind=h["kind"],
            value=h.get("value"),
            field=h.get("field"),
            fallback=h.get("fallback"),
            fallback_key=h.get("fallback_key"),
            widen_values=tuple(h["widen_values"]) if h.get("widen_values") else None,
            requires_bimodal_source=bool(h.get("requires_bimodal_source", False)),
        )
        for h in sorted(raw["holdouts"], key=lambda h: h["order"])
    )
    names = [h.name for h in holdouts]
    if len(set(names)) != len(names):
        raise RulesError(f"{path}: duplicate holdout names in {names}")
    if len(holdouts) != 6:
        raise RulesError(f"{path}: expected 6 holdouts (design §3.2), got {len(holdouts)}")
    return RulesConfig(
        raw=raw,
        path=path,
        seed=raw["seed"],
        ratios=dict(raw["ratios"]),
        min_groups=raw["min_groups"],
        tolerance=dict(raw["tolerance"]),
        size_guards=dict(raw["size_guards"]),
        allocator=dict(raw["allocator"]),
        holdouts=holdouts,
        geo_fallback=dict(raw.get("geo_fallback") or {}),
        platform_fallback=dict(raw.get("platform_fallback") or {}),
        source_min_depth_m=dict(raw.get("source_min_depth_m") or {}),
    )


def rules_sha256(config: RulesConfig) -> str:
    """Stable digest of the whole document — mirrors ``benchmarks.benchmarks_sha256``."""
    canonical = json.dumps(config.raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _fallback_value(rule: HoldoutRule, config: RulesConfig, source_id: str) -> Any:
    if rule.fallback is None:
        return None
    table = config.lookup_table(rule.fallback)
    entry = table.get(source_id)
    if entry is None:
        return None
    if rule.fallback_key is not None:
        return entry.get(rule.fallback_key) if isinstance(entry, dict) else None
    return entry


def effective_field(rule: HoldoutRule, config: RulesConfig, sample: Any) -> Any:
    """The sample's own value for ``rule.field``, else the per-source fallback."""
    value = getattr(sample, rule.field) if rule.field else None
    if value is not None:
        return value
    return _fallback_value(rule, config, sample.source_id)


def matches(rule: HoldoutRule, config: RulesConfig, sample: Any, *, allowed: Any = None) -> bool:
    """True if ``sample`` matches ``rule``. Missing metadata (no value, no fallback)
    never matches — the design's "missing geography stays ID" invariant lives here.

    ``allowed`` overrides ``rule.value``/``rule.widen_values`` for the widened re-check
    the orchestrator does for #5 (design §3.2: "widens once to {auv, towed}").
    """
    if rule.kind == "source_eq":
        return sample.source_id == rule.value
    value = effective_field(rule, config, sample)
    if value is None:
        return False
    if rule.kind == "field_eq":
        candidates = allowed if allowed is not None else (rule.value,)
        return value in candidates
    if rule.kind == "field_gte":
        return value >= rule.value
    raise RulesError(f"unknown rule kind {rule.kind!r}")  # pragma: no cover - guarded at load
