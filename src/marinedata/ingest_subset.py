"""Measured totals and subset strategies for TB-scale ingest specs (SPEC-w3).

An ingest spec (``registry/ingest-specs/<id>.yaml``) may carry two documentation-only
blocks that the runner ignores but the WP-6c server Job reads when it plans a batch:

``measured``
    Totals read from upstream *metadata* (an API count, a bucket listing, a CSV pass),
    never from an estimate. ``items``/``bytes`` may be null when the upstream exposes
    no size (then ``bytes_basis`` says why), and ``bytes_basis`` names how bytes were
    obtained (``listing``, ``hf-tree``, ``pixels x sampled bytes/px``...).
``subset``
    Required when a source is over :data:`SUBSET_ITEMS` items or :data:`SUBSET_BYTES`
    bytes: a numeric ``target_items``, the ``stratify`` keys, and the selection ``rule``.

:func:`check_spec` returns the list of problems (empty = ok); the w3 dry-run driver and
``tests/test_ingest_subset.py`` run it over every spec that has a ``measured`` block.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

SUBSET_ITEMS = 1_000_000
SUBSET_BYTES = 10**12  # 1 TB (decimal, as upstreams report it)
JOB_MB_S = 200.0  # WP-6c server ingest Job
MAC_MB_S = 10.0  # the Mac uplink


@dataclass(frozen=True)
class Measured:
    items: int | None
    bytes: int | None
    method: str
    date: str
    bytes_basis: str = ""


@dataclass(frozen=True)
class SubsetPlan:
    target_items: int
    stratify: tuple[str, ...]
    rule: str
    target_bytes: int | None = None
    cap_per_stratum: int | None = None


def _opt_int(raw: Mapping[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer or null, got {value!r}")
    return value


def parse_measured(raw: Mapping[str, Any]) -> Measured:
    method, date = str(raw.get("method") or ""), str(raw.get("date") or "")
    if not method or not date:
        raise ValueError("measured needs a method and a date")
    return Measured(
        items=_opt_int(raw, "items"),
        bytes=_opt_int(raw, "bytes"),
        method=method,
        date=date,
        bytes_basis=str(raw.get("bytes_basis") or ""),
    )


def parse_subset(raw: Mapping[str, Any] | None) -> SubsetPlan | None:
    if not raw:
        return None
    target = _opt_int(raw, "target_items")
    if not target:
        raise ValueError("subset.target_items must be a positive integer")
    stratify = raw.get("stratify") or []
    if (
        not isinstance(stratify, list)
        or not stratify
        or not all(isinstance(s, str) for s in stratify)
    ):
        raise ValueError("subset.stratify must be a non-empty list of strings")
    rule = str(raw.get("rule") or "")
    if not rule:
        raise ValueError("subset.rule is required")
    return SubsetPlan(
        target_items=target,
        stratify=tuple(stratify),
        rule=rule,
        target_bytes=_opt_int(raw, "target_bytes"),
        cap_per_stratum=_opt_int(raw, "cap_per_stratum"),
    )


def subset_required(items: int | None, n_bytes: int | None) -> bool:
    return (items or 0) > SUBSET_ITEMS or (n_bytes or 0) > SUBSET_BYTES


def transfer_hours(n_bytes: int | None, mb_per_s: float) -> float | None:
    if n_bytes is None:
        return None
    return n_bytes / (mb_per_s * 1e6) / 3600


def check_spec(raw: Mapping[str, Any]) -> list[str]:
    """Problems with a spec's ``measured``/``subset`` blocks (empty list = ok)."""
    problems: list[str] = []
    if not raw.get("measured"):
        return ["no measured block"]
    try:
        m = parse_measured(raw["measured"])
    except ValueError as exc:
        return [f"measured: {exc}"]
    if m.items is not None and m.bytes is None and not m.bytes_basis:
        problems.append("measured.bytes is null without a bytes_basis explaining why")
    try:
        plan = parse_subset(raw.get("subset"))
    except ValueError as exc:
        return [*problems, f"subset: {exc}"]
    if subset_required(m.items, m.bytes) and plan is None:
        problems.append(f"over {SUBSET_ITEMS:,} items or 1 TB but no subset block")
    if plan and m.items is not None and plan.target_items > m.items:
        problems.append(f"subset.target_items {plan.target_items} > measured items {m.items}")
    return problems
