"""Runtime subset selection for an ingest spec's ``subset`` block (WP-6e-B, D-AB).

The ``subset`` block stays documentation for every spec (``ingest_subset.check_spec``)
unless it sets ``enforce: true``; then the runner applies it per decoded sample:

``filters``
    Predicates on a sample's labels/fields (all must hold), each
    ``{field, op, value}`` with ``op`` in ``eq ne in not_in``.
``stratify`` + ``cap_per_stratum``
    At most ``cap`` samples per group (the tuple of the ``stratify`` values).
``group_counts`` (optional, ``{"a|b": n}`` or a JSON file path)
    Upstream count per group. With it the choice is a deterministic hash draw: a sample
    is kept when ``hash(seed, upstream_id)`` falls under ``oversample * cap / n`` — the
    same samples win whatever the fetch order. The hard cap then trims the (small)
    overshoot in upstream order, which is itself pinned by the spec's revision.

:func:`select_bottom_k` is the exact, order-independent variant for manifest builders
that see every candidate up front (keep the ``cap`` lowest hashes per group).
"""

from __future__ import annotations

import hashlib
import heapq
import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

OPS = ("eq", "ne", "in", "not_in")


def hash_unit(seed: str, key: str) -> float:
    """Deterministic uniform [0, 1) from ``seed`` + ``key`` (blake2b, first 8 bytes)."""
    h = hashlib.blake2b(f"{seed}\x00{key}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big") / 2**64


@dataclass(frozen=True)
class Predicate:
    field: str
    op: str
    value: Any

    @classmethod
    def parse(cls, raw: Mapping[str, Any]) -> Predicate:
        op = str(raw.get("op", "eq"))
        if op not in OPS or not raw.get("field"):
            raise ValueError(f"subset filter needs a field and op in {OPS}: {dict(raw)!r}")
        value = raw.get("value")
        if op in ("in", "not_in"):
            if not isinstance(value, list):
                raise ValueError(f"subset filter {op} needs a list value: {dict(raw)!r}")
            value = frozenset(str(v) for v in value)
        return cls(str(raw["field"]), op, value)

    def holds(self, row: Mapping[str, Any]) -> bool:
        got = row.get(self.field)
        got = None if got is None else str(got)
        if self.op == "eq":
            return got == str(self.value)
        if self.op == "ne":
            return got != str(self.value)
        if self.op == "in":
            return got in self.value
        return got not in self.value


@dataclass
class SubsetFilter:
    predicates: tuple[Predicate, ...] = ()
    stratify: tuple[str, ...] = ()
    cap: int | None = None
    seed: str = "marinedata-subset-v1"
    group_counts: Mapping[str, int] = field(default_factory=dict)
    oversample: float = 1.1
    kept: Counter = field(default_factory=Counter)
    seen: int = 0
    rejected_filter: int = 0
    rejected_hash: int = 0
    rejected_cap: int = 0

    @classmethod
    def from_spec(
        cls, subset: Mapping[str, Any] | None, base: Path | None = None
    ) -> SubsetFilter | None:
        """``None`` unless the block says ``enforce: true`` (documentation-only otherwise)."""
        if not subset or subset.get("enforce") is not True:
            return None
        counts = subset.get("group_counts") or {}
        if isinstance(counts, str):
            path = Path(counts)
            if not path.is_absolute() and base is not None:
                path = base / path
            counts = json.loads(path.read_text())
        cap = subset.get("cap_per_stratum")
        return cls(
            predicates=tuple(Predicate.parse(p) for p in subset.get("filters") or []),
            stratify=tuple(subset.get("stratify") or ()) if cap else (),
            cap=int(cap) if cap else None,
            seed=str(subset.get("seed") or cls.seed),
            group_counts={str(k): int(v) for k, v in dict(counts).items()},
            oversample=float(subset.get("oversample", 1.1)),
        )

    def group_of(self, row: Mapping[str, Any]) -> str:
        return "|".join("" if row.get(k) is None else str(row.get(k)) for k in self.stratify)

    def admit_row(self, key: str, row: Mapping[str, Any]) -> bool:
        self.seen += 1
        if not all(p.holds(row) for p in self.predicates):
            self.rejected_filter += 1
            return False
        group = self.group_of(row)
        if self.cap is not None:
            n = self.group_counts.get(group)
            if n and n > self.cap and hash_unit(self.seed, key) >= self.oversample * self.cap / n:
                self.rejected_hash += 1
                return False
            if self.kept[group] >= self.cap:
                self.rejected_cap += 1
                return False
        self.kept[group] += 1
        return True

    def admit(self, decoded: Any) -> bool:
        """Runner hook: the row is the decoded sample's ``fields`` overlaid by its ``labels``."""
        row = {**dict(decoded.fields or {}), **dict(decoded.labels or {})}
        return self.admit_row(decoded.upstream_id, row)

    def stats(self) -> dict[str, int]:
        return {
            "seen": self.seen,
            "kept": sum(self.kept.values()),
            "groups": len(self.kept),
            "rejected_filter": self.rejected_filter,
            "rejected_hash": self.rejected_hash,
            "rejected_cap": self.rejected_cap,
        }


def expected_total(counts: Mapping[str, int], cap: int) -> int:
    """Exact size of a capped selection given per-group counts: sum(min(n, cap))."""
    return sum(min(int(n), cap) for n in counts.values())


def select_bottom_k(
    rows: Iterable[tuple[str, Mapping[str, Any]]],
    flt: SubsetFilter,
) -> list[str]:
    """Exact variant: every candidate seen up front, keep the ``cap`` lowest hashes per
    group (predicates first). Returns kept keys, sorted — independent of input order."""
    heaps: dict[str, list[tuple[float, str]]] = {}
    for key, row in rows:
        if not all(p.holds(row) for p in flt.predicates):
            continue
        h = heaps.setdefault(flt.group_of(row), [])
        item = (-hash_unit(flt.seed, key), key)
        if flt.cap is None or len(h) < flt.cap:
            heapq.heappush(h, item)
        elif item > h[0]:
            heapq.heapreplace(h, item)
    return sorted(k for h in heaps.values() for _, k in h)
