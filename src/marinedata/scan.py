"""Streaming corpus scan — split and index a corpus without holding it in memory.

``DatasetBuilder.build()` materialises every :class:`~marinedata.sample.Sample` in a
list. That is correct, simple, and fine at the scale the labelled corpus sits at today.
It stops being fine sooner than it looks:

    measured, deep-sized, with realistic paths and metadata
    per Sample                    7,772 bytes
        4,000 samples (today)         0.03 GB
    1,000,000 samples (BenthicNet-1M)  7.8 GB   ← already registered, already CC-BY
   11,400,000 samples (BenthicNet-11M) 88.6 GB
   50,000,000 samples                 388.6 GB

So the ceiling is not a TB of *bytes* — it is roughly one million *samples*, and one
already-permissive dataset in the registry crosses it alone.

The fix does not need a new storage system. Loaders are already generators; only the
builder was eager. What genuinely requires whole-corpus knowledge is the label index and
the split assignment, and both need only **counters**, not samples:

    group counts   one int per site         thousands of entries
    label counts   one int per class        hundreds of entries

Both are constant in corpus size. So: scan once collecting counters, compute the plan,
then stream samples through it. Memory becomes independent of corpus size, and the split
assignment is bit-identical to the eager path because both call the same function.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from .sample import Sample
from .schema import Axis

SplitName = str


def group_key(sample: Sample, by: str) -> str:
    """The unit that must not be divided across splits.

    Kept here rather than inlined so the eager and streaming paths cannot drift — a
    split that differs between them would be almost impossible to notice and would
    invalidate every comparison between runs.
    """
    if by == "source":
        return sample.source_id
    if by == "site":
        return f"{sample.source_id}/{sample.meta.get('partition', '')}"
    if by == "random":
        return f"{sample.source_id}/{sample.key}"
    raise ValueError(f"unknown split strategy '{by}' (site | source | random)")


@dataclass(frozen=True)
class CorpusScan:
    """Counters describing a corpus, gathered without retaining any sample.

    Everything here is O(groups + classes), never O(samples).
    """

    total: int = 0
    groups: Counter[str] = field(default_factory=Counter)
    labels: dict[Axis, Counter[str]] = field(default_factory=dict)
    sources: Counter[str] = field(default_factory=Counter)
    supervised: Counter[str] = field(default_factory=Counter)

    def summary(self) -> str:
        axes = "  ".join(
            f"{axis}={count / max(self.total, 1):.0%}"
            for axis, count in sorted(self.supervised.items())
        )
        return (
            f"scan: {self.total:,} samples · {len(self.groups):,} groups · "
            f"{len(self.sources)} source(s)\n  axes  {axes}"
        )


def scan(samples: Iterator[Sample], *, by: str = "site") -> CorpusScan:
    """Consume a sample stream and return only counters.

    The stream is consumed exactly once and nothing is retained, so this runs in constant
    memory over a corpus of any size.
    """
    total = 0
    groups: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    supervised: Counter[str] = Counter()
    labels: dict[Axis, Counter[str]] = {}

    for sample in samples:
        total += 1
        groups[group_key(sample, by)] += 1
        sources[sample.source_id] += 1
        for axis in sample.supervised:
            supervised[axis.value] += 1
        for axis, value in sample.labels.items():
            labels.setdefault(axis, Counter())[value.node_id] += 1

    return CorpusScan(
        total=total, groups=groups, labels=labels, sources=sources, supervised=supervised
    )


def assign_splits(
    counts: dict[str, int],
    ratios: dict[SplitName, float],
    *,
    seed: int = 0,
    total: int | None = None,
    filled: dict[SplitName, int] | None = None,
) -> dict[str, SplitName]:
    """Pack whole groups into splits, weighted by sample count.

    Shared by the eager and streaming paths so their assignments are identical.

    Groups are ordered by a seeded hash — deterministic and stable as data arrives — then
    sorted largest-first and given to whichever split is furthest below its target share.
    Placing the big groups while every split is still empty lands far closer to target
    than meeting them last with no room left.

    Groups are never divided. That is the entire point of grouping: consecutive transect
    frames overlap heavily, so splitting one across train and test leaks near-duplicates
    and inflates every metric.

    ``total`` and ``filled`` let a caller assign only the groups *not yet* pinned by a
    persisted map, against the quota the pinned groups already used, without resorting
    them: pass the corpus-wide total and each split's already-filled count, and only the
    still-unassigned ``counts`` here. Omit both for the original, whole-corpus behaviour.
    """
    ordered = sorted(
        counts,
        key=lambda k: (-counts[k], hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()),
    )
    if total is None:
        total = sum(counts.values())
    targets = {name: fraction * total for name, fraction in ratios.items()}
    filled = {name: (filled or {}).get(name, 0) for name in ratios}

    assignment: dict[str, SplitName] = {}
    for key in ordered:
        name = max(ratios, key=lambda n: (targets[n] - filled[n], n))
        assignment[key] = name
        filled[name] += counts[key]
    return assignment


def check_ratios(
    achieved: dict[SplitName, int],
    ratios: dict[SplitName, float],
    total: int,
    *,
    tolerance: float | None,
    by: str,
    largest_group: int,
) -> None:
    """Raise if the achieved split proportions miss the request by more than tolerance.

    Whole groups cannot be divided, so a corpus dominated by one huge group may be unable
    to hit the requested ratios however well it is packed. That is a real constraint, not
    a bug — but it must be reported, because a test split that is 3% instead of 15%
    yields confident metrics over a fraction of the data.
    """
    if tolerance is None or by == "random" or not total:
        return
    skewed = {
        name: count / total
        for name, count in achieved.items()
        if abs(count / total - ratios[name]) > tolerance
    }
    if not skewed:
        return
    got = "  ".join(f"{n}={v:.1%}" for n, v in sorted(skewed.items()))
    want = "  ".join(f"{n}={ratios[n]:.1%}" for n in sorted(skewed))
    raise ValueError(
        f"split(by={by!r}) could not hit the requested ratios within {tolerance:.0%}: "
        f"wanted {want}, got {got}. Group sizes are too uneven to divide this way — the "
        f"largest group holds {largest_group:,} of {total:,} samples. Rebalance the "
        f"corpus, relax `tolerance`, or split by a finer unit."
    )
