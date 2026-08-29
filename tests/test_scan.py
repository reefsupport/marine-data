"""Streaming scan tests — constant memory, and identical results to the eager path.

The eager builder holds every Sample in a list. Measured at 7,772 bytes deep-sized with
realistic paths, that is 7.8 GB at one million samples — and BenthicNet-1M is already in
the registry under CC-BY. The ceiling is not a terabyte of bytes; it is about a million
samples, and one already-permissive dataset crosses it alone.

The two paths must agree exactly. A split that differed between them would be nearly
impossible to notice and would invalidate every comparison between runs.
"""

from __future__ import annotations

import tracemalloc
from collections.abc import Iterator
from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.builder import Dataset
from marinedata.labelindex import LabelIndex
from marinedata.lineage import build_lineage
from marinedata.sample import LabelValue, Sample
from marinedata.scan import assign_splits, group_key, scan
from marinedata.schema import Axis

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


_SYNTH_IMAGE = Path("/data/big/placeholder.jpg")
"""Shared by every synthetic sample below — deliberately not one `Path(...)` per
sample. `scan()` never reads `.image`; only `key`, `meta`, `supervised` and `labels`
matter to it. But `pathlib` interns every *new* path component for the life of the
process and never frees it, so a memory test running late in a large, growing suite
pays for whatever a prior test's `Path()` calls left the global intern table primed to
do next: even a handful of genuinely new components can trigger a table resize whose
size depends on the table's accumulated state, not on this test's own sample count —
bounding this test's *own* cardinality doesn't fix that, since the trigger isn't under
its control. Constructing zero new `Path` objects per sample removes the coupling
entirely, which is the only way to keep this test's signal about `scan()` and not
about whatever ran before it."""


def synth(count: int, *, sites: int = 40) -> Iterator[Sample]:
    for i in range(count):
        site = f"site{i % sites}"
        yield Sample(
            source_id="big",
            key=f"{site}/{i}",
            image=_SYNTH_IMAGE,
            labels={Axis.TAXON: LabelValue("HC" if i % 3 else "SC")},
            supervised=frozenset({Axis.TAXON}),
            meta={"partition": site},
        )


def test_scan_counts_without_retaining_samples() -> None:
    result = scan(synth(5_000, sites=10), by="site")
    assert result.total == 5_000
    assert len(result.groups) == 10
    assert sum(result.groups.values()) == 5_000
    assert result.labels[Axis.TAXON]["HC"] + result.labels[Axis.TAXON]["SC"] == 5_000
    assert result.supervised["taxon"] == 5_000


def test_scan_memory_is_flat_in_corpus_size() -> None:
    """The claim the whole module rests on. Measured, not asserted.

    Eager peaks at ~1.7 GB for a million samples; the scan stays under 10 MB because it
    is bounded by GROUP count, not sample count.
    """
    peaks = []
    for count in (50_000, 500_000):
        tracemalloc.start()
        scan(synth(count), by="site")
        peaks.append(tracemalloc.get_traced_memory()[1])
        tracemalloc.stop()

    assert peaks[1] < 10_000_000, f"scan peaked at {peaks[1] / 1e6:.1f} MB"
    # A 10x corpus must not cost 10x memory. Allow generous slack for allocator noise.
    assert peaks[1] < peaks[0] * 3, f"memory grew with corpus: {peaks[0]} -> {peaks[1]}"


def test_streaming_split_hits_requested_ratios() -> None:
    result = scan(synth(200_000, sites=40), by="site")
    assignment = assign_splits(dict(result.groups), RATIOS)

    sizes: dict[str, int] = {}
    for key, count in result.groups.items():
        sizes[assignment[key]] = sizes.get(assignment[key], 0) + count

    for name, want in RATIOS.items():
        got = sizes[name] / result.total
        assert abs(got - want) < 0.02, f"{name}: wanted {want:.0%}, got {got:.1%}"


def test_streaming_and_eager_splits_are_identical(registry: Registry) -> None:
    """Both paths call assign_splits, so agreement is structural — pin it anyway."""
    samples = list(synth(4_000, sites=17))
    schema = registry.label_schema("rs-benthic-v1")
    eager = Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )
    eager.split(by="site", ratios=RATIOS)

    result = scan(iter(samples), by="site")
    assignment = assign_splits(dict(result.groups), RATIOS)

    for name, positions in eager.splits.items():
        for position in positions:
            key = group_key(samples[position], "site")
            assert assignment[key] == name, f"{key}: eager={name} streaming={assignment[key]}"


def test_label_index_from_counts_matches_from_samples(registry: Registry) -> None:
    samples = list(synth(2_000, sites=8))
    schema = registry.label_schema("rs-benthic-v1")
    result = scan(iter(samples), by="site")

    from_samples = LabelIndex.from_samples(samples, schema)
    from_counts = LabelIndex.from_counts(result.labels, schema)
    assert from_counts.num_classes() == from_samples.num_classes()
    assert from_counts.axes[Axis.TAXON].classes == from_samples.axes[Axis.TAXON].classes


def test_group_key_rejects_unknown_strategy() -> None:
    sample = next(synth(1))
    with pytest.raises(ValueError, match="unknown split strategy"):
        group_key(sample, "by-vibes")


def test_assign_splits_never_divides_a_group() -> None:
    counts = {f"site{i}": 100 * (i + 1) for i in range(20)}
    assignment = assign_splits(counts, RATIOS)
    assert set(assignment) == set(counts)
    assert len(set(assignment.values())) == 3


def test_assign_splits_is_deterministic() -> None:
    counts = {f"site{i}": 100 * (i + 1) for i in range(20)}
    assert assign_splits(counts, RATIOS, seed=7) == assign_splits(counts, RATIOS, seed=7)


def test_seed_only_matters_when_group_sizes_tie() -> None:
    """Documents real behaviour that surprised me while writing these tests.

    Packing sorts largest-first, so when every group is a different size the assignment
    is fully determined by size and the seed changes nothing. The hash is a TIEBREAK, not
    a shuffle. That is desirable — a corpus of distinct sites gives the same split
    whatever seed you pass — but it means seed sweeps only vary anything when sizes tie.
    """
    distinct = {f"site{i}": 100 * (i + 1) for i in range(20)}
    assert assign_splits(distinct, RATIOS, seed=7) == assign_splits(distinct, RATIOS, seed=8)

    tied = {f"site{i}": 100 for i in range(20)}
    assert assign_splits(tied, RATIOS, seed=7) != assign_splits(tied, RATIOS, seed=8)
