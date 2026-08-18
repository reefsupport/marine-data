"""Builder, label-index and framework-adapter tests.

The invariants here are the ones that silently corrupt model development if broken:
group-wise splitting, ``IGNORE_INDEX`` on unsupervised axes, and refusing to mix
vocabularies.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.builder import Dataset, DatasetBuilder
from marinedata.labelindex import IGNORE_INDEX, LabelIndex
from marinedata.lineage import build_lineage
from marinedata.sample import LabelValue, Sample
from marinedata.schema import Axis

pd = pytest.importorskip("pandas")


def _sample(source_id: str, key: str, taxon: str | None, partition: str = "") -> Sample:
    labels = {Axis.TAXON: LabelValue(node_id=taxon)} if taxon else {}
    return Sample(
        source_id=source_id,
        key=key,
        image=Path(f"/nonexistent/{key}"),
        labels=labels,
        supervised=frozenset({Axis.TAXON}) if taxon else frozenset(),
        meta={"partition": partition} if partition else {},
    )


@pytest.fixture
def dataset(registry: Registry) -> Dataset:
    # Twelve sites: enough groups for a 70/15/15 grouped split to fill every bucket.
    samples = [
        _sample("a", f"a{i}", "HC" if i % 3 else "SC", partition=f"site{i % 12}")
        for i in range(120)
    ]
    schema = registry.label_schema("rs-benthic-v1")
    index = LabelIndex.from_samples(samples, schema)
    lineage = build_lineage([], registry.profile("research"))
    return Dataset(samples=samples, label_index=index, lineage=lineage)


# ── label index ───────────────────────────────────────────────────────────


def test_unsupervised_axis_encodes_to_ignore_index(registry: Registry) -> None:
    """The single most important behaviour: unsupervised axes must not train."""
    schema = registry.label_schema("rs-benthic-v1")
    sample = _sample("a", "k", "HC")
    index = LabelIndex.from_samples([sample], schema)
    encoded = index.encode(sample)
    assert encoded[Axis.TAXON] != IGNORE_INDEX
    # FORM was never supervised by this source.
    assert Axis.FORM not in index.axes or encoded.get(Axis.FORM) == IGNORE_INDEX


def test_ignore_index_matches_torch_default() -> None:
    """-100 is PyTorch's CrossEntropyLoss default; changing it breaks the masked loss."""
    assert IGNORE_INDEX == -100


def test_out_of_vocabulary_label_is_ignored_not_invented(registry: Registry) -> None:
    schema = registry.label_schema("rs-benthic-v1")
    index = LabelIndex.from_samples([_sample("a", "k", "HC")], schema)
    stranger = _sample("a", "k2", "NEVER_SEEN")
    assert index.encode(stranger)[Axis.TAXON] == IGNORE_INDEX


def test_min_count_drops_rare_classes(registry: Registry) -> None:
    schema = registry.label_schema("rs-benthic-v1")
    samples = [_sample("a", f"k{i}", "HC") for i in range(5)] + [_sample("a", "rare", "SC")]
    index = LabelIndex.from_samples(samples, schema, min_count=2)
    assert "HC" in index.axes[Axis.TAXON].classes
    assert "SC" not in index.axes[Axis.TAXON].classes


def test_class_weights_are_inverse_frequency(registry: Registry) -> None:
    schema = registry.label_schema("rs-benthic-v1")
    samples = [_sample("a", f"k{i}", "HC") for i in range(9)] + [_sample("a", "s", "SC")]
    index = LabelIndex.from_samples(samples, schema)
    weights = index.class_weights(samples)[Axis.TAXON]
    hc = index.axes[Axis.TAXON].index_of("HC")
    sc = index.axes[Axis.TAXON].index_of("SC")
    assert weights[sc] > weights[hc], "the rare class must be up-weighted"


# ── splitting ─────────────────────────────────────────────────────────────


def test_site_split_has_no_group_leakage(dataset: Dataset) -> None:
    """⭐ The default. Consecutive transect frames overlap; a leaked group inflates
    every metric, which is exactly how this field produces irreproducible numbers."""
    from marinedata.integrations.pandas import leakage_report

    dataset.split(by="site")
    assert len(leakage_report(dataset)) == 0


def test_random_split_leaks_and_that_is_why_it_is_not_the_default(dataset: Dataset) -> None:
    from marinedata.integrations.pandas import leakage_report

    dataset.split(by="random")
    assert len(leakage_report(dataset)) > 0, (
        "random splitting should scatter groups across splits — if this ever passes "
        "cleanly the fixture stopped having groups, not the leak stopped existing"
    )


def test_split_is_deterministic(dataset: Dataset) -> None:
    dataset.split(by="site", seed=7)
    first = {k: list(v) for k, v in dataset.splits.items()}
    dataset.split(by="site", seed=7)
    assert dataset.splits == first


def test_split_seed_changes_assignment(dataset: Dataset) -> None:
    dataset.split(by="site", seed=1)
    first = {k: list(v) for k, v in dataset.splits.items()}
    dataset.split(by="site", seed=2)
    assert dataset.splits != first


def test_ratios_must_sum_to_one(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        dataset.split(ratios={"train": 0.5, "test": 0.2})


def test_empty_split_raises_rather_than_returning_unusable_test_set(
    registry: Registry,
) -> None:
    samples = [_sample("a", f"k{i}", "HC", partition="only-site") for i in range(10)]
    schema = registry.label_schema("rs-benthic-v1")
    ds = Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )
    with pytest.raises(ValueError, match="empty splits"):
        ds.split(by="site")


def test_unknown_strategy_rejected(dataset: Dataset) -> None:
    with pytest.raises(ValueError, match="unknown split strategy"):
        dataset.split(by="whatever")


# ── builder guards ────────────────────────────────────────────────────────


def test_builder_refuses_unmapped_sources(registry: Registry, tmp_path: Path) -> None:
    """A source with no crosswalk would inject native labels into the canonical index.

    Uses `ozfish`, which emits SCALAR labels (bounding boxes with species names) and has
    no crosswalk authored yet. Scalar labels are the ones that can pollute a shared label
    index; dense-mask sources are exempt because their classes live in the raster, which
    the index never sees.
    """
    builder = DatasetBuilder(
        registry,
        profile="research",
        roots={"ozfish": tmp_path},
        schema_id="rs-benthic-v1",
    )
    with pytest.raises(ValueError, match="no crosswalk"):
        builder.build()


def test_builder_accepts_a_source_once_its_crosswalk_exists(registry: Registry) -> None:
    """The complement: a source WITH a crosswalk passes the mapping guard.

    Guards against the refusal above being unconditional — which would look identical
    in a red/green run while blocking every source.
    """
    source = registry.source("reef-support-benthic-own")
    assert source.loader is not None
    assert source.loader.crosswalk_id == "reef-support-labelbox"
    harmonizer = registry.harmonizer_for("reef-support-benthic-own")
    assert harmonizer is not None
    mapped = harmonizer.map_label("Soft Coral")
    assert mapped.labels[Axis.TAXON].node_id == "SC"


def test_builder_gate_runs_before_any_read(registry: Registry, tmp_path: Path) -> None:
    """Disallowed sources never reach the loader — the path does not even have to exist."""
    builder = DatasetBuilder(
        registry, profile="ship-commercial", roots={"coralscop-masks-rs": tmp_path}
    )
    with pytest.raises(ValueError, match="No permitted sources"):
        builder.build()


# ── pandas adapter ────────────────────────────────────────────────────────


def test_to_pandas_shape_and_columns(dataset: Dataset) -> None:
    dataset.split(by="site")
    frame = dataset.to_pandas()
    assert len(frame) == len(dataset)
    for column in ("source_id", "key", "split", "licence_tier", "taxon", "taxon_index"):
        assert column in frame.columns
    assert frame["taxon_supervised"].all()


def test_to_pandas_split_subset(dataset: Dataset) -> None:
    dataset.split(by="site")
    train = dataset.to_pandas(split="train")
    assert len(train) == len(dataset.splits["train"])
    assert set(train["split"]) == {"train"}


def test_unused_axes_are_dropped_from_the_frame(dataset: Dataset) -> None:
    frame = dataset.to_pandas()
    assert "condition" not in frame.columns


# ── statistics ────────────────────────────────────────────────────────────


def test_class_counts_and_coverage(dataset: Dataset) -> None:
    counts = dataset.class_counts()
    assert counts["HC"] > counts["SC"]
    assert dataset.supervision_coverage()["taxon"] == 1.0


def test_summary_is_informative(dataset: Dataset) -> None:
    dataset.split(by="site")
    text = dataset.summary()
    assert "samples" in text and "splits" in text and "LabelIndex" in text
