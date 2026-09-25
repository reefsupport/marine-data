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
from marinedata.splitmap import resolve_splits
from marinedata.task import TaskKind

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


def test_cross_source_duplicate_lands_in_one_split(registry: Registry) -> None:
    """Byte-identical images ingested under two different sources share a `split_group`
    (`by="group"`) and must never straddle a split boundary — the leakage rule the
    site-keyed map cannot make when the duplicate crosses `source_id`."""
    shared_group = "seaview/12345"
    shared = [
        Sample(source_id="seaview-survey-imagery", key="s1", meta={"split_group": shared_group}),
        Sample(source_id="coralvqa", key="c1", meta={"split_group": shared_group}),
    ]
    filler = [
        Sample(source_id="other", key=f"o{i}", meta={"split_group": f"other/g{i}"})
        for i in range(20)
    ]
    samples = shared + filler
    schema = registry.label_schema("rs-benthic-v1")
    index = LabelIndex.from_samples(samples, schema)
    lineage = build_lineage([], registry.profile("research"))
    dataset = Dataset(samples=samples, label_index=index, lineage=lineage)

    dataset.split(by="group", tolerance=None)

    shared_positions = {0, 1}
    containing = [
        name for name, positions in dataset.splits.items() if shared_positions & set(positions)
    ]
    assert len(containing) == 1, "the shared group must not straddle two splits"
    assert shared_positions <= set(dataset.splits[containing[0]])


def test_self_supervised_with_map_adopts_map_ratios(registry: Registry, tmp_path: Path) -> None:
    """A self-supervised build's own 95/5 default must not raise against a map already
    written at 70/15/15 for the whole corpus — the builder adopts the map's ratios."""
    samples = [
        _sample("a", f"a{i}", "HC" if i % 3 else "SC", partition=f"site{i % 12}")
        for i in range(120)
    ]
    schema = registry.label_schema("rs-benthic-v1")
    index = LabelIndex.from_samples(samples, schema)
    lineage = build_lineage([], registry.profile("research"))
    dataset = Dataset(
        samples=samples, label_index=index, lineage=lineage, task_kind=TaskKind.SELF_SUPERVISED
    )

    path = tmp_path / "SPLIT_MAP.json"
    counts = {f"a/site{i}": 10 for i in range(12)}
    resolve_splits(
        path, counts, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0, by="site",
        now="2026-09-23T00:00:00Z",
    )

    dataset.split(by="site", split_map=path)  # must not raise

    # A shared map speaks train/val/test — the labelled-task vocabulary. A
    # self-supervised task must never inherit the map's `test` split: those groups are
    # excluded from `self.splits` altogether, and `val` demotes to `probe`.
    assert set(dataset.splits) == {"train", "probe"}
    assert dataset.splits["probe"], "the map's own val split must be reachable as probe"
    included = sum(len(positions) for positions in dataset.splits.values())
    assert included < len(samples), "the map's `test` groups must be excluded from `self.splits`"


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


# ── split proportions: a live correctness bug found 2026-08-18 ────────────


def _grouped(registry: Registry, groups: dict[str, int]) -> Dataset:
    samples = [
        Sample(
            source_id=key.split("/")[0],
            key=f"{key}/{i}",
            image=Path(f"/x/{key}/{i}.jpg"),
            labels={Axis.TAXON: LabelValue("HC")},
            supervised=frozenset({Axis.TAXON}),
            meta={"partition": key.split("/")[1]},
        )
        for key, count in groups.items()
        for i in range(count)
    ]
    schema = registry.label_schema("rs-benthic-v1")
    return Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )


def test_split_hits_requested_ratios_with_uneven_groups(registry: Registry) -> None:
    """Groups always differ in size; the split must still land near the target.

    The previous implementation sliced the group LIST proportionally, ignoring size.
    Measured on a realistic corpus it turned a requested 70/15/15 into 95.8/1.0/3.2 —
    and the empty-split guard did not fire, because no split was empty.
    """
    dataset = _grouped(registry, {f"rs/site{i}": 300 + i * 40 for i in range(14)})
    dataset.split(by="site", ratios={"train": 0.7, "val": 0.15, "test": 0.15})

    total = len(dataset)
    for name, want in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
        got = len(dataset.splits[name]) / total
        assert abs(got - want) < 0.10, f"{name}: wanted {want:.0%}, got {got:.1%}"


def test_split_raises_when_group_sizes_make_ratios_unreachable(registry: Registry) -> None:
    """One dominant group makes the target unreachable. Say so; do not pretend."""
    dataset = _grouped(
        registry,
        {"benthicnet-1m/global": 60000, "coralscapes/red-sea": 2075, "rs/tayrona": 658},
    )
    with pytest.raises(ValueError, match="could not hit the requested ratios"):
        dataset.split(by="site")


def test_split_tolerance_none_accepts_what_group_sizes_allow(registry: Registry) -> None:
    """Escape hatch: an unreachable target is a real constraint, not always an error."""
    dataset = _grouped(
        registry,
        {"benthicnet-1m/global": 60000, "coralscapes/red-sea": 2075, "rs/tayrona": 658},
    )
    dataset.split(by="site", tolerance=None)
    assert all(dataset.splits[name] for name in ("train", "val", "test"))


def test_split_is_deterministic_across_runs(registry: Registry) -> None:
    """Same corpus and seed must give byte-identical splits, or nothing is reproducible."""
    groups = {f"rs/s{i}": 100 * (i + 1) for i in range(12)}
    first = _grouped(registry, groups).split(by="site")
    second = _grouped(registry, groups).split(by="site")
    assert first.splits == second.splits


def test_split_never_divides_a_group(registry: Registry) -> None:
    """The whole point of grouping: a site lands entirely in one split."""
    dataset = _grouped(registry, {f"rs/site{i}": 200 + i * 30 for i in range(12)})
    dataset.split(by="site")
    seen: dict[str, str] = {}
    for name, positions in dataset.splits.items():
        for position in positions:
            sample = dataset.samples[position]
            group = f"{sample.source_id}/{sample.meta['partition']}"
            assert seen.setdefault(group, name) == name, f"{group} split across sets"


# ── unlabelled and mixed corpora (T1) ───────────────────────────────────────


@pytest.fixture
def mixed_roots(tmp_path: Path) -> dict[str, Path]:
    """A labelled source (coralscapes: real dense masks) and an unlabelled one
    (sweet-corals: bare images), both real registry entries, faked on disk.

    coralscapes' registry entry declares `loader.layout: staged-tree` (S62, D-O) —
    the shape its bucket copy actually is — so this fixture is a staged tree:
    `images/`+`labels/masks/`+`metadata.parquet`, not bare `images/`+`masks/` dirs.
    """
    from marinedata.tables import StagedImage, write_metadata_table

    labelled = tmp_path / "coralscapes"
    rows = []
    for i in range(4):
        stem = f"f{i}"
        (labelled / "images" / "default").mkdir(parents=True, exist_ok=True)
        (labelled / "labels" / "masks" / "default").mkdir(parents=True, exist_ok=True)
        (labelled / "images" / "default" / f"{stem}.jpg").write_bytes(b"\x89PNG\r\n\x1a\n")
        (labelled / "labels" / "masks" / "default" / f"{stem}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )
        rows.append(
            StagedImage(
                stem=stem,
                partition="default",
                upstream_path=f"orig/{stem}.jpg",
                upstream_split=None,
                width=10,
                height=10,
                # coralscapes declares an explicit split_group rule (pattern
                # `(?i)(site[0-9]+)` on `stem`) — StagedTreeLoader.validate() rejects
                # a null split_group when the source has one.
                split_group=f"coralscapes/site{i}",
            )
        )
    write_metadata_table(labelled / "metadata.parquet", rows)

    unlabelled = tmp_path / "sweet-corals"
    unlabelled.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (unlabelled / f"g{i}.jpg").write_bytes(b"\x89PNG\r\n\x1a\n")

    return {"coralscapes": labelled, "sweet-corals": unlabelled}


def test_mixed_build_combines_labelled_and_unlabelled_sources(
    registry: Registry, mixed_roots: dict[str, Path]
) -> None:
    """⭐ Pins the property the brief calls most likely to regress silently: a build
    spanning a labelled and an unlabelled source yields samples from both, and the
    unlabelled ones abstain rather than being dropped or corrupting the label index."""
    builder = DatasetBuilder(registry, profile="research", roots=mixed_roots)
    dataset = builder.build()

    by_source = {s.source_id for s in dataset.samples}
    assert by_source == {"coralscapes", "sweet-corals"}

    unlabelled_samples = [s for s in dataset.samples if s.source_id == "sweet-corals"]
    assert unlabelled_samples and all(s.supervised == frozenset() for s in unlabelled_samples)

    for sample in unlabelled_samples:
        encoded = dataset.label_index.encode(sample)
        assert all(value == IGNORE_INDEX for value in encoded.values())

    labelled_samples = [s for s in dataset.samples if s.source_id == "coralscapes"]
    assert labelled_samples and all(Axis.TAXON in s.supervised for s in labelled_samples)


def test_self_supervised_build_needs_no_crosswalk(
    registry: Registry, mixed_roots: dict[str, Path]
) -> None:
    """A `general-pretraining` build over the same roots must not demand a crosswalk —
    that used to be exactly what a supervised `task_id` required."""
    builder = DatasetBuilder(
        registry, profile="research", roots=mixed_roots, task_id="general-pretraining"
    )
    assert builder.projector is None
    dataset = builder.build()
    assert {s.source_id for s in dataset.samples} == {"coralscapes", "sweet-corals"}
    assert dataset.task_kind is not None and dataset.task_kind.value == "self_supervised"


def test_self_supervised_split_defaults_to_train_probe(
    registry: Registry, mixed_roots: dict[str, Path]
) -> None:
    """No val/test carved off a pretraining corpus by default — see
    SELF_SUPERVISED_DEFAULT_RATIOS for why 70/15/15 would be the wrong default here."""
    builder = DatasetBuilder(
        registry, profile="research", roots=mixed_roots, task_id="general-pretraining"
    )
    dataset = builder.build()
    # by="random" only to keep this fixture's 2 groups from tripping the "empty split"
    # guard that group-wise splitting rightly enforces elsewhere — the point here is the
    # *names and ratios* picked by default, not site-grouping fidelity.
    dataset.split(by="random")
    assert set(dataset.splits) == {"train", "probe"}
