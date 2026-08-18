"""Task projection — combining datasets that do not share a label vocabulary.

The problem this solves: sources annotate at different depths. ReefNet labels genus, our
own masks label L2, CoralNet's functional groups sit at L1. Harmonised but unprojected,
those become separate classes for what is ecologically one thing, and a model trained on
the union learns that `HC` and `HC_ORBICELLA` differ.

The rule that carries the design: **a label coarser than the target is not a label, it is
an abstention.** `BIOTIC` cannot tell you hard from soft coral, and forcing it into either
fabricates supervision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.labelindex import IGNORE_INDEX, LabelIndex
from marinedata.sample import LabelValue, Sample
from marinedata.schema import Axis
from marinedata.task import Coarser, TaskProjector, TaskSpec

COARSE = TaskSpec(
    id="t", schema_id="rs-benthic-v1", axis=Axis.TAXON, classes=("HC", "SC", "ABIOTIC")
)


@pytest.fixture
def projector(registry: Registry) -> TaskProjector:
    return TaskProjector(COARSE, registry.label_schema("rs-benthic-v1"))


def test_finer_labels_roll_up(projector: TaskProjector) -> None:
    """A genus becomes its L2 parent — this is what unifies sources of differing depth."""
    result = projector.project("HC_ORBICELLA")
    assert result.target_class == "HC"
    assert "rolled up" in result.reason


def test_exact_labels_pass_through(projector: TaskProjector) -> None:
    assert projector.project("HC").target_class == "HC"
    assert projector.project("HC").reason == "exact"


def test_coarser_labels_abstain_rather_than_guess(projector: TaskProjector) -> None:
    """⭐ The rule the whole module exists for.

    `BIOTIC` is an ancestor of both HC and SC. Assigning it to either would invent
    supervision the source never provided.
    """
    result = projector.project("BIOTIC")
    assert result.target_class is None
    assert "coarser" in result.reason
    assert not result.supervised


def test_unrelated_labels_abstain_with_a_different_reason(projector: TaskProjector) -> None:
    """Coarser and unrelated both abstain, but only one is lost supervision worth reporting."""
    result = projector.project("TL")  # transect line
    assert result.target_class is None
    assert "outside" in result.reason


def test_coarser_can_raise_when_a_corpus_should_be_fully_annotated(
    registry: Registry,
) -> None:
    strict = COARSE.model_copy(update={"coarser": Coarser.ERROR})
    projector = TaskProjector(strict, registry.label_schema("rs-benthic-v1"))
    with pytest.raises(ValueError, match="coarser than the target"):
        projector.project("BIOTIC")


def test_nested_target_classes_are_rejected(registry: Registry) -> None:
    """HC_ORBICELLA rolls up to HC, so having both as targets is ambiguous."""
    bad = TaskSpec(
        id="bad", schema_id="rs-benthic-v1", axis=Axis.TAXON, classes=("HC", "HC_ORBICELLA")
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        TaskProjector(bad, registry.label_schema("rs-benthic-v1"))


def test_classes_must_exist_and_sit_on_the_task_axis(registry: Registry) -> None:
    schema = registry.label_schema("rs-benthic-v1")
    with pytest.raises(ValueError, match="absent from"):
        TaskProjector(
            TaskSpec(id="x", schema_id="rs-benthic-v1", axis=Axis.TAXON, classes=("NOPE",)),
            schema,
        )
    with pytest.raises(ValueError, match="not on axis"):
        TaskProjector(
            TaskSpec(id="x", schema_id="rs-benthic-v1", axis=Axis.TAXON, classes=("BLEACHED",)),
            schema,
        )


def test_four_sources_at_four_depths_unify(registry: Registry, projector: TaskProjector) -> None:
    """The end-to-end case. Before projection this produced four classes for two things."""
    index = LabelIndex.for_task(COARSE, registry.label_schema("rs-benthic-v1"))

    def encode(node_id: str) -> int:
        sample = Sample(
            source_id="s",
            key="k",
            image=Path("/x.jpg"),
            labels={Axis.TAXON: LabelValue(node_id)},
            supervised=frozenset({Axis.TAXON}),
        )
        return index.encode(sample, projector=projector)[Axis.TAXON]

    assert encode("HC_ORBICELLA") == encode("HC") == encode("HC_ACROPORA")
    assert encode("SC") != encode("HC")
    assert encode("SD") == encode("ABIOTIC")
    assert encode("BIOTIC") == IGNORE_INDEX


def test_head_width_is_fixed_by_the_task_not_the_data(registry: Registry) -> None:
    """Two runs over different corpora must produce comparable checkpoints."""
    index = LabelIndex.for_task(COARSE, registry.label_schema("rs-benthic-v1"))
    assert index.num_classes()[Axis.TAXON] == 3
    assert index.axes[Axis.TAXON].classes == ("ABIOTIC", "HC", "SC")


def test_coverage_reports_what_a_task_loses(projector: TaskProjector) -> None:
    coverage = projector.coverage({"HC_ORBICELLA": 60, "SC": 20, "BIOTIC": 15, "TL": 5})
    assert coverage.supervised == 80
    assert coverage.coarser == 15
    assert coverage.outside == 5
    assert coverage.rate == pytest.approx(0.8)
    assert coverage.per_class["HC"] == 60


def test_coverage_flags_classes_with_no_examples(projector: TaskProjector) -> None:
    """A head cannot learn a class it never sees — say so rather than training anyway."""
    coverage = projector.coverage({"HC": 100})
    assert "NO examples" in coverage.summary()
    assert "SC" in coverage.summary()


# ── the registry's task catalogue ────────────────────────────────────────


def test_registry_tasks_all_validate(registry: Registry) -> None:
    """Every declared task must be coherent with its schema — checked at load."""
    assert registry.tasks
    for task in registry.tasks:
        task.validate_against(registry.label_schema(task.schema_id))


def test_eligible_is_not_the_same_as_contributing(registry: Registry) -> None:
    """⭐ Reporting eligibility alone would overstate support badly.

    Six sources are eligible for the Caribbean genus task; one contributes, and only one
    class. That is worth knowing before planning a training run around it.
    """
    everything = registry.sources_for_task("coral-genus-caribbean", contributing_only=False)
    contributing = [fit for _, fit in everything if fit.contributes]
    assert len(everything) > len(contributing), "the distinction should be visible here"
    assert contributing, "at least one source should reach a genus class"


def test_contributing_sources_are_reported_without_duplicates(registry: Registry) -> None:
    """Fits are keyed on the SOURCE, not its crosswalk — sources sharing one are distinct."""
    pairs = registry.sources_for_task("benthic-coarse")
    ids = [fit.source_id for _, fit in pairs]
    assert len(ids) == len(set(ids)), f"duplicate source ids: {ids}"
    assert all(source.id == fit.source_id for source, fit in pairs)
