"""Tasks — turning many differently-labelled datasets into one training set.

The registry can already map each source's native vocabulary onto a canonical schema.
That is necessary and not sufficient. Sources annotate at **different depths**: ReefNet
labels genus, our own masks label L2 (`HC`/`SC`), CoralNet's functional groups sit at L1.
Harmonised but unprojected, those become four separate classes for what is ecologically
two, and a model trained on the union learns that `HC` and `HC_ORBICELLA` are different
things.

A :class:`TaskSpec` fixes the **purpose**: one axis, one target vocabulary, one head. Every
label is then projected onto it:

    HC_ORBICELLA  (genus, finer)   → HC        rolled up via the hierarchy
    HC            (exactly target) → HC        kept
    SC            (exactly target) → SC        kept
    BIOTIC        (L1, coarser)    → ABSTAIN   cannot disambiguate HC from SC

That last case is the one that matters. **A label coarser than the target is not a label —
it is an abstention.** "Biotic" tells you nothing about hard versus soft coral, and forcing
it into either fabricates supervision. It becomes ``IGNORE_INDEX``, exactly as an
unmappable crosswalk edge does, so the sample still contributes its image to the batch
while contributing no gradient to that head.

The payoff is that adding a dataset stops being a schema negotiation. Declare what it
speaks, write its crosswalk once, and every task that can use it picks it up — at whatever
depth it happens to annotate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, model_validator

from .enums import Modality
from .schema import Axis, LabelSchema


class TaskKind(str, Enum):
    """What a :class:`TaskSpec` is asking for.

    Kept as one type rather than a separate ``PretrainSpec`` — see the note on
    :class:`TaskSpec` for why.
    """

    SUPERVISED = "supervised"
    """A fixed target vocabulary. Requires ``schema_id``, ``axis`` and ``classes``."""

    SELF_SUPERVISED = "self_supervised"
    """A pretraining corpus. No vocabulary to project onto — every permitted source
    matching ``modalities`` (or any modality, if unset) qualifies, labelled or not."""


class Coarser(str, Enum):
    """What to do with a label above the target level in the hierarchy."""

    ABSTAIN = "abstain"
    """Treat as unsupervised. The honest default: the source genuinely did not record
    enough detail to answer this task's question."""

    ERROR = "error"
    """Raise. Use when a corpus is supposed to be fully annotated at the target depth and
    a coarser label means an upstream mistake."""


@dataclass(frozen=True)
class Projection:
    """Where one source label landed, and how."""

    source_node: str
    target_class: str | None
    reason: str

    @property
    def supervised(self) -> bool:
        return self.target_class is not None


class TaskSpec(BaseModel):
    """A training objective.

    Two shapes, one type. ``SUPERVISED`` (the default, and every task declared before
    this field existed) fixes a target vocabulary: one axis, one fixed set of classes,
    one head. ``SELF_SUPERVISED`` fixes nothing to project onto — it names a pretraining
    corpus by ``modalities`` instead, and any permitted source matching qualifies,
    labelled or not.

    Args:
        kind: ``SUPERVISED`` (default) or ``SELF_SUPERVISED``.
        classes: the target vocabulary, in the canonical schema. Required and non-empty
            for a supervised task; must be empty for a self-supervised one. Every node
            that is one of these, or a descendant of one, becomes supervision.
            Everything else abstains.
        schema_id, axis: required for a supervised task; must be unset for a
            self-supervised one, since there is no vocabulary to validate against.
        modalities: for a self-supervised task, restricts which source modalities
            qualify (empty means any). Ignored for a supervised task — its
            eligibility already comes from the crosswalk.
        coarser: policy for labels above the target level. Supervised only.

    **Why one type instead of a separate ``PretrainSpec``.** Every entry point that
    matters — ``registry.task(id)``, ``registry.tasks``, ``DatasetBuilder(task_id=...)``,
    ``registry.sources_for_task()`` — is already keyed on a ``TaskSpec`` id. A parallel
    spec type would need its own registry section, its own loader, and its own builder
    entry point for what is fundamentally the same question ("what corpus does this
    training run see"), and it would mean contributors learn two spec shapes instead of
    one with an optional vocabulary. The asymmetry the self-supervised path actually
    needs is narrow — no schema/axis/classes to validate, no projector to build — and is
    cheaper to express as "these fields are optional for this one kind" than as a second
    type threaded through every call site above.

    The class list is explicit rather than derived from a "level", because real
    vocabularies are not flat slices of a tree: a task may want `HC` and `SC` at L2 but
    keep `MIL` separate, or split `HC` into genera only where the data supports it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: TaskKind = TaskKind.SUPERVISED
    schema_id: str | None = None
    axis: Axis | None = None
    classes: tuple[str, ...] = ()
    modalities: tuple[Modality, ...] = ()
    description: str = ""
    coarser: Coarser = Coarser.ABSTAIN

    @model_validator(mode="after")
    def _shape_matches_kind(self) -> TaskSpec:
        if self.kind is TaskKind.SUPERVISED:
            missing = [
                name
                for name, value in (("schema_id", self.schema_id), ("axis", self.axis))
                if value is None
            ]
            if missing:
                raise ValueError(f"task {self.id}: supervised task missing {missing}")
            if not self.classes:
                raise ValueError(f"task {self.id}: supervised task declares no classes")
            if self.modalities:
                raise ValueError(
                    f"task {self.id}: 'modalities' only applies to self-supervised "
                    f"tasks — a supervised task's eligibility comes from its crosswalk"
                )
        else:
            set_anyway = [
                name
                for name, value in (
                    ("schema_id", self.schema_id),
                    ("axis", self.axis),
                    ("classes", self.classes),
                )
                if value
            ]
            if set_anyway:
                raise ValueError(
                    f"task {self.id}: self-supervised task must not set {set_anyway} — "
                    f"there is no vocabulary to project onto"
                )
        return self

    def validate_against(self, schema: LabelSchema) -> None:
        """Check the target vocabulary against the schema. Raises on any problem.

        Supervised tasks only — a self-supervised task has no schema to check.
        """
        if self.kind is not TaskKind.SUPERVISED:
            raise ValueError(f"task {self.id}: self-supervised, has no schema to validate")
        if schema.id != self.schema_id:
            raise ValueError(
                f"task {self.id}: expects schema '{self.schema_id}', got '{schema.id}'"
            )
        if not self.classes:
            raise ValueError(f"task {self.id}: no target classes declared")

        missing = [c for c in self.classes if schema.node(c) is None]
        if missing:
            raise ValueError(f"task {self.id}: classes absent from {schema.id}: {missing}")

        wrong_axis = [
            c for c in self.classes if (node := schema.node(c)) and node.axis is not self.axis
        ]
        if wrong_axis:
            raise ValueError(
                f"task {self.id}: classes not on axis '{self.axis.value}': {wrong_axis}"
            )

        # Nested targets make projection ambiguous: HC_ORBICELLA would roll up to HC, but
        # HC is also a target, so the same pixel has two valid answers.
        for candidate in self.classes:
            overlap = [a for a in schema.ancestors(candidate) if a in self.classes]
            if overlap:
                raise ValueError(
                    f"task {self.id}: '{candidate}' is a descendant of target class(es) "
                    f"{overlap}. Target classes must be mutually exclusive — a label "
                    f"cannot belong to two of them."
                )


class TaskProjector:
    """Projects canonical labels onto a task's vocabulary."""

    def __init__(self, task: TaskSpec, schema: LabelSchema) -> None:
        task.validate_against(schema)
        self.task = task
        self.schema = schema
        self._targets = set(task.classes)
        self._cache: dict[str, Projection] = {}

    def project(self, node_id: str) -> Projection:
        """Map one canonical node onto the task vocabulary."""
        if node_id in self._cache:
            return self._cache[node_id]

        if node_id in self._targets:
            result = Projection(node_id, node_id, "exact")
        elif self.schema.node(node_id) is None:
            result = Projection(node_id, None, "not in schema")
        else:
            ancestors = self.schema.ancestors(node_id)
            hit = next((a for a in ancestors if a in self._targets), None)
            if hit is not None:
                result = Projection(node_id, hit, f"rolled up from {node_id}")
            elif self._is_coarser(node_id):
                if self.task.coarser is Coarser.ERROR:
                    raise ValueError(
                        f"task {self.task.id}: '{node_id}' is coarser than the target "
                        f"vocabulary and coarser=error. It cannot be resolved to one of "
                        f"{sorted(self._targets)} without inventing detail."
                    )
                result = Projection(
                    node_id, None, "coarser than target — abstains rather than guessing"
                )
            else:
                result = Projection(node_id, None, "outside the task vocabulary")

        self._cache[node_id] = result
        return result

    def _is_coarser(self, node_id: str) -> bool:
        """True when the node is an ancestor of some target class.

        Distinguished from "unrelated" because the two mean different things: a coarser
        label is a source that looked and did not record enough detail; an unrelated one
        is a source describing something else entirely. Both abstain, but only the first
        is worth reporting as lost supervision.
        """
        return any(node_id in self.schema.ancestors(target) for target in self._targets)

    def coverage(self, counts: dict[str, int]) -> TaskCoverage:
        """How much of a corpus this task can actually use, weighted by instances."""
        supervised = 0
        coarser = 0
        outside = 0
        per_class: dict[str, int] = dict.fromkeys(self.task.classes, 0)

        for node_id, count in counts.items():
            projection = self.project(node_id)
            if projection.target_class is not None:
                supervised += count
                per_class[projection.target_class] += count
            elif "coarser" in projection.reason:
                coarser += count
            else:
                outside += count

        return TaskCoverage(
            task_id=self.task.id,
            supervised=supervised,
            coarser=coarser,
            outside=outside,
            per_class=per_class,
        )


@dataclass(frozen=True)
class TaskCoverage:
    """What fraction of a corpus a task can supervise, and what it loses."""

    task_id: str
    supervised: int
    coarser: int
    outside: int
    per_class: dict[str, int]

    @property
    def total(self) -> int:
        return self.supervised + self.coarser + self.outside

    @property
    def rate(self) -> float:
        return self.supervised / self.total if self.total else 0.0

    def summary(self) -> str:
        lines = [
            f"task {self.task_id}: {self.rate:.1%} of {self.total:,} labels supervised",
            f"  coarser than target (abstain) {self.coarser:>10,}",
            f"  outside the vocabulary        {self.outside:>10,}",
        ]
        empty = [c for c, n in self.per_class.items() if n == 0]
        for name, count in sorted(self.per_class.items(), key=lambda kv: -kv[1]):
            share = count / max(self.supervised, 1)
            lines.append(f"    {name:<16} {count:>10,}  {share:>6.1%}")
        if empty:
            lines.append(
                f"  ⚠ {len(empty)} class(es) with NO examples: {', '.join(empty)} — "
                f"a head cannot learn a class it never sees"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class SourceFit:
    """How well one source can serve one task.

    Eligibility and usefulness are different questions. A source that supervises the
    right axis through a valid crosswalk is *eligible*; whether any of its labels
    actually reach the task's classes is a separate matter. ReefNet is eligible for a
    coarse benthic task and contributes richly; Coralscapes is eligible for a Caribbean
    genus task and contributes nothing, because its genera are Indo-Pacific.

    Reporting only eligibility would hand back six sources for a task that three of them
    cannot supply a single example to.
    """

    source_id: str
    reachable: tuple[str, ...]
    """Task classes this source can actually produce."""

    abstaining: tuple[str, ...]
    """Native labels that project to nothing — coarser than the target, or outside it."""

    unsupervised: bool = False
    """Set for a self-supervised task: there is no vocabulary to reach, so the source
    contributes its images with no labels rather than some of ``reachable``."""

    @property
    def contributes(self) -> bool:
        return bool(self.reachable) or self.unsupervised

    def line(self) -> str:
        if self.unsupervised:
            return f"  ✓ {self.source_id:<28} unlabelled — contributes images, no supervision"
        mark = "✓" if self.contributes else "·"
        if not self.contributes:
            return f"  {mark} {self.source_id:<28} contributes nothing to this task"
        shown = ", ".join(self.reachable[:6])
        more = f" +{len(self.reachable) - 6}" if len(self.reachable) > 6 else ""
        return f"  {mark} {self.source_id:<28} {len(self.reachable):>2} class(es): {shown}{more}"


def fit_source(projector: TaskProjector, crosswalk, source_id: str) -> SourceFit:
    """Which of a task's classes a source can reach, via its crosswalk.

    Static analysis over the crosswalk edges — no data required, so this answers "is it
    worth fetching this dataset for this task?" before anything is downloaded.
    """
    reachable: set[str] = set()
    abstaining: list[str] = []
    axis = projector.task.axis

    for edge in crosswalk.edges:
        node_id = edge.targets.get(axis)
        if node_id is None:
            abstaining.append(edge.source_label)
            continue
        target = projector.project(node_id).target_class
        if target is None:
            abstaining.append(edge.source_label)
        else:
            reachable.add(target)

    return SourceFit(
        source_id=source_id,
        reachable=tuple(sorted(reachable)),
        abstaining=tuple(sorted(abstaining)),
    )
