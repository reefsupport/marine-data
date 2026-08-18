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

from pydantic import BaseModel, ConfigDict

from .schema import Axis, LabelSchema


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
    """A training objective: one axis, one fixed vocabulary, one head.

    Args:
        classes: the target vocabulary, in the canonical schema. Every node that is one
            of these, or a descendant of one, becomes supervision. Everything else
            abstains.
        coarser: policy for labels above the target level.

    The class list is explicit rather than derived from a "level", because real
    vocabularies are not flat slices of a tree: a task may want `HC` and `SC` at L2 but
    keep `MIL` separate, or split `HC` into genera only where the data supports it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    schema_id: str
    axis: Axis
    classes: tuple[str, ...]
    description: str = ""
    coarser: Coarser = Coarser.ABSTAIN

    def validate_against(self, schema: LabelSchema) -> None:
        """Check the target vocabulary against the schema. Raises on any problem."""
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

    @property
    def contributes(self) -> bool:
        return bool(self.reachable)

    def line(self) -> str:
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
