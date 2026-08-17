"""Label schemas and crosswalks — the harmonisation layer.

The field's datasets use a dozen incompatible label schemes. Training on more than one
at a time requires mapping them onto a common target, and doing that *honestly* requires
recording what each mapping loses.

Two ideas carry the design:

**Axes, not flat classes.** Reef labels factor into independent questions — what is it,
what shape is it, what condition is it in. Flattening them into a single class list is
what produced Coralscapes-39, a partially-enumerated cross-product with no soft-coral
row. Axes keep the questions separable and let a source supervise only what it knows.

**Mappings are lossy and say so.** ``catami:Substrate/Unconsolidated/Sand`` → ``SD`` is
exact. ``coralscapes:other coral alive`` → ``HC`` drops the (absent) growth form. A
crosswalk edge that cannot record the difference will silently manufacture false
precision.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Axis(str, Enum):
    """Independent label dimensions. A source supervises a subset."""

    TAXON = "taxon"
    """What the thing is — the substrate/biota hierarchy."""

    FORM = "form"
    """Morphology / growth form. Defined only under some taxon branches."""

    CONDITION = "condition"
    """Health state: healthy, pale, bleached, diseased, dead."""

    SIZE = "size"
    """Length or size class — fish, colonies."""

    COUNT = "count"
    """Abundance."""


class Fidelity(str, Enum):
    """How faithfully a crosswalk edge maps."""

    EXACT = "exact"
    """Source and target mean the same thing."""

    COARSENED = "coarsened"
    """Target is a strict ancestor — detail is lost but nothing is invented."""

    APPROXIMATE = "approximate"
    """Best-effort. The concepts overlap imperfectly; review before relying on it."""

    UNMAPPABLE = "unmappable"
    """No target exists. The source label is dropped, and the pixel/point becomes
    unsupervised on that axis rather than being forced into a wrong class."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LabelNode(_Frozen):
    """One node in a hierarchical label schema."""

    id: str
    name: str
    parent: str | None = None
    axis: Axis = Axis.TAXON
    aliases: tuple[str, ...] = ()
    worms_aphia_id: int | None = None
    notes: str | None = None


class LabelSchema(_Frozen):
    """A named, hierarchical label vocabulary."""

    id: str
    name: str
    description: str = ""
    axes: tuple[Axis, ...] = (Axis.TAXON,)
    nodes: tuple[LabelNode, ...] = ()
    canonical: bool = False
    """True for our target schemas; crosswalks map *into* these."""

    @model_validator(mode="after")
    def _parents_exist(self) -> LabelSchema:
        ids = {n.id for n in self.nodes}
        for node in self.nodes:
            if node.parent is not None and node.parent not in ids:
                raise ValueError(
                    f"schema {self.id}: node '{node.id}' has unknown parent '{node.parent}'"
                )
        return self

    def node(self, node_id: str) -> LabelNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def ancestors(self, node_id: str) -> tuple[str, ...]:
        """Ancestor chain, nearest first. Used to coarsen a label when a model abstains."""
        out: list[str] = []
        seen: set[str] = set()
        current = self.node(node_id)
        while current is not None and current.parent is not None:
            if current.parent in seen:  # defensive: a cycle would otherwise hang
                raise ValueError(f"schema {self.id}: cycle at '{current.parent}'")
            seen.add(current.parent)
            out.append(current.parent)
            current = self.node(current.parent)
        return tuple(out)

    def leaves(self) -> tuple[LabelNode, ...]:
        parents = {n.parent for n in self.nodes if n.parent}
        return tuple(n for n in self.nodes if n.id not in parents)


class CrosswalkEdge(_Frozen):
    """One source-label → canonical-label mapping."""

    source_label: str
    targets: dict[Axis, str] = Field(default_factory=dict)
    """Canonical node id per axis. A single source label can populate several axes —
    ``coralscapes:massive/meandering bleached`` sets taxon, form *and* condition."""

    fidelity: Fidelity = Fidelity.EXACT
    note: str | None = None

    @model_validator(mode="after")
    def _unmappable_has_no_targets(self) -> CrosswalkEdge:
        if self.fidelity is Fidelity.UNMAPPABLE and self.targets:
            raise ValueError(
                f"edge '{self.source_label}': fidelity=unmappable but targets were given"
            )
        if self.fidelity is not Fidelity.UNMAPPABLE and not self.targets:
            raise ValueError(
                f"edge '{self.source_label}': no targets — use fidelity=unmappable "
                f"if the label genuinely has no canonical equivalent"
            )
        return self


class Crosswalk(_Frozen):
    """A complete mapping from one schema into a canonical schema."""

    id: str
    source_schema: str
    target_schema: str
    description: str = ""
    edges: tuple[CrosswalkEdge, ...] = ()

    def edge(self, source_label: str) -> CrosswalkEdge | None:
        return next((e for e in self.edges if e.source_label == source_label), None)

    @property
    def coverage(self) -> dict[Fidelity, int]:
        counts = dict.fromkeys(Fidelity, 0)
        for edge in self.edges:
            counts[edge.fidelity] += 1
        return counts
