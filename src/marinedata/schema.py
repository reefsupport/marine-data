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

from datetime import date
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

    TROPHIC = "trophic"
    """Feeding role — herbivore, piscivore, corallivore.

    Independent of taxonomy on purpose: parrotfish and surgeonfish are both herbivores
    in different families, groupers and morays both piscivores across different orders.
    Trophic role cross-cuts phylogeny, and reef monitoring cares about the role.
    """


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
    notes: str | None = None

    # ── taxonomic anchor ──────────────────────────────────────────────
    # AphiaID is an ATTRIBUTE, never the node id. Most of this library is
    # substrate, morphology, condition and equipment, none of which has one —
    # keying on AphiaID would orphan the majority of nodes.
    #
    # OBIS uses the WoRMS AphiaID as its `taxonID`, so this single integer joins
    # us to OBIS occurrences, WoRMS classification and any other WoRMS-backed
    # dataset, at every rank.

    worms_aphia_id: int | None = None
    """Accepted AphiaID."""

    worms_aphia_id_asserted: int | None = None
    """The AphiaID a source originally asserted, when it differs from the accepted one.

    Taxa get reassigned: *Montastraea annularis* (207479) is now
    *Orbicella annularis* (758260). Recording both means two datasets labelled a decade
    apart unify automatically — WoRMS does the work, we just keep the trail.
    """

    # The four fields below are FROZEN AT RESOLVE TIME so that drift is detectable.
    # Without them a wrong id is invisible: three of the first eight AphiaIDs entered
    # here by hand were wrong, one pointing at a red alga and one at a bryozoan, and
    # nothing in the system could tell. A bare integer cannot be reviewed.
    worms_scientificname: str | None = None
    worms_rank: str | None = None
    worms_status: str | None = None
    worms_checked_on: date | None = None

    @model_validator(mode="after")
    def _worms_fields_need_an_id(self) -> LabelNode:
        frozen = (self.worms_scientificname, self.worms_rank, self.worms_status)
        if any(f is not None for f in frozen) and self.worms_aphia_id is None:
            raise ValueError(f"node {self.id}: carries frozen WoRMS fields but no worms_aphia_id")
        return self

    @property
    def is_taxon(self) -> bool:
        """True when this node denotes an organism rather than substrate or equipment."""
        return self.worms_aphia_id is not None


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
    def _validate_structure(self) -> LabelSchema:
        """Validate the whole node set at construction.

        An earlier version checked only that parents existed, which let four distinct
        classes of malformed schema load clean — all four verified against real code:

        * duplicate node ids (the second becomes silently unreachable via ``node()``)
        * self-parent and multi-node cycles, caught only later and only if a caller
          happened to call :meth:`ancestors`
        * cross-axis parents — a ``taxon`` node could descend from a ``condition`` node,
          which makes "ancestor" meaningless and would silently corrupt any rollup
        * nodes on an axis the schema does not declare

        A cycle also makes :meth:`leaves` return an empty tuple rather than raising,
        which is the worst outcome available: a schema that reports no leaves looks like
        a schema with nothing in it.
        """
        by_id: dict[str, LabelNode] = {}
        for node in self.nodes:
            if node.id in by_id:
                raise ValueError(f"schema {self.id}: duplicate node id '{node.id}'")
            by_id[node.id] = node

        for node in self.nodes:
            if node.axis not in self.axes:
                raise ValueError(
                    f"schema {self.id}: node '{node.id}' is on axis '{node.axis.value}' "
                    f"which the schema does not declare "
                    f"(declared: {[a.value for a in self.axes]})"
                )
            if node.parent is None:
                continue
            parent = by_id.get(node.parent)
            if parent is None:
                raise ValueError(
                    f"schema {self.id}: node '{node.id}' has unknown parent '{node.parent}'"
                )
            if parent.axis is not node.axis:
                raise ValueError(
                    f"schema {self.id}: node '{node.id}' ({node.axis.value}) has a "
                    f"cross-axis parent '{parent.id}' ({parent.axis.value}). Ancestry is "
                    f"only meaningful within one axis; use a crosswalk to relate axes."
                )

        self._assert_acyclic(by_id)
        return self

    def _assert_acyclic(self, by_id: dict[str, LabelNode]) -> None:
        """Iterative DFS with separate visited and on-stack sets.

        Separate sets matter: a shared one cannot distinguish a cycle from a node
        legitimately reached twice, which is precisely the bug that made a multi-parent
        model unworkable here.
        """
        visited: set[str] = set()
        for start in by_id:
            if start in visited:
                continue
            path: list[str] = []
            on_stack: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in on_stack:
                    cycle = " -> ".join([*path[path.index(current) :], current])
                    raise ValueError(f"schema {self.id}: cycle in parent chain: {cycle}")
                if current in visited:
                    break
                visited.add(current)
                on_stack.add(current)
                path.append(current)
                node = by_id.get(current)
                current = node.parent if node else None

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
