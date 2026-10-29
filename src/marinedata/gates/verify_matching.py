"""Registry-level proof that every source's native labels resolve into ``rs-benthic-v1``.

No image data is read: the proof runs on ``registry/`` alone (schemas, crosswalks, harvested
vocabularies, ``label-schemes``). It is the offline counterpart of the G9 ``labelmap`` gate, which
checks built tables. This module holds the per-source checks; ``verify_matching_cross`` holds the
cross-source checks and assembles the report; ``scripts/verify_label_matching.py`` is the CLI.

For every unit (a registry source that declares a ``schema_id`` and/or ``crosswalk_id``, plus every
harvested vocabulary and every crosswalk no source points at) it checks

1 coverage    each native label has exactly one crosswalk edge or an explicit ignore (an
              ``unmappable`` edge, which must carry a note); none missing, none duplicated
2 resolution  each edge target is a node of the target schema on the declared axis and resolves to
              one node key (``labelmap.resolve_node``; a bare id in two namespaces is ambiguous)
3 fidelity    unmappable edges carry a note; an exact edge on an inner node matches the native rank

Native labels come from the harvested vocabulary and the published ids when they exist, else from
the schema nodes of the crosswalk's source schema. Schema nodes no edge maps in any crosswalk of
that schema are ``declared_unmapped`` (graded against ``BASELINE_DECLARED_GAPS``).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from ..registry import Registry, _default_root
from ..schema import Axis, Crosswalk, CrosswalkEdge, Fidelity, LabelSchema
from ..taxonomy import load_meta, read_vocab_rows
from .labelmap import node_namespaces, resolve_node

TARGET = "rs-benthic-v1"
LOSSY = frozenset({Fidelity.COARSENED, Fidelity.APPROXIMATE, Fidelity.NARROWER})
CAP = 25
COUNT_KEYS = (
    "labels", "edges", "exact", "coarsened", "approximate", "narrower", "ignored",
    "unresolved", "ambiguous", "missing", "duplicate", "declared_unmapped", "unused_edges",
    "lossy_without_note",
)  # fmt: skip

# Findings that predate this verifier. The registry owner fixes them; the verifier fails when a
# count rises above its baseline, so none can grow unnoticed. Lower an entry when it is fixed.
# Lossy edges without a note that predate this check (frozen v1.0/v1.1 crosswalks); may only fall.
BASELINE_UNNOTED: dict[str, int] = {
    "agrra-benthic": 3,
    "catami-1.4": 20,
    "coralnet-labelset": 2,
    "coralnet-noaa-pifsc": 4,
    "coralscapes-39": 6,
    "deepseagrass": 1,
    "mermaid-attributes": 2,
    "seaclear-marine-debris": 28,
}
BASELINE_DECLARED_GAPS: dict[str, int] = {}  # source id -> unmapped schema nodes


@dataclass(frozen=True)
class Label:
    """One native label: a display name plus every key an edge may use for it (name, native id)."""

    name: str
    keys: frozenset[str]


@dataclass(frozen=True)
class Unit:
    """A verification unit: a registry source, a vocabulary-only source or a bare crosswalk."""

    id: str
    origin: str  # registry | vocab | crosswalk
    schema_id: str | None
    crosswalk_id: str | None
    via: str  # how the crosswalk was found: loader | vocab | schema | crosswalk
    observed: tuple[Label, ...]  # harvested vocabulary and published ids
    declared: tuple[Label, ...]  # nodes of the crosswalk's source schema
    reason: str = ""  # why there is no crosswalk (taxonomy.yaml)
    schema_open: bool = True  # the source schema lists no classes
    canonical: bool = False  # the source schema is itself canonical


@dataclass(frozen=True)
class UnitResult:
    record: dict[str, Any]  # JSON-ready
    pairs: tuple[tuple[str, CrosswalkEdge], ...] = ()  # (native label, edge) the unit uses


class Coverage(NamedTuple):
    labels: int
    missing: list[str]
    multiple: list[str]
    declared_unmapped: list[str]
    pairs: list[tuple[str, CrosswalkEdge]]


@dataclass(frozen=True)
class Context:
    registry: Registry
    root: Path
    meta: Mapping[str, Any]
    schemas: Mapping[str, LabelSchema]
    crosswalks: Mapping[str, Crosswalk]
    nodes: Mapping[str, Any]  # "<schema>:<node id>" of every canonical schema
    namespaces: tuple[str, ...]
    mapped: Mapping[str, frozenset[str]]  # source schema -> keys any of its crosswalks maps


def note_of(edge: CrosswalkEdge) -> str:
    return (edge.note or "").strip()


def capped(items: Iterable[str]) -> dict[str, Any]:
    ordered = sorted(set(items))
    return {"count": len(ordered), "items": ordered[:CAP]}


# ── labels and coverage (check 1) ────────────────────────────────────────────────


def merge_labels(labels: Iterable[Label]) -> tuple[Label, ...]:
    """Labels sharing a key are one label (a vocabulary and a schema naming the same thing)."""
    groups: list[tuple[str, set[str]]] = []
    owner: dict[str, int] = {}
    for label in labels:
        hit = next((owner[k] for k in label.keys if k in owner), None)
        if hit is None:
            hit = len(groups)
            groups.append((label.name, set()))
        groups[hit][1].update(label.keys)
        owner.update(dict.fromkeys(label.keys, hit))
    return tuple(Label(name, frozenset(keys)) for name, keys in groups)


def edge_index(edges: Iterable[CrosswalkEdge]) -> dict[str, list[CrosswalkEdge]]:
    index: dict[str, list[CrosswalkEdge]] = defaultdict(list)
    for edge in edges:
        for key in {edge.source_label, edge.source_label_id} - {None}:
            index[key].append(edge)
    return index


def matches(label: Label, index: Mapping[str, list[CrosswalkEdge]]) -> list[CrosswalkEdge]:
    return list({id(e): e for k in label.keys for e in index.get(k, ())}.values())


def coverage(
    observed: tuple[Label, ...],
    declared: tuple[Label, ...],
    edges: Iterable[CrosswalkEdge],
    mapped_elsewhere: frozenset[str],
) -> Coverage:
    """Check 1. The harvested vocabulary is the hard basis; without one, the schema is."""
    index = edge_index(edges)
    hard = merge_labels(observed) or merge_labels(declared)
    pairs: dict[int, tuple[str, CrosswalkEdge]] = {}
    missing: list[str] = []
    multiple: list[str] = []
    for lab in hard:
        found = matches(lab, index)
        if not found:
            missing.append(lab.name)
        elif len(found) > 1:
            multiple.append(lab.name)
        pairs.update({id(e): (lab.name, e) for e in found})
    unmapped: list[str] = []
    if observed:
        seen = {k for lab in hard for k in lab.keys}
        for lab in merge_labels(declared):
            if lab.keys & seen:
                continue
            found = matches(lab, index)
            pairs.update({id(e): (lab.name, e) for e in found})
            if not found and not lab.keys & mapped_elsewhere:
                unmapped.append(lab.name)
    return Coverage(len(hard), missing, multiple, unmapped, list(pairs.values()))


def duplicate_edges(edges: Iterable[CrosswalkEdge]) -> list[str]:
    edges = list(edges)
    names = Counter(e.source_label for e in edges)
    ids = Counter(e.source_label_id for e in edges if e.source_label_id)
    return sorted({k for k, n in names.items() if n > 1} | {k for k, n in ids.items() if n > 1})


# ── checks 2 and 3 ───────────────────────────────────────────────────────────────


def resolution(
    edges: Iterable[CrosswalkEdge], target: LabelSchema, ctx: Context
) -> tuple[list[str], list[str]]:
    """Check 2: ``(unresolved, ambiguous)`` edge targets, with the G9 bare-id resolver."""
    by_id = {n.id: n for n in target.nodes}
    unresolved: list[str] = []
    ambiguous: list[str] = []
    for edge in edges:
        for axis, ident in edge.targets.items():
            node = by_id.get(ident)
            where = f"{edge.source_label} -> {axis.value}:{ident}"
            if node is None:
                unresolved.append(f"{where} (not a node of {target.id})")
            elif node.axis is not axis:
                unresolved.append(f"{where} (the node is on the {node.axis.value} axis)")
            elif len(resolve_node(ident, ctx.nodes, ctx.namespaces)) > 1:
                ambiguous.append(where)
    return unresolved, ambiguous


def fidelity_checks(
    edges: Iterable[CrosswalkEdge], target: LabelSchema, source: LabelSchema | None
) -> tuple[list[str], list[str], list[str]]:
    """Check 3: ``(failures, lossy edges without a note, exact edges on an inner node whose
    native rank is unknown)``. An unmappable edge must say why it drops the label."""
    parents = {n.parent for n in target.nodes if n.parent}
    by_id = {n.id: n for n in target.nodes}
    ranks = {
        key: n.worms_rank.lower()
        for n in (source.nodes if source else ())
        if n.worms_rank
        for key in (n.id, n.name)
    }
    failures: list[str] = []
    unnoted: list[str] = []
    inner: list[str] = []
    for edge in edges:
        if edge.fidelity is Fidelity.UNMAPPABLE and not note_of(edge):
            failures.append(f"{edge.source_label}: unmappable edge without a note")
        if edge.fidelity in LOSSY and not note_of(edge):
            unnoted.append(edge.source_label)
        node = by_id.get(edge.targets.get(Axis.TAXON, ""))
        if edge.fidelity is not Fidelity.EXACT or node is None or node.id not in parents:
            continue
        native = ranks.get(edge.source_label) or ranks.get(edge.source_label_id or "")
        if native and (node.worms_rank or "").lower() != native:
            failures.append(
                f"{edge.source_label}: exact to {node.id} ({node.worms_rank}), native rank {native}"
            )
        elif not native:
            inner.append(edge.source_label)
    return failures, unnoted, inner


# ── units ────────────────────────────────────────────────────────────────────────


def _declared(source: Any) -> tuple[str | None, str | None]:
    loader = source.loader
    schema = getattr(loader, "schema_id", None) if loader else None
    schema = schema or next((a.schema_id for a in source.annotations if a.schema_id), None)
    return schema, (getattr(loader, "crosswalk_id", None) if loader else None)


def _vocab_index(root: Path) -> dict[str, tuple[str, tuple[Label, ...]]]:
    """source id -> (crosswalk id, harvested labels) from ``registry/taxonomy/vocab/*.tsv``."""
    found: dict[str, tuple[str, list[Label]]] = {}
    for path in sorted((root / "taxonomy" / "vocab").glob("*.tsv")):
        head, rows = read_vocab_rows(path)
        if "crosswalk" not in head or "source" not in head:
            continue  # a side table
        labels = [
            Label(r["label"], frozenset({r["label"], r["label_native_id"]} - {""})) for r in rows
        ]
        found.setdefault(head["source"], (head["crosswalk"], []))[1].extend(labels)
    return {sid: (cw, merge_labels(labels)) for sid, (cw, labels) in found.items()}


def _published(root: Path) -> dict[str, tuple[Label, ...]]:
    """source id -> labels of the published masks, instances and boxes (``label-schemes``)."""
    raw = yaml.safe_load((root / "label-schemes" / "sources.yaml").read_text())["sources"]
    out: dict[str, tuple[Label, ...]] = {}
    for sid, entry in raw.items():
        names = list((entry.get("ids") or {}).values()) or list(entry.get("labels") or ())
        out[sid] = tuple(Label(str(n), frozenset({str(n)})) for n in names)
    return out


def _schema_labels(schema: LabelSchema | None) -> tuple[Label, ...]:
    return tuple(Label(n.name, frozenset({n.id, n.name})) for n in (schema.nodes if schema else ()))


def make_context(registry: Registry | None = None, root: Path | None = None) -> Context:
    root = Path(root) if root else _default_root()
    registry = registry or Registry.load(root)
    mapped: dict[str, set[str]] = defaultdict(set)
    for cw in registry.crosswalks:
        for edge in cw.edges:
            mapped[cw.source_schema].update(
                k for k in (edge.source_label, edge.source_label_id) if k
            )
    nodes = {f"{s.id}:{n.id}": n for s in registry.schemas if s.canonical for n in s.nodes}
    return Context(
        registry=registry,
        root=root,
        meta=load_meta(root),
        schemas={s.id: s for s in registry.schemas},
        crosswalks={c.id: c for c in registry.crosswalks},
        nodes=nodes,
        namespaces=node_namespaces(nodes),
        mapped={k: frozenset(v) for k, v in mapped.items()},
    )


def _unit(
    ctx: Context, uid: str, origin: str, schema_id: str | None, cw_id: str | None, via: str,
    observed: tuple[Label, ...], reason: str = "",
) -> Unit:  # fmt: skip
    cw = ctx.crosswalks.get(cw_id or "")
    schema = ctx.schemas.get(cw.source_schema if cw else (schema_id or ""))
    return Unit(
        id=uid, origin=origin, schema_id=schema_id or (cw.source_schema if cw else None),
        crosswalk_id=cw_id, via=via, observed=observed, declared=_schema_labels(schema),
        reason=reason.strip(), schema_open=not (schema and schema.nodes),
        canonical=bool(schema and schema.canonical),
    )  # fmt: skip


def collect_units(ctx: Context) -> list[Unit]:
    """Registry sources first, then vocabulary-only sources, then crosswalks nothing points at."""
    registry, root, meta = ctx.registry, ctx.root, ctx.meta
    by_schema: dict[str, list[str]] = defaultdict(list)
    for cw in registry.crosswalks:
        by_schema[cw.source_schema].append(cw.id)
    vocab, published = _vocab_index(root), _published(root)
    pending = {k: str(v) for k, v in (meta.get("no_crosswalk_yet") or {}).items()}
    pending.update(
        {k: str(v.get("reason", "")) for k, v in (meta.get("source_crosswalks") or {}).items()}
    )
    units: list[Unit] = []
    for source in sorted(registry.sources, key=lambda s: s.id):
        schema_id, cw_id = _declared(source)
        if not (schema_id or cw_id):
            continue
        harvested = vocab.get(source.id)
        via = "loader" if cw_id else ""
        if not cw_id and harvested:
            cw_id, via = harvested[0], "vocab"
        schema = ctx.schemas.get(schema_id or "")
        if (
            not cw_id
            and schema is not None
            and schema.nodes
            and len(by_schema[schema_id or ""]) == 1
        ):
            cw_id, via = by_schema[schema_id or ""][0], "schema"
        observed = merge_labels(
            [*(harvested[1] if harvested else ()), *published.get(source.id, ())]
        )
        reason = "" if cw_id else pending.get(source.id, "")
        units.append(_unit(ctx, source.id, "registry", schema_id, cw_id, via, observed, reason))
    have = {u.id for u in units}
    for sid, (cw_id, labels) in sorted(vocab.items()):
        if sid not in have:
            units.append(_unit(ctx, sid, "vocab", None, cw_id, "vocab", labels))
    used = {u.crosswalk_id for u in units}
    for cw in sorted(registry.crosswalks, key=lambda c: c.id):
        if cw.id not in used:
            units.append(_unit(ctx, cw.id, "crosswalk", None, cw.id, "crosswalk", ()))
    return units


# ── evaluation ───────────────────────────────────────────────────────────────────


def evaluate_crosswalk(cw: Crosswalk, ctx: Context) -> dict[str, Any]:
    """Checks 2 and 3 on every edge of one crosswalk, whoever uses it."""
    target, source = ctx.schemas[cw.target_schema], ctx.schemas.get(cw.source_schema)
    unresolved, ambiguous = resolution(cw.edges, target, ctx)
    bad, unnoted, _ = fidelity_checks(cw.edges, target, source)
    dup = duplicate_edges(cw.edges)
    allowed = BASELINE_UNNOTED.get(cw.id, 0)
    failures = (
        [f"unresolved: {x}" for x in unresolved]
        + [f"ambiguous: {x}" for x in ambiguous]
        + [f"duplicate edge: {x}" for x in dup]
        + [f"fidelity: {x}" for x in bad]
    )
    if len(unnoted) > allowed:
        failures.append(f"lossy edges without a note: {len(unnoted)} (baseline {allowed})")
    kinds = Counter(
        "ignored" if e.fidelity is Fidelity.UNMAPPABLE else e.fidelity.value for e in cw.edges
    )
    return {
        "id": cw.id, "source_schema": cw.source_schema, "target_schema": cw.target_schema,
        "edges": len(cw.edges),
        **{
            k: kinds.get(k, 0)
            for k in ("exact", "coarsened", "approximate", "narrower", "ignored")
        },
        "unresolved": len(unresolved), "ambiguous": len(ambiguous), "duplicate": len(dup),
        "lossy_without_note": len(unnoted), "lossy_without_note_baseline": allowed,
        "lossy_without_note_labels": sorted(unnoted),
        "unmappable": [
            {"label": e.source_label, "note": note_of(e)}
            for e in cw.edges if e.fidelity is Fidelity.UNMAPPABLE
        ],
        "failures": failures, "status": "fail" if failures else "pass",
    }  # fmt: skip


def _blank(unit: Unit) -> dict[str, Any]:
    parts = [("vocab/published", unit.observed), ("schema", unit.declared)]
    return {
        "id": unit.id, "origin": unit.origin, "schema": unit.schema_id,
        "crosswalk": unit.crosswalk_id, "crosswalk_via": unit.via, "target_schema": None,
        "basis": "+".join(name for name, labels in parts if labels) or "edges",
        "status": "pass", "counts": dict.fromkeys(COUNT_KEYS, 0), "failures": [], "review": {},
        "unmappable": [],
    }  # fmt: skip


def _without_crosswalk(rec: dict[str, Any], unit: Unit) -> dict[str, Any]:
    rec["counts"]["labels"] = len(merge_labels([*unit.observed, *unit.declared]))
    rec["schema_open"] = unit.schema_open
    if unit.canonical:
        rec["status"] = "canonical-native"  # labelled in a canonical schema: nothing to cross-walk
    elif unit.schema_open:
        rec["status"] = "no-crosswalk"
        rec["documented"] = bool(unit.reason)
        rec["reason"] = unit.reason or "no class list registered (open vocabulary)"
    else:
        rec["status"] = "fail"
        rec["failures"] = [f"schema {unit.schema_id!r} lists classes but no crosswalk covers them"]
    return rec


def evaluate_unit(unit: Unit, ctx: Context, cw_records: Mapping[str, dict[str, Any]]) -> UnitResult:
    """Checks 1-3 for one unit. A crosswalk-level failure also fails every unit that uses it."""
    rec = _blank(unit)
    if not unit.crosswalk_id:
        return UnitResult(_without_crosswalk(rec, unit))
    cw = ctx.crosswalks.get(unit.crosswalk_id)
    if cw is None:
        rec.update(
            status="fail", failures=[f"crosswalk {unit.crosswalk_id!r} is not in the registry"]
        )
        return UnitResult(rec)
    target, source = ctx.schemas[cw.target_schema], ctx.schemas.get(cw.source_schema)
    declared = () if (source and source.canonical) else unit.declared  # a bridge has no class list
    cov = coverage(unit.observed, declared, cw.edges, ctx.mapped.get(cw.source_schema, frozenset()))
    pairs = cov.pairs if cov.labels else [(e.source_label, e) for e in cw.edges]
    edges = [e for _, e in pairs]
    unresolved, ambiguous = resolution(edges, target, ctx)
    bad, unnoted, inner = fidelity_checks(edges, target, source)
    kinds = Counter(
        "ignored" if e.fidelity is Fidelity.UNMAPPABLE else e.fidelity.value for e in edges
    )
    counts = rec["counts"]
    counts.update(
        {k: kinds.get(k, 0) for k in ("exact", "coarsened", "approximate", "narrower", "ignored")}
    )
    counts.update(
        labels=cov.labels or len(edges), edges=len(edges), unresolved=len(unresolved),
        ambiguous=len(ambiguous), missing=len(cov.missing), duplicate=len(cov.multiple),
        declared_unmapped=len(cov.declared_unmapped), lossy_without_note=len(unnoted),
        unused_edges=len(cw.edges) - len(edges) if cov.labels else 0,
    )  # fmt: skip
    failures = (
        [f"missing edge: {x}" for x in cov.missing]
        + [f"duplicate edge: {x}" for x in cov.multiple]
        + [f"unresolved: {x}" for x in unresolved]
        + [f"ambiguous: {x}" for x in ambiguous]
        + [f"fidelity: {x}" for x in bad]
    )
    allowed = BASELINE_DECLARED_GAPS.get(unit.id, 0)
    if len(cov.declared_unmapped) > allowed:
        failures.append(
            f"schema nodes without an edge: {len(cov.declared_unmapped)} (baseline {allowed})"
        )
    if cw_records[cw.id]["failures"]:
        failures.append(f"crosswalk {cw.id} fails {len(cw_records[cw.id]['failures'])} check(s)")
    rec.update(target_schema=target.id, failures=failures, status="fail" if failures else "pass")
    rec["unmappable"] = [
        {"label": e.source_label, "note": note_of(e)}
        for e in edges
        if e.fidelity is Fidelity.UNMAPPABLE
    ]
    rec["review"] = {
        "declared_unmapped": capped(cov.declared_unmapped),
        "lossy_without_note": capped(unnoted),
        "exact_on_inner_node": capped(inner),
    }
    return UnitResult(rec, tuple(pairs))
