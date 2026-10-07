"""Training label schemes: loader and resolver behind :mod:`marinedata.labels`.

Reads ``registry/label-schemes/{schemes,sources}.yaml`` (packaged as ``marinedata/_registry``),
and resolves a native label to a scheme class through ``registry/crosswalks`` (label -> taxon node
[+ condition]) and ``registry/tasks`` (class list of ``benthic-coarse``). Pure python, no numpy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .registry import Registry, _default_root
from .schema import Axis

IGNORE = 255
_TARGET_SCHEMA = "rs-benthic-v1"


@dataclass(frozen=True)
class SourceSpec:
    """One published source: where it lives and how its labels are keyed."""

    id: str
    repo: str
    config: str
    kind: str  # semantic | instances | boxes
    crosswalk: str
    dense: bool
    ids: Mapping[int, str] = field(default_factory=dict)  # semantic: pixel value -> native label
    labels: tuple[str, ...] = ()  # instances / boxes: native labels

    @property
    def native_labels(self) -> tuple[str, ...]:
        return tuple(self.ids.values()) if self.kind == "semantic" else self.labels


@dataclass(frozen=True)
class SchemeSpec:
    id: str
    description: str
    classes: tuple[str, ...]
    sources: tuple[str, ...]
    rules: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # class -> nodes
    ignore_nodes: tuple[str, ...] = ()
    source_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    labels: Mapping[str, str] | None = None  # explicit by-name table (scene)
    unmapped_labels: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Options:
    """Normalised, hashable lut options."""

    exclude_conditions: frozenset[str] = frozenset()
    ignore: frozenset[str] = frozenset()


def _root() -> Path:
    return _default_root() / "label-schemes"


def _load_yaml(name: str) -> dict[str, Any]:
    with (_root() / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@cache
def _registry() -> Registry:
    return Registry.load()


@cache
def sources() -> Mapping[str, SourceSpec]:
    out: dict[str, SourceSpec] = {}
    for sid, raw in _load_yaml("sources.yaml")["sources"].items():
        out[sid] = SourceSpec(
            id=sid,
            repo=raw["repo"],
            config=raw["config"],
            kind=raw["kind"],
            crosswalk=raw["crosswalk"],
            dense=bool(raw["dense"]),
            ids={int(k): v for k, v in (raw.get("ids") or {}).items()},
            labels=tuple(raw.get("labels") or ()),
        )
    return out


@cache
def condition_aliases() -> Mapping[str, tuple[str, ...]]:
    return {k: tuple(v) for k, v in _load_yaml("schemes.yaml")["condition_aliases"].items()}


@cache
def schemes() -> Mapping[str, SchemeSpec]:
    out: dict[str, SchemeSpec] = {}
    for sid, raw in _load_yaml("schemes.yaml")["schemes"].items():
        if "task" in raw:
            classes = tuple(_registry().task(raw["task"]).classes)
            rules = {c: (c,) for c in classes}
        else:
            classes = tuple(raw["classes"])
            rules = {c: tuple(n) for c, n in (raw.get("rules") or {}).items()}
        if len(classes) >= IGNORE:
            raise ValueError(
                f"scheme {sid}: {len(classes)} classes do not fit uint8 with 255 ignored"
            )
        out[sid] = SchemeSpec(
            id=sid,
            description=" ".join(str(raw.get("description", "")).split()),
            classes=classes,
            sources=tuple(raw["sources"]),
            rules=rules,
            ignore_nodes=tuple(raw.get("ignore_nodes") or ()),
            source_overrides={s: dict(m) for s, m in (raw.get("source_overrides") or {}).items()},
            labels=dict(raw["labels"]) if "labels" in raw else None,
            unmapped_labels=dict(raw.get("unmapped_labels") or {}),
        )
    return out


def scheme_spec(scheme: str) -> SchemeSpec:
    try:
        return schemes()[scheme]
    except KeyError:
        raise ValueError(f"unknown scheme {scheme!r}; known: {sorted(schemes())}") from None


def source_spec(source: str, scheme: str | None = None) -> SourceSpec:
    try:
        spec = sources()[source]
    except KeyError:
        raise ValueError(f"unknown source {source!r}; known: {sorted(sources())}") from None
    if scheme is not None and source not in scheme_spec(scheme).sources:
        raise ValueError(
            f"source {source!r} is not defined for scheme {scheme!r}; "
            f"sources: {list(scheme_spec(scheme).sources)}"
        )
    return spec


def make_options(exclude_conditions: Iterable[str] = (), ignore: Iterable[str] = ()) -> Options:
    """Validate and normalise the two public options (typos raise rather than silently no-op)."""
    if isinstance(exclude_conditions, str) or isinstance(ignore, str):
        raise TypeError("exclude_conditions and ignore take a sequence of strings, not a string")
    schema = _registry().label_schema(_TARGET_SCHEMA)
    aliases = condition_aliases()
    nodes: set[str] = set()
    for name in exclude_conditions:
        if name in aliases:
            nodes.update(aliases[name])
            continue
        node = schema.node(name)
        if node is None or node.axis is not Axis.CONDITION:
            raise ValueError(
                f"unknown condition {name!r}; use an alias {sorted(aliases)} or a condition node"
            )
        nodes.add(name)
    return Options(frozenset(nodes), frozenset(ignore))


def _walk(node: str | None) -> tuple[str, ...]:
    """``node`` then its ancestors, nearest first (empty for no node)."""
    if not node:
        return ()
    return (node, *_registry().label_schema(_TARGET_SCHEMA).ancestors(node))


def _class_of_node(spec: SchemeSpec, node: str) -> str | None:
    """Rule resolution: the nearest matching node wins (self, then ancestors).

    A node that is an ancestor of a rule node of a *different* class is coarser than the target
    vocabulary and abstains (``None``) rather than being forced into a class.
    """
    schema = _registry().label_schema(_TARGET_SCHEMA)
    chosen: str | None = None
    for hit in _walk(node):
        if hit in spec.ignore_nodes:
            return None
        chosen = next((c for c, nodes in spec.rules.items() if hit in nodes), None)
        if chosen is not None:
            break
    if chosen is None:
        return None
    for cls, nodes in spec.rules.items():
        if cls != chosen and any(node in schema.ancestors(r) for r in nodes):
            return None
    return chosen


def resolve_class(
    source: str, label: str, scheme: str, options: Options | None = None
) -> str | None:
    """The scheme class name of ``label`` in ``source``, or ``None`` (-> 255)."""
    options = options or Options()
    spec = scheme_spec(scheme)
    src = source_spec(source, scheme)
    if label not in src.native_labels:
        raise ValueError(f"label {label!r} is not in the vocabulary of source {source!r}")
    if label in options.ignore:
        return None
    override = spec.source_overrides.get(source, {})
    if label in override:
        return override[label]
    if spec.labels is not None:  # explicit by-name table
        if label in spec.labels:
            return spec.labels[label]
        if label in spec.unmapped_labels:
            return None
        raise ValueError(f"scheme {scheme!r} has no entry for label {label!r} of {source!r}")
    edge = _registry().crosswalk(src.crosswalk).edge(label)
    taxon = edge.targets.get(Axis.TAXON) if edge is not None else None
    if taxon is None:
        return None
    if options.ignore and options.ignore.intersection(_walk(taxon)):
        return None
    condition = edge.targets.get(Axis.CONDITION)
    if condition and options.exclude_conditions.intersection(_walk(condition)):
        return None
    return _class_of_node(spec, taxon)


def class_id(scheme: str, name: str | None) -> int:
    spec = scheme_spec(scheme)
    return IGNORE if name is None else spec.classes.index(name)
