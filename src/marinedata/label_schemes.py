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
    source_overrides: Mapping[str, Mapping[str, str | None]] = field(default_factory=dict)
    labels: Mapping[str, str] | None = None  # explicit by-name table (scene)
    unmapped_labels: Mapping[str, str] = field(default_factory=dict)
    default_exclude_conditions: tuple[str, ...] = ()  # aliases / condition nodes, see make_options


@dataclass(frozen=True)
class Options:
    """Normalised, hashable lut options (condition aliases already expanded to nodes)."""

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
        overrides = {s: dict(m) for s, m in (raw.get("source_overrides") or {}).items()}
        for source, table in overrides.items():
            for label, name in table.items():
                if name is not None and name not in classes:
                    raise ValueError(
                        f"scheme {sid}: override {source}/{label!r} names {name!r}, "
                        f"not one of {list(classes)} (use null for 255)"
                    )
        defaults = tuple(raw.get("default_exclude_conditions") or ())
        _condition_nodes(defaults)  # a typo in the YAML fails at load, not at first use
        out[sid] = SchemeSpec(
            id=sid,
            description=" ".join(str(raw.get("description", "")).split()),
            classes=classes,
            sources=tuple(raw["sources"]),
            rules=rules,
            ignore_nodes=tuple(raw.get("ignore_nodes") or ()),
            source_overrides=overrides,
            labels=dict(raw["labels"]) if "labels" in raw else None,
            unmapped_labels=dict(raw.get("unmapped_labels") or {}),
            default_exclude_conditions=defaults,
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


def _condition_nodes(names: Iterable[str]) -> frozenset[str]:
    """Expand condition aliases (``dead``) and validate raw condition nodes; typos raise."""
    if isinstance(names, str):
        raise TypeError("exclude_conditions takes a sequence of strings, not a string")
    schema = _registry().label_schema(_TARGET_SCHEMA)
    aliases = condition_aliases()
    nodes: set[str] = set()
    for name in names:
        if name in aliases:
            nodes.update(aliases[name])
            continue
        node = schema.node(name)
        if node is None or node.axis is not Axis.CONDITION:
            raise ValueError(
                f"unknown condition {name!r}; use an alias {sorted(aliases)} or a condition node"
            )
        nodes.add(name)
    return frozenset(nodes)


def make_options(
    scheme: str,
    exclude_conditions: Iterable[str] | None = None,
    ignore: Iterable[str] = (),
) -> Options:
    """Validate and normalise the two public options (typos raise rather than silently no-op).

    ``exclude_conditions=None`` is the scheme's ``default_exclude_conditions``; any explicit value,
    including ``()``, replaces that default.
    """
    if isinstance(ignore, str):
        raise TypeError("ignore takes a sequence of strings, not a string")
    if exclude_conditions is None:
        exclude_conditions = scheme_spec(scheme).default_exclude_conditions
    return Options(_condition_nodes(exclude_conditions), frozenset(ignore))


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


def _crosswalk_class(spec: SchemeSpec, src: SourceSpec, label: str, options: Options) -> str | None:
    """Class of ``label`` through crosswalk -> taxon (-> condition) -> scheme rules."""
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


def _check_label(src: SourceSpec, label: str) -> None:
    if label not in src.native_labels:
        raise ValueError(f"label {label!r} is not in the vocabulary of source {src.id!r}")


def resolve_class(
    source: str, label: str, scheme: str, options: Options | None = None
) -> str | None:
    """The scheme class name of ``label`` in ``source``, or ``None`` (-> 255).

    ``options=None`` means the scheme defaults (``default_exclude_conditions``, nothing ignored).
    """
    options = options if options is not None else make_options(scheme)
    spec = scheme_spec(scheme)
    src = source_spec(source, scheme)
    _check_label(src, label)
    if label in options.ignore:
        return None
    override = spec.source_overrides.get(source, {})
    if label in override:  # null -> 255
        return override[label]
    if spec.labels is not None:  # explicit by-name table
        if label in spec.labels:
            return spec.labels[label]
        if label in spec.unmapped_labels:
            return None
        raise ValueError(f"scheme {scheme!r} has no entry for label {label!r} of {source!r}")
    return _crosswalk_class(spec, src, label, options)


def registry_class(source: str, label: str, scheme: str) -> str | None:
    """What the registry alone gives: crosswalk -> taxon -> class, with no scheme overrides.

    No default condition exclusions apply either. This is the column the registry publishes as
    ``coarse`` (for ``benthic-coarse``), so ``remap_row`` validates a row against it, not against
    the scheme output.
    """
    spec = scheme_spec(scheme)
    src = source_spec(source, scheme)
    _check_label(src, label)
    if spec.labels is not None:
        return resolve_class(source, label, scheme, Options())
    return _crosswalk_class(spec, src, label, Options())


def class_id(scheme: str, name: str | None) -> int:
    spec = scheme_spec(scheme)
    return IGNORE if name is None else spec.classes.index(name)
