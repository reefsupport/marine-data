"""Stem-keyed links between two registry sources that carry the same frames (Coralscapes V1 / V2).

V2 re-exports every V1 frame (image bytes and mask pixels differ, so digests never match) and adds
new ones. A source entry declares the relation::

    links:
      - {source_id: coralscapes, key: stem, relation: re-annotation}

The V2 ``stem`` equals the V1 file stem minus ``_leftImg8bit`` (:func:`normalise_stem`). Every frame
the two sources share is a *twin* pair; the groups of its two images are one split component, so the
pair always shares a split and the v1.0 split wins (:func:`~marinedata.release.split_map_inherit.
plan_inheritance`). A frame only the declaring source has (V2-only) follows its own upstream split
through the benchmark pins; a pin that would move a component holding a v1.0 group is dropped and
counted (:func:`protect_v1_pins`), never applied. A twin whose V2 upstream split differs from the
split v1.0 published is a conflict: counted and listed, never silently resolved the other way.

The generator records :func:`link_block` under ``SPLIT_MAP.json`` ``inherits.stem_links`` (counts,
capped conflict list and the twin digest pairs). Gates read it with :func:`read_link_table`.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..benchmark_pins import MAP_NAME, BenchmarkPins, accepted

KEY_STEM = "stem"
RELATIONS = frozenset({"re-annotation"})
MAX_LISTED = 200
BLOCK = "stem_links"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp")
LEFT_IMG = "_leftImg8bit"
UPSTREAM_TO_MAP = {
    "train": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "test": "test",
}
Row = tuple[str, str, str]  # image sha256, split group, source id


class StemLinkError(ValueError):
    """A registry ``links:`` declaration the stem-link machinery cannot honour."""


def normalise_stem(value: object) -> str:
    """``dir/site1_000001_016200_leftImg8bit.png`` and ``site1_000001_016200`` -> the latter."""
    name = str(value).replace("\\", "/").rsplit("/", 1)[-1].strip()
    if name.lower().endswith(IMAGE_SUFFIXES):
        name = name.rsplit(".", 1)[0]
    return name.removesuffix(LEFT_IMG)


def map_split(value: object) -> str | None:
    """An upstream split label as a split-map name (``validation`` -> ``val``); unknown -> None."""
    return UPSTREAM_TO_MAP.get(str(value or "").strip().lower())


@dataclass(frozen=True)
class LinkDecl:
    """``source_id`` declares ``links: [{source_id: target_id, key, relation}]``."""

    source_id: str
    target_id: str
    key: str = KEY_STEM
    relation: str = "re-annotation"


def _field(item: object, name: str) -> object:
    return item.get(name) if isinstance(item, Mapping) else getattr(item, name, None)


def parse_link_decls(source_id: str, raw: Iterable[object] | None) -> tuple[LinkDecl, ...]:
    """The validated ``links:`` of one source entry (mappings or objects with those attributes)."""
    out: list[LinkDecl] = []
    for item in raw or ():
        target = str(_field(item, "source_id") or "")
        key = str(_field(item, "key") or KEY_STEM)
        relation = str(_field(item, "relation") or "re-annotation")
        if not target or target == source_id:
            raise StemLinkError(f"{source_id}: links entry names no other source ({target!r})")
        if key != KEY_STEM:
            raise StemLinkError(f"{source_id}: links key {key!r} unsupported (only {KEY_STEM!r})")
        if relation not in RELATIONS:
            raise StemLinkError(
                f"{source_id}: links relation {relation!r} not in {sorted(RELATIONS)}"
            )
        out.append(LinkDecl(source_id, target, key, relation))
    return tuple(out)


def declared_links(registry: Iterable[Any]) -> tuple[LinkDecl, ...]:
    """Every ``links:`` declaration of the registry's sources (none until a source declares one)."""
    return tuple(
        decl
        for source in registry
        for decl in parse_link_decls(source.id, getattr(source, "links", None))
    )


@dataclass(frozen=True)
class StemEntry:
    """One image of a source as the link table sees it."""

    source_id: str
    stem: str
    sha256: str
    group: str
    upstream_split: str | None = None


@dataclass(frozen=True)
class Twin:
    stem: str
    declaring_source: str
    declaring_sha: str
    declaring_group: str
    declaring_upstream: str | None
    target_source: str
    target_sha: str
    target_group: str


@dataclass(frozen=True)
class StemConflict:
    stem: str
    kind: str  # upstream_vs_v1 | pin_dropped
    v1: str  # the split(s) v1.0 holds for the frame / component
    v2: str  # what V2 asked for: its upstream split, or the pin target


@dataclass(frozen=True)
class StemLinks:
    decls: tuple[LinkDecl, ...] = ()
    twins: tuple[Twin, ...] = ()
    declaring_only: tuple[str, ...] = ()
    """Digests of declaring-source frames with no twin (V2-only): they follow their own pins."""
    target_only: int = 0
    ambiguous: int = 0
    """Stems that name more than one digest in a source (all of them are linked)."""
    upstream: Mapping[str, str] = field(default_factory=dict)
    """Declaring-source digest -> its upstream split as a split-map name (twins and V2-only)."""

    @property
    def pairs(self) -> list[tuple[str, str]]:
        """``(target digest, declaring digest)``: the ``links`` for ``rows_to_counts``."""
        return sorted(
            {(t.target_sha, t.declaring_sha) for t in self.twins if t.target_sha != t.declaring_sha}
        )

    def group_edges(self) -> list[tuple[str, str]]:
        return sorted({(t.target_group, t.declaring_group) for t in self.twins})

    def merge(self, canonical: Mapping[str, str]) -> dict[str, str]:
        """``canonical`` with the twins' components joined (smallest member = representative).
        Idempotent: a map ``rows_to_counts`` already merged through :attr:`pairs` is unchanged."""
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for a, b in self.group_edges():
            ra, rb = find(canonical.get(a, a)), find(canonical.get(b, b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)  # the smaller id stays the root
        out = {g: find(c) if c in parent else c for g, c in canonical.items()}
        for a, b in self.group_edges():
            out.setdefault(a, find(a))
            out.setdefault(b, find(b))
        return out


def build_stem_links(entries: Iterable[StemEntry], decls: Sequence[LinkDecl]) -> StemLinks:
    """The twin table of ``entries`` under ``decls``, keyed ``(source_id, normalised stem)``."""
    items = list(entries)
    declaring = {d.source_id for d in decls}
    upstream = {
        e.sha256: split
        for e in items
        if e.source_id in declaring and (split := map_split(e.upstream_split)) is not None
    }
    table: dict[tuple[str, str], dict[str, StemEntry]] = defaultdict(dict)
    for e in items:
        table[(e.source_id, normalise_stem(e.stem))].setdefault(e.sha256, e)
    twins: list[Twin] = []
    only: list[str] = []
    ambiguous = 0
    matched: set[tuple[str, str]] = set()
    for decl in sorted(decls, key=lambda d: (d.source_id, d.target_id)):
        for (source, stem), mine in sorted(table.items()):
            if source != decl.source_id:
                continue
            theirs = table.get((decl.target_id, stem))
            if not theirs:
                only.extend(sorted(mine))
                continue
            matched.add((decl.target_id, stem))
            ambiguous += len(mine) > 1 or len(theirs) > 1
            twins.extend(
                Twin(stem, source, a.sha256, a.group, map_split(a.upstream_split),
                     decl.target_id, b.sha256, b.group)
                for a in sorted(mine.values(), key=lambda x: x.sha256)
                for b in sorted(theirs.values(), key=lambda x: x.sha256)
            )  # fmt: skip
    targets = {d.target_id for d in decls}
    target_only = sum(1 for key in table if key[0] in targets and key not in matched)
    return StemLinks(
        tuple(decls), tuple(twins), tuple(sorted(set(only))), target_only, ambiguous, upstream
    )


def linked_entries(
    registry: Any,
    roots: Mapping[str, str | Path],
    decls: Sequence[LinkDecl],
    digest: Callable[[Path], str],
) -> list[StemEntry]:
    """``StemEntry`` of every image of the sources ``decls`` name (the same enumeration the split
    map uses, so a digest here is a digest in its rows). ``digest`` may be a caching one."""
    from .entries import source_release_entries

    wanted = sorted({s for d in decls for s in (d.source_id, d.target_id)} & set(roots))
    return [
        StemEntry(sid, normalise_stem(e.path.name), e.sha256, e.group, e.upstream_split)
        for sid in wanted
        for e in source_release_entries(registry.source(sid), roots[sid], digest)
    ]


def twin_conflicts(
    links: StemLinks, prior_of: Callable[[Twin], Collection[str]]
) -> tuple[StemConflict, ...]:
    """Twins whose V2 upstream split is not among the splits v1.0 gave the frame (``prior_of``)."""
    out = []
    for t in links.twins:
        held = set(prior_of(t))
        if t.declaring_upstream is not None and held and t.declaring_upstream not in held:
            out.append(
                StemConflict(t.stem, "upstream_vs_v1", "/".join(sorted(held)), t.declaring_upstream)
            )
    return tuple(sorted(out, key=lambda c: (c.kind, c.stem)))


def protect_v1_pins(
    pins: BenchmarkPins,
    links: StemLinks,
    rows: Sequence[Row],
    canonical: Mapping[str, str],
    v1_splits: Mapping[str, Collection[str]],
) -> tuple[BenchmarkPins, tuple[StemConflict, ...]]:
    """``pins`` without the declaring sources' pins a v1.0 component does not accept.

    A pin wins over a v1.0 assignment in :func:`~marinedata.benchmark_pins.apply_benchmark_pins`;
    for the re-annotation (V2) source v1.0 must win instead (its configs are frozen), so such a pin
    is dropped here and returned as a ``pin_dropped`` conflict. ``canonical`` is the merged
    component map (:meth:`StemLinks.merge`), ``v1_splits`` the v1.0 tree's ``sha -> splits``.
    Pins of components without a v1.0 image, or whose v1.0 splits the pin accepts, stay."""
    declaring = {d.source_id for d in links.decls}
    held: dict[str, set[str]] = defaultdict(set)
    for sha, group, _source in rows:
        if sha in v1_splits:
            held[canonical.get(group, group)].update(v1_splits[sha])
    stems = {t.declaring_sha: t.stem for t in links.twins}
    kept = dict(pins.by_sha)
    dropped: list[StemConflict] = []
    for sha, group, source in sorted(rows):
        hit = pins.by_sha.get(sha)
        splits = held.get(canonical.get(group, group))
        if hit is None or source not in declaring or not splits:
            continue
        if not all(accepted(hit[0], split) for split in splits):
            kept.pop(sha, None)
            label = stems.get(sha, sha[:12])
            dropped.append(
                StemConflict(label, "pin_dropped", "/".join(sorted(splits)), MAP_NAME[hit[0]])
            )
    return replace(pins, by_sha=kept), tuple(dropped)


def link_block(links: StemLinks, conflicts: Sequence[StemConflict] = ()) -> dict[str, object]:
    """``inherits.stem_links`` of ``SPLIT_MAP.json``: counts, capped conflicts, the twin pairs."""
    return {
        "links": [d.__dict__ for d in links.decls],
        "twins": len(links.twins),
        "declaring_only": len(links.declaring_only),
        "target_only": links.target_only,
        "ambiguous_stems": links.ambiguous,
        "conflicts": {
            "count": len(conflicts),
            "by_kind": dict(sorted(Counter(c.kind for c in conflicts).items())),
            "listed": [c.__dict__ for c in conflicts[:MAX_LISTED]],
        },
        "pairs": [list(p) for p in links.pairs],
    }


@dataclass(frozen=True)
class LinkTable:
    """What a gate knows of the links: the twin digest pairs of the map and its counts."""

    pairs: tuple[tuple[str, str], ...] = ()
    block: Mapping[str, object] = field(default_factory=dict)
    _index: frozenset[tuple[str, str]] = frozenset()

    def linked(self, a: str, b: str) -> bool:
        return (a, b) in self._index

    def summary(self) -> dict[str, object]:
        """The block without the pair list: what a gate prints."""
        return {k: v for k, v in self.block.items() if k != "pairs"}


def read_link_table(split_map: str | Path) -> LinkTable | None:
    """The ``inherits.stem_links`` block of a ``SPLIT_MAP.json`` (``None``: the map has none)."""
    try:
        data = json.loads(Path(split_map).read_text())
    except (OSError, ValueError):
        return None  # malformed maps are reported by the map gates, not here
    inherits = data.get("inherits") if isinstance(data, dict) else None
    block = inherits.get(BLOCK) if isinstance(inherits, dict) else None
    if not isinstance(block, dict):
        return None
    raw = block.get("pairs")
    pairs = tuple(
        (str(p[0]), str(p[1]))
        for p in (raw if isinstance(raw, list) else ())
        if isinstance(p, list | tuple) and len(p) == 2
    )
    return LinkTable(pairs, block, frozenset({*pairs, *((b, a) for a, b in pairs)}))


def load_link_table(
    build_dir: str | Path, release_dir: str | Path | None = None
) -> LinkTable | None:
    """The link table of the ``SPLIT_MAP.json`` the gates read (build dir, else release dir);
    ``None`` when there is no map, it does not parse or it carries no ``inherits.stem_links``."""
    from ..gates.shards import find_split_map  # lazy: the gates import this module

    found = find_split_map(Path(build_dir), Path(release_dir) if release_dir else None)
    if found is None:
        return None
    try:
        return read_link_table(found[0])
    except (OSError, ValueError):
        return None


def gate_counts(table: LinkTable) -> dict[str, object]:
    """Flat counters a gate prints: twins, V2-only frames and the V1-wins conflicts."""
    raw = table.block.get("conflicts")
    conflicts = raw if isinstance(raw, dict) else {}
    return {
        "stem_link_twins": table.block.get("twins", len(table.pairs)),
        "stem_link_declaring_only": table.block.get("declaring_only", 0),
        "stem_link_conflicts": conflicts.get("count", 0),
        "stem_link_conflicts_by_kind": conflicts.get("by_kind", {}),
    }


def train_forced(
    links: StemLinks, rows: Sequence[Row], canonical: Mapping[str, str]
) -> dict[str, str]:
    """component -> ``train`` for every component made only of declaring-source (V2) frames whose
    upstream split is ``train``: no v1.0 image touches it and no benchmark pin names it, so V2's own
    train frames stay in train instead of drawing a random evaluation split."""
    declaring = {d.source_id for d in links.decls}
    kinds: dict[str, set[str | None]] = defaultdict(set)
    for sha, group, source in rows:
        upstream = links.upstream.get(sha) if source in declaring else None
        kinds[canonical.get(group, group)].add(upstream)
    return {comp: "train" for comp, seen in sorted(kinds.items()) if seen == {"train"}}
