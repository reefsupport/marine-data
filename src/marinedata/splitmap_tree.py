"""The published v1.0 tree as a split-map prior (``release split-map --inherit-tree DIR``).

v1.0's splits were made by hand and exist only in its Hub tree
(``<dir>/data/<config>/<split>-*.parquet``, column ``image_sha256``). They are read per image with
:func:`marinedata.gates.v11_common.v1_splits`, the reader gate G6 uses, and mapped onto *this*
release's groups, never onto v1.0's own ``split_group`` ids:

* a group whose v1.0 images share a v1.0 split keeps it; it is never withheld and only a listed
  benchmark pin moves it (precedence 1). Its dHash <= 4 near-copies join its component and adopt
  it (2), and its images anchor the never-eval / never-train band (3);
* a component whose v1.0 groups sit in different v1.0 splits keeps each group's split and is not
  re-split; its new groups are withheld and it is listed in ``inherits.v1_tree.straddles``, which
  gate G6 reads;
* a single group whose own v1.0 images sit in different v1.0 splits cannot keep both in a map keyed
  by group: it is withheld (its v1.0 configs keep the published splits) and listed there too.

A sha that v1.0 published in several splits accepts any of them. The choice prefers the split a
benchmark pin asks for, then ``train``, ``test``, ``val``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .scan import SplitName
from .splitmap import SplitMapError

TREE_SPLITS: dict[str, SplitName] = {
    "train": "train",
    "validation": "val",
    "test": "test",
    "test_ood": "test_ood",
}
PREFERENCE: tuple[SplitName, ...] = ("train", "test", "val", "test_ood")
Row = tuple[str, str, str]  # image sha256, split group, source id


@dataclass(frozen=True)
class V1Tree:
    splits: Mapping[str, frozenset[SplitName]]
    """``image_sha256 -> the v1.0 splits that published it`` (split-map names: ``val``)."""
    header: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TreePrior:
    groups: dict[str, SplitName] = field(default_factory=dict)
    """This release's groups that hold v1.0 images -> the v1.0 split each keeps."""
    anchors: dict[str, SplitName] = field(default_factory=dict)
    """v1.0 image -> its split as a band anchor (``train`` when v1.0 trained on it)."""
    straddles: dict[str, str] = field(default_factory=dict)
    """Group -> reason: v1.0 put its own images in different splits (withheld, listed)."""
    split_components: tuple[str, ...] = ()
    """Components whose v1.0 groups keep different v1.0 splits (not re-split, listed)."""


def load_v1_tree(path: str | Path) -> V1Tree:
    """``sha -> v1.0 splits`` of the published tree at ``path``; raises if it holds none."""
    from .gates.v11_common import v1_splits

    root = Path(path)
    if not (root / "data").is_dir():
        raise SplitMapError(f"{root}: no data/ directory: --inherit-tree needs the v1.0 Hub tree")
    raw = v1_splits(root)
    unknown = sorted({s for found in raw.values() for s in found} - set(TREE_SPLITS))
    if unknown:
        raise SplitMapError(f"{root}: v1.0 split(s) {unknown} are not one of {sorted(TREE_SPLITS)}")
    if not raw:
        raise SplitMapError(f"{root}: no image_sha256 under data/*/: not a v1.0 tree")
    splits = {sha: frozenset(TREE_SPLITS[s] for s in found) for sha, found in raw.items()}
    by_split = Counter(s for found in splits.values() for s in found)
    header = {
        "dir": root.name,
        "images": len(splits),
        "by_split": dict(sorted(by_split.items())),
        "multi_split_images": sum(1 for found in splits.values() if len(found) > 1),
    }
    return V1Tree(splits=splits, header=header)


def _choose(options: Collection[SplitName]) -> SplitName:
    return next(split for split in PREFERENCE if split in options)


def tree_prior(
    rows: Sequence[Row],
    tree: V1Tree,
    canonical: Mapping[str, str],
    prefer: Mapping[str, SplitName] | None = None,
) -> TreePrior:
    """The v1.0 split of every group of ``rows`` that holds a v1.0 image (see the module doc).

    ``canonical`` maps a group to its merged component; ``prefer`` is component -> the split a
    benchmark pin asks for, so a group that may sit there does and only a real move is listed."""
    common: dict[str, frozenset[SplitName]] = {}
    seen: dict[str, set[SplitName]] = {}
    anchors: dict[str, SplitName] = {}
    for sha, group, _source in sorted(rows):
        found = tree.splits.get(sha)
        if found is None:
            continue
        anchors[sha] = _choose(found)
        common[group] = common[group] & found if group in common else frozenset(found)
        seen.setdefault(group, set()).update(found)
    straddles = {
        group: (
            f"v1.0 published this group's images in {' and '.join(sorted(seen[group]))}: a map "
            "keyed by group cannot keep both, so v1.1 does not re-split it (v1.0 keeps them)"
        )
        for group, options in sorted(common.items())
        if not options
    }
    by_component: dict[str, list[str]] = {}
    for group in sorted(common):
        if common[group]:
            by_component.setdefault(canonical.get(group, group), []).append(group)
    chosen: dict[str, SplitName] = {}
    split_components: list[str] = []
    for canon, groups in sorted(by_component.items()):
        want = (prefer or {}).get(canon)
        shared = frozenset.intersection(*(common[g] for g in groups))
        for group in groups:
            if want is not None and want in common[group]:
                chosen[group] = want
            else:
                chosen[group] = _choose(shared or common[group])
        if len({chosen[g] for g in groups}) > 1:
            split_components.append(canon)
    return TreePrior(
        groups=chosen,
        anchors=anchors,
        straddles=straddles,
        split_components=tuple(split_components),
    )


def straddle_block(
    prior: TreePrior, canonical: Mapping[str, str], moved: Collection[str] = ()
) -> dict[str, object]:
    """``inherits.v1_tree.straddles``: the v1.0 groups of a component v1.0 already split (G6
    accepts exactly these as ``group_straddles``) and the withheld single-group straddles.
    ``moved`` are components a benchmark pin moved whole: they no longer straddle."""
    split = set(prior.split_components) - set(moved)
    group_ids = sorted(g for g in prior.groups if canonical.get(g, g) in split)
    return {
        "components": len(split),
        "group_ids": group_ids,
        "withheld_groups": len(prior.straddles),
        "withheld_group_ids": sorted(prior.straddles),
    }
