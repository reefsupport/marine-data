"""Row-level inputs of an inheriting split map (D1, D2).

:func:`marinedata.release.generate_split_map` hands its enumerated rows and dHashes here and gets
back what :func:`marinedata.splitmap_inherit.resolve_inherited_splits` applies: the prior
assignments it embeds, the prior-release neighbour band and the out-of-distribution holdouts. Kept
apart so the generator stays readable.

Two priors, first match wins (run-3 triage, PR-1):

1. ``--inherit-tree``: the published v1.0 tree, per image (:mod:`marinedata.splitmap_tree`). A
   group holding a v1.0 image keeps its v1.0 split (only a listed benchmark pin moves it), its
   dHash <= 4 near-copies share its component and adopt that split, and its images anchor the
   5-8 band: a component near a v1.0 train image is never put in an evaluation split.
2. ``--inherit``: a group-keyed map. Only the groups whose component the tree left untouched (no
   v1.0 group, no tree band entry) are embedded, and only their images anchor its band.

Every other component is assigned fresh under the seed.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from ..neardup import NearDupConfig, v1_band_pairs
from ..scan import SplitName
from ..splitmap import SplitMap, SplitMapError
from ..splitmap_inherit import band_constraints, inherit_header
from ..splitmap_tree import TreePrior, V1Tree, straddle_block, tree_prior
from ..splitv2.release_ood import plan_ood, release_samples
from ..splitv2.rules import RulesConfig

Row = tuple[str, str, str]  # image sha256, split group, source id


@dataclass(frozen=True)
class InheritPlan:
    forced: dict[str, SplitName] = field(default_factory=dict)
    """Component -> split, from the prior-release neighbour band (never-eval / never-train)."""
    withhold: dict[str, str] = field(default_factory=dict)
    """Component -> reason: it neighbours prior train and prior evaluation images alike."""
    band_pairs: int = 0
    ood: dict[str, list[str]] = field(default_factory=dict)
    """Component -> the holdout names that matched one of its groups."""
    ood_header: dict[str, object] = field(default_factory=dict)
    prior: SplitMap | None = None
    """The map embedded verbatim: the v1.0 tree's groups plus the map prior's untouched groups."""
    tree: TreePrior | None = None
    tree_header: dict[str, object] = field(default_factory=dict)
    map_header: dict[str, object] = field(default_factory=dict)

    @property
    def straddles(self) -> dict[str, str]:
        """Group -> reason: v1.0 put its own images in different splits (withheld, listed)."""
        return dict(self.tree.straddles) if self.tree is not None else {}

    def inherits(
        self, canonical: Mapping[str, str], moved: Collection[str] = ()
    ) -> dict[str, object] | None:
        """The map's ``inherits`` block with a tree prior (``None``: the map-only header).
        ``moved`` are the components a benchmark pin moved whole."""
        if self.tree is None:
            return None
        block = {
            **self.tree_header,
            "groups": len(self.tree.groups),
            "straddles": straddle_block(self.tree, canonical, moved),
        }
        return {**self.map_header, "v1_tree": block}


def unconfirmed_band_pairs(
    unconfirmed: Iterable[tuple[str, str, int]], prior_split: Mapping[str, SplitName]
) -> list[tuple[str, str, int]]:
    """Rule-B candidates the confirm-hash rejected (``ConfirmedPairs.rejected``) that pair a new
    image with a prior-release one, as band pairs ``(new_sha, prior_sha, hamming)``: closer than
    the band yet unconfirmed, so they keep the band's never-eval / never-train constraint."""
    return [
        (a, b, d) if b in prior_split else (b, a, d)
        for a, b, d in unconfirmed
        if (a in prior_split) != (b in prior_split)
    ]


def _band(
    rows: Sequence[Row],
    anchors: Mapping[str, SplitName],
    *,
    hashes: Mapping[str, int],
    near_dup: NearDupConfig | None,
    unconfirmed: Sequence[tuple[str, str, int]],
    canonical: Mapping[str, str],
) -> tuple[dict[str, SplitName], dict[str, str], int]:
    """``(forced, withhold, pairs)`` of the band around the ``anchors`` (prior images)."""
    if near_dup is None or not anchors:
        return {}, {}, 0
    band = v1_band_pairs(
        {sha: h for sha, h in hashes.items() if sha not in anchors},
        {sha: hashes[sha] for sha in anchors if sha in hashes},
        union_max=near_dup.union_max,
        band_max=near_dup.exclude_max,
    )
    band = sorted([*band, *unconfirmed_band_pairs(unconfirmed, anchors)])
    component_of = {sha: canonical.get(group, group) for sha, group, _ in rows}
    forced, withhold = band_constraints(band, v1_split=anchors, component_of=component_of)
    return forced, withhold, len(band)


def plan_inheritance(
    rows: Sequence[Row],
    *,
    inherit: SplitMap | None,
    never_eval_sources: Collection[str],
    canonical: Mapping[str, str],
    hashes: Mapping[str, int],
    near_dup: NearDupConfig | None,
    ood_rules: RulesConfig | None,
    ood_metadata: Mapping[str, Mapping[str, object]] | None,
    unconfirmed: Iterable[tuple[str, str, int]] = (),
    v1_tree: V1Tree | None = None,
    inherit_sha256: str = "",
    prefer: Mapping[str, SplitName] | None = None,
    ratios: Mapping[SplitName, float] | None = None,
    seed: int = 0,
    now: str = "",
) -> InheritPlan:
    """``rows`` are ``(image_sha256, split_group, source_id)``; ``hashes`` the dHash of every
    eval-capable image (empty without ``near_dup``); ``canonical`` maps a group to its component.
    ``prefer`` (component -> split a benchmark pin asks for) breaks ties between the v1.0 splits a
    sha was published in; ``ratios``/``seed``/``now`` describe the map built from the tree alone.

    Raises ``SplitMapError`` when neither prior is given, or when a constructible OOD holdout
    exceeds the size ceiling: a human narrows the yaml rule, nothing subsamples a holdout."""
    if inherit is None and v1_tree is None:
        raise SplitMapError("an inheriting split map needs --inherit-tree or --inherit")
    pairs = list(unconfirmed)
    band_args = {"hashes": hashes, "near_dup": near_dup, "unconfirmed": pairs}
    forced: dict[str, SplitName] = {}
    withhold: dict[str, str] = {}
    band_pairs = 0
    persisted: dict[str, SplitName] = {}
    touched: set[str] = set()
    tree = None
    if v1_tree is not None:
        tree = tree_prior(rows, v1_tree, canonical, prefer)
        forced, withhold, band_pairs = _band(rows, tree.anchors, canonical=canonical, **band_args)
        touched = {canonical.get(g, g) for g in (*tree.groups, *tree.straddles)}
        touched |= set(forced) | set(withhold)
        persisted = dict(tree.groups)
    map_header: dict[str, object] = {}
    if inherit is not None:
        kept = {g: s for g, s in inherit.assignments.items() if canonical.get(g, g) not in touched}
        prior_split: dict[str, SplitName] = {}
        for sha, group, source in sorted(rows):
            if group in kept and source not in never_eval_sources:
                prior_split.setdefault(sha, kept[group])
        map_forced, map_withhold, map_pairs = _band(
            rows, prior_split, canonical=canonical, **band_args
        )
        forced = {**{c: s for c, s in map_forced.items() if c not in touched}, **forced}
        withhold = {**{c: r for c, r in map_withhold.items() if c not in touched}, **withhold}
        band_pairs += map_pairs
        persisted = {**kept, **persisted}
        if tree is not None:
            map_header = {**inherit_header(inherit, inherit_sha256), "groups_kept": len(kept)}
    ood: dict[str, list[str]] = {}
    header: dict[str, object] = {}
    if ood_rules is not None:
        plan = plan_ood(release_samples(rows, never_eval_sources, ood_metadata), ood_rules)
        if plan.issues:
            raise SplitMapError(
                "OOD holdouts exceed the size guards (narrow the rule in splits/v2.yaml; nothing "
                "subsamples a holdout): " + "; ".join(plan.issues)
            )
        for group, tags in sorted(plan.group_tags.items()):
            names = ood.setdefault(canonical.get(group, group), [])
            names.extend(tag for tag in tags if tag not in names)
        header = plan.header()
    if tree is None:
        prior = inherit
    elif inherit is not None:
        prior = replace(inherit, assignments=persisted)
    else:
        prior = SplitMap(
            by="group",
            seed=seed,
            ratios=dict(ratios or {}),
            assignments=persisted,
            generated_at=now,
            release="v1.0",
            stratify="source",
        )
    return InheritPlan(
        forced=forced,
        withhold=withhold,
        band_pairs=band_pairs,
        ood=ood,
        ood_header=header,
        prior=prior,
        tree=tree,
        tree_header=dict(v1_tree.header) if v1_tree is not None else {},
        map_header=map_header,
    )
