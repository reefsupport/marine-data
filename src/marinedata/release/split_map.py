"""Generating the frozen SPLIT_MAP.json: dedup links and the near-duplicate guards."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from ..benchmark_pins import (
    MAP_NAME,
    BenchmarkPins,
    apply_benchmark_pins,
    check_overridden,
    component_pins,
    stamp_pins,
)
from ..builder import SUPERVISED_DEFAULT_RATIOS, SplitName
from ..checksums import file_digest
from ..licence_class import release_excluded
from ..neardup import (
    ConfirmedPairs,
    NearDupChainError,
    NearDupConfig,
    compute_confirm_features,
    compute_dhashes,
    confirm_pairs,
    near_dup_record,
    near_pairs,
)
from ..neardup_phash import (
    PhashSelection,
    phash_confirm_shas,
    phash_exclusions,
    phash_pairs,
    select_phash_links,
)
from ..registry import Registry
from ..splitmap import (
    SCHEMA_VERSION,
    BandHit,
    MergeInfo,
    SplitMap,
    load_split_map,
    resolve_splits,
    rows_to_counts,
    save_split_map,
)
from ..splitmap_inherit import resolve_inherited_splits
from ..splitmap_tree import V1Tree
from ..splitv2.rules import RulesConfig
from ..strata import DEFAULT_MIN_GROUPS, TEST, TRAIN
from ..upstream_split import honoured_test_groups
from .entries import _admitted_source_ids, _never_eval_source_ids, enumerate_release_rows
from .records import SplitMapStats
from .split_map_inherit import plan_inheritance


def dedup_group_links(
    sha_to_group: Mapping[str, str], shas: Iterable[str]
) -> list[tuple[str, str]]:
    """Chain links ``(first, other)`` for every dedup group (``dedup run`` groups.parquet) that has
    two or more images among ``shas``: unioned like a shared digest, so the group is one split
    component. Deterministic: members are sorted, the first is the anchor."""
    members: dict[str, list[str]] = {}
    for sha in set(shas):
        group = sha_to_group.get(sha)
        if group is not None:
            members.setdefault(group, []).append(sha)
    return [
        (ordered[0], other)
        for _, group_members in sorted(members.items())
        for ordered in (sorted(group_members),)
        for other in ordered[1:]
    ]


def near_dup_links(
    hashes: Mapping[str, int], paths: Mapping[str, Path], config: NearDupConfig
) -> ConfirmedPairs:
    """Rule B's links: every dHash pair within ``config.union_max`` (v1.0), or — with
    ``config.union_confirm`` (v1.1.0) — only the candidates the confirm-hash keeps (see
    :class:`~marinedata.neardup.UnionConfirm`). Only images in a candidate pair are decoded."""
    candidates = near_pairs(hashes, config.union_max)
    if config.union_confirm is None:
        return ConfirmedPairs(links=[(a, b) for a, b, _ in candidates], candidates=len(candidates))
    touched = sorted({sha for a, b, _ in candidates for sha in (a, b)})
    features = compute_confirm_features(
        {sha: paths[sha] for sha in touched}, cache_dir=config.cache_dir, workers=config.workers
    )
    return confirm_pairs(candidates, features, config.union_confirm)


def generate_split_map(
    registry: Registry,
    *,
    out: str | Path,
    roots: dict[str, str | Path],
    profile: str = "research",
    ratios: dict[SplitName, float] | None = None,
    seed: int = 0,
    min_groups: int = DEFAULT_MIN_GROUPS,
    now: str | None = None,
    release: str | None = None,
    skipped: dict[str, str] | None = None,
    near_dup: NearDupConfig | None = None,
    honour_upstream_test: bool = True,
    upstream_test_off: Iterable[str] = (),
    near_dup_groups: Mapping[str, str] | None = None,
    near_dup_header: Mapping[str, object] | None = None,
    inherit: SplitMap | None = None,
    inherit_sha256: str = "",
    inherit_tree: V1Tree | None = None,
    ood_rules: RulesConfig | None = None,
    ood_metadata: Mapping[str, Mapping[str, object]] | None = None,
    benchmark_pins: BenchmarkPins | None = None,
) -> SplitMapStats:
    """Enumerate every admitted staged tree in ``roots`` and write a fresh, stratified
    ``SPLIT_MAP.json`` at ``out`` — the "no hand-built TSV" path from staged trees straight
    to a map ``release build`` can then freeze against. ``stratify="source"`` always:
    the map is the registry's cross-source dedup unit, and a stratum with fewer than
    ``min_groups`` groups is train-only (see :mod:`marinedata.strata`).

    Raises if ``out`` already exists — ``resolve_splits`` treats that as extending a map,
    and a release's own map is meant to be generated fresh once, not silently appended
    to under a different corpus. Regenerate deliberately: remove it first.

    ``skipped``, if given, is filled in-place with ``{source_id: reason}`` for every
    admitted source :func:`enumerate_release_rows` left out for not being a
    ``staged-tree`` layout (see there) — the caller's hook for surfacing "this source
    isn't in the release" rather than it disappearing silently.

    Two duplicate-content rules apply before allocation, both WS-D S15c:

    * Groups that share an image digest — same or different source — are merged into
      one connected component and assigned as a unit (see
      :mod:`marinedata.splitmap`'s ``MergeInfo``/``merge_canonical``), rather than
      raising the way ``rows_to_counts`` used to.
    * A component whose only contributing sources are all tagged ``never-eval`` in the
      registry goes straight to ``train`` — no hashing lottery — since a
      machine-generated group must never be eligible for an evaluation split. A mixed
      component (some never-eval, some not) is allocated normally; excluding a
      never-eval row that lands in a non-``train`` split happens later, per row, in
      :func:`build_release`.

    Returns :class:`SplitMapStats` — how many components merged, and how many of those
    also mixed more than one ``upstream_split`` value (report-only upstream-leakage
    signal, from ``upstream_split``, never a blocker).

    ``near_dup`` (WS-D S47 rule B): images of eval-capable (not ``never-eval``) sources
    within ``near_dup.union_max`` dHash Hamming of each other — same or different
    source — are unioned into one component exactly like a shared digest, so a
    re-encoded copy can never land on the other side of a split. Raises
    :class:`~marinedata.neardup.NearDupChainError` before anything is written if the
    largest component a near-dup union formed exceeds ``near_dup.chain_fraction`` of all
    images (dHash links chain; the thresholds are policy, never auto-retuned). With
    ``near_dup.union_confirm`` (v1.1.0) a dHash pair is only a candidate: it is unioned when the
    pair's pHash-256 agree and neither image is flat (:func:`near_dup_links`); a rejected pair
    links nothing, and the rule is recorded in the map's ``near_dup.union_confirm`` header.

    ``honour_upstream_test`` (WP-R2c, default on): every split group holding at least one row
    whose upstream split normalises to ``test`` (:mod:`marinedata.upstream_split`) is forced
    into OUR ``test`` split before the stratified assigner balances the rest, so an upstream
    held-out set is never trained on. The group's whole merged component goes with it (shared
    digest or near-dup union), and a mixed upstream train+test group goes whole. A component of
    only ``never-eval`` sources stays ``train``. Sources without any upstream-test row are
    unaffected; ``upstream_test_off`` lists source ids that opt out (``"*"`` = all sources).

    ``near_dup_groups`` (WP-R6): ``image_sha256 -> dedup group id`` from ``dedup run``'s
    groups.parquet. Every group's images in the enumeration are unioned into one component (all
    sources, never-eval included), so a near-duplicate group has one split and ``dedup gate``
    finds no spanning group. ``near_dup_header`` is recorded under the map's ``near_dup`` header
    (the groups file digest, or the explicit ``--no-near-dup`` opt-out).

    ``inherit`` (D1/D2, a v1.1.0 map from the v1.0 one): the prior release's
    :class:`~marinedata.splitmap.SplitMap` (``inherit_sha256`` = the digest of its bytes). Its
    assignments are embedded verbatim and never reassigned; ``roots`` must hold the prior trees as
    well as the new ones, because the near-duplicate search spans both. New groups are placed
    around them by :func:`~marinedata.splitmap_inherit.resolve_inherited_splits`: a group sharing a
    component with prior groups takes their split (and is withheld when the prior map split those
    apart), and a new image within the near-duplicate band of a prior train image is never-eval,
    of a prior evaluation image never-train. ``ood_rules`` (``splits/v2.yaml``) routes matching
    groups to ``test_ood``, outside the 70/15/15 quota; ``ood_metadata`` supplies the per-image
    geo, depth, platform and time fields. ``inherit=None`` is the unchanged v1.0 behaviour.

    ``benchmark_pins`` (:func:`marinedata.benchmark_pins.load_benchmark_pins`, design 2.2 rule 1):
    the route-to-our-test manifests pin their upstream eval images to ``test`` and the second
    held-out split to ``val`` by sha256, whatever the staged ``upstream_split`` column says. A
    component takes the pin of any member (test beats val). A pin yields to a never-eval train
    component and an OOD claim, and wins over a v1.0 assignment: the whole component moves, and the
    moved v1.0 groups are recorded in the map's ``pins.v1_continuity_overridden`` header block
    (the v1.0 configs stay frozen, so this is a documented v1.0 -> v1.1 split change). The pins are
    counted in the same header. ``None`` is the unchanged behaviour.
    """
    if load_split_map(out) is not None:
        raise ValueError(f"{out} already exists — remove it first to regenerate")
    # One map serves every flavour (WP-R2d): a source released in no flavour never shapes it.
    roots = {s: r for s, r in roots.items() if not release_excluded(s)}
    never_eval_sources = _never_eval_source_ids(
        registry, _admitted_source_ids(registry, roots, profile)
    )
    upstream_splits: dict[str, set[str]] = {}
    upstream_by_source: dict[str, dict[str, set[str]]] = {}
    paths: dict[str, Path] = {}
    rows = list(
        enumerate_release_rows(
            registry,
            roots,
            profile,
            skipped=skipped,
            upstream_splits=upstream_splits,
            upstream_splits_by_source=upstream_by_source,
            paths=paths if near_dup is not None else None,
        )
    )
    links: list[tuple[str, str]] = []
    phash_found: list[tuple[str, str, int]] = []
    phash_selection: PhashSelection | None = None
    prior_shas = frozenset(inherit_tree.splits) if inherit_tree is not None else frozenset()
    hashes: dict[str, int] = {}
    union = ConfirmedPairs(links=[], candidates=0)
    if near_dup is not None:
        eval_shas = sorted({sha for sha, _, source in rows if source not in never_eval_sources})
        hashes = compute_dhashes(
            {sha: paths[sha] for sha in eval_shas},
            cache_dir=near_dup.cache_dir,
            workers=near_dup.workers,
        )
        union = near_dup_links(hashes, paths, near_dup)
        links = list(union.links)
        if near_dup.phash_max is not None:  # the gates' rule: DCT pHash-64 within their radius
            phash_found = phash_pairs(
                {sha: paths[sha] for sha in eval_shas},
                cache_dir=near_dup.cache_dir,
                workers=near_dup.workers,
                max_distance=near_dup.phash_max,
            )
            phash_union = [(a, b) for a, b, _ in phash_found]  # #45: every pair is a link
            if near_dup.union_confirm is not None:  # run 8: unconfirmed links chained whole sites
                single = _single_image_shas(rows)
                need = phash_confirm_shas(phash_found, single=single, prior=prior_shas)
                phash_selection = select_phash_links(
                    phash_found,
                    features=compute_confirm_features(
                        {sha: paths[sha] for sha in sorted(need)},
                        cache_dir=near_dup.cache_dir,
                        workers=near_dup.workers,
                    ),
                    rule=near_dup.union_confirm,
                    single=single,
                    prior=prior_shas,
                )
                phash_union = phash_selection.links
            links = [*links, *phash_union]
    dedup_links: list[tuple[str, str]] = []
    if near_dup_groups is not None:
        dedup_links = dedup_group_links(near_dup_groups, {sha for sha, _, _ in rows})
        links = [*links, *dedup_links]
    counts, strata, merge_info = rows_to_counts(rows, stratified=True, links=links)
    near_dup_max_component = 0
    if near_dup is not None:
        near_dup_max_component = _check_near_dup_chain(counts, merge_info, near_dup)

    group_to_strata: dict[str, set[str]] = {}
    for source_id, groups in (strata or {}).items():
        for group in groups:
            group_to_strata.setdefault(group, set()).add(source_id)
    forced = {
        group: TRAIN
        for group, sources in group_to_strata.items()
        if sources and sources <= never_eval_sources
    }
    prefer = (  # a sha v1.0 published in several splits sits where its benchmark pin asks
        {
            canon: MAP_NAME[pin]
            for canon, pin in component_pins(rows, merge_info.canonical, benchmark_pins).items()
        }
        if inherit_tree is not None and benchmark_pins is not None
        else {}
    )
    plan = (
        plan_inheritance(
            rows,
            inherit=inherit,
            inherit_sha256=inherit_sha256,
            v1_tree=inherit_tree,
            prefer=prefer,
            ratios=ratios or dict(SUPERVISED_DEFAULT_RATIOS),
            seed=seed,
            now=now or "",
            never_eval_sources=never_eval_sources,
            canonical=merge_info.canonical,
            hashes=hashes,
            near_dup=near_dup,
            unconfirmed=union.rejected,
            ood_rules=ood_rules,
            ood_metadata=ood_metadata,
        )
        if inherit is not None or inherit_tree is not None
        else None
    )
    train_only = frozenset(forced)  # never-eval-only components: a pin never moves them
    if plan is not None:  # a prior-release neighbour binds a component; never-eval-only wins
        forced = {**plan.forced, **forced}
    # WP-R2c: a group with an upstream-test row pulls its whole merged component to test.
    # `merge_info.canonical` already joins digest and near-dup components; the never-eval
    # train rule above wins over it, and (D2) so does a test_ood holdout.
    test_groups = honoured_test_groups(
        upstream_by_source, honour=honour_upstream_test, off=upstream_test_off
    )
    upstream_test_components = {merge_info.canonical.get(g, g) for g in test_groups}
    for canon in sorted(upstream_test_components):
        if plan is None or canon not in plan.ood:
            forced.setdefault(canon, TEST)
    # Design 2.2 rule 1: the benchmark manifests pin upstream eval -> test and the second held-out
    # split -> val by sha256, whatever the staged `upstream_split` says. Never-eval train and an OOD
    # claim win over a pin; a pin wins over a v1.0 assignment (`override` below moves the whole
    # component, and `pins.v1_continuity_overridden` records it).
    pins_header: dict[str, object] = {}
    pin_overrides: dict[str, str] = {}
    pin_withheld: dict[str, str] = {}  # pinned components the v1.0 band forces to train
    if benchmark_pins is not None:
        forced, pins_header, pin_overrides, pin_withheld = apply_benchmark_pins(
            forced,
            rows,
            merge_info.canonical,
            benchmark_pins,
            prior=plan.prior.assignments if plan is not None and plan.prior is not None else None,
            ood=plan.ood if plan is not None else (),
            train_only=train_only,
        )

    all_ratios = ratios or dict(SUPERVISED_DEFAULT_RATIOS)
    near_dup_value = {
        **(near_dup_record(near_dup) if near_dup is not None else {}),
        **(near_dup_header or {}),
    }
    inherited = None
    if plan is None or plan.prior is None:
        resolve_splits(
            out,
            counts,
            all_ratios,
            seed=seed,
            by="group",
            now=now,
            release=release,
            strata=strata,
            stratify="source",
            min_groups=min_groups,
            merge_canonical=merge_info.canonical,
            forced=forced,
            near_dup=near_dup_value or None,
        )
    else:
        inherited = resolve_inherited_splits(
            out,
            counts,
            all_ratios,
            prior=plan.prior,
            prior_sha256=inherit_sha256,
            strata=strata or {},
            merge_canonical=merge_info.canonical,
            seed=seed,
            now=now,
            release=release,
            min_groups=min_groups,
            forced=forced,
            withhold=pin_withheld,
            band_excluded=plan.band_excluded,
            ood=plan.ood,
            ood_header=plan.ood_header,
            near_dup=near_dup_value or None,
            override=pin_overrides,
            straddles=plan.straddles,
            inherits=plan.inherits(merge_info.canonical, pin_overrides),
        )
        if pins_header:
            check_overridden(pins_header, inherited.overridden)

    members_of: dict[str, list[str]] = {}
    for group, canon in merge_info.canonical.items():
        members_of.setdefault(canon, []).append(group)
    merged_cross_partition = sum(
        1
        for members in members_of.values()
        if len(members) > 1
        and len({split for g in members for split in upstream_splits.get(g, set())}) > 1
    )

    if pins_header:
        stamp_pins(out, pins_header)
    phash_excluded, phash_frozen = _exclude_phash_pairs(out, rows, phash_selection, prior_shas)
    return SplitMapStats(
        pins=pins_header or None,
        merged_components=merge_info.merged_components,
        merged_cross_partition=merged_cross_partition,
        near_dup_pairs=len(links),
        near_dup_unions=merge_info.near_dup_unions,
        near_dup_max_component=near_dup_max_component,
        near_dup_candidates=union.candidates,
        near_dup_rejected_phash=union.rejected_phash,
        near_dup_rejected_entropy=union.rejected_entropy,
        near_dup_phash_pairs=len(phash_found),
        near_dup_phash_unioned=len(phash_selection.links) if phash_selection else len(phash_found),
        near_dup_phash_prior_pairs=phash_selection.prior_pairs if phash_selection else 0,
        phash_excluded_images=len(phash_excluded),
        phash_frozen_pairs=phash_frozen,
        upstream_test_groups=len(test_groups),
        upstream_test_components=sum(1 for c in upstream_test_components if forced.get(c) == TEST),
        dedup_group_links=len(dedup_links),
        **(
            {
                "inherited_groups": inherited.inherited_groups,
                "inherit_adopted_groups": inherited.adopted_groups,
                "inherit_conflict_components": inherited.conflict_components,
                "inherit_band_pairs": plan.band_pairs if plan is not None else 0,
                "inherit_band_excluded": inherited.band_excluded,
                "withheld_groups": len(inherited.withheld),
                "ood_components": inherited.ood_components,
                "ood_groups": inherited.ood_groups,
                "ood_blocked_components": inherited.ood_blocked,
            }
            if inherited is not None
            else {}
        ),
    )


def _check_near_dup_chain(
    counts: dict[str, int], merge_info: MergeInfo, config: NearDupConfig
) -> int:
    """Largest near-dup-formed component in images; raise if it breaks the chain guard,
    with the size distribution and the biggest components' member groups as evidence."""
    sizes = sorted(
        ((counts[canon], canon) for canon in merge_info.near_dup_canonicals),
        key=lambda item: (-item[0], item[1]),
    )
    largest = sizes[0][0] if sizes else 0
    total = sum(counts.values())
    if largest <= config.chain_fraction * total:
        return largest
    members: dict[str, list[str]] = {}
    for group, canon in merge_info.canonical.items():
        members.setdefault(canon, []).append(group)
    edges = (1, 2, 5, 20, 100, 1000)
    histogram = []
    for lo, hi in zip(edges, (*edges[1:], None), strict=True):
        n = sum(1 for size, _ in sizes if size > lo and (hi is None or size <= hi))
        histogram.append(f"({lo},{hi if hi is not None else 'inf'}]={n}")
    examples = "; ".join(
        f"{canon} size={size} groups={len(members[canon])} e.g. {sorted(members[canon])[:5]}"
        for size, canon in sizes[:3]
    )
    raise NearDupChainError(
        f"near-dup chain guard: largest near-dup-merged component has {largest} images "
        f"> {config.chain_fraction:.2%} of {total}; top sizes {[s for s, _ in sizes[:10]]}; "
        f"components={len(sizes)} sizes {' '.join(histogram)}; examples: {examples}"
    )


def _never_eval_near_dups(
    registry: Registry,
    roots: dict[str, Path],
    profile: str,
    never_eval_sources: set[str],
    config: NearDupConfig,
    *,
    digest: Callable[[Path], str] = file_digest,
) -> list[str]:
    """Sorted sha256 of never-eval images within ``config.exclude_max`` of any
    eval-capable admitted image (rule A). Enumerates the same staged-tree rows the split
    map is generated from, so every admitted row counts, whatever split or task."""
    paths: dict[str, Path] = {}
    never_eval: set[str] = set()
    eval_capable: set[str] = set()
    for sha, _, source_id in enumerate_release_rows(
        registry, roots, profile, paths=paths, digest=digest
    ):
        (never_eval if source_id in never_eval_sources else eval_capable).add(sha)
    if not never_eval or not eval_capable:
        return []
    hashes = compute_dhashes(
        {sha: paths[sha] for sha in sorted(never_eval | eval_capable)},
        cache_dir=config.cache_dir,
        workers=config.workers,
    )
    pairs = near_pairs(
        {sha: hashes[sha] for sha in never_eval},
        config.exclude_max,
        {sha: hashes[sha] for sha in eval_capable},
    )
    return sorted({sha for sha, _, _ in pairs})


def _single_image_shas(rows: list[tuple[str, str, str]]) -> frozenset[str]:
    """Images alone in every split group they belong to: a union between two of them moves two
    images, never a site (the pHash-64 union's second clause)."""
    members: dict[str, set[str]] = {}
    groups_of: dict[str, set[str]] = {}
    for sha, group, _ in rows:
        members.setdefault(group, set()).add(sha)
        groups_of.setdefault(sha, set()).add(group)
    return frozenset(
        sha for sha, groups in groups_of.items() if all(len(members[g]) == 1 for g in groups)
    )


def _image_splits(
    rows: list[tuple[str, str, str]], split_map: SplitMap
) -> dict[str, frozenset[str]]:
    """``image_sha256 -> the splits it ships in``: its groups' splits, unless the map withholds the
    group or band-excludes the image."""
    splits: dict[str, set[str]] = {}
    for sha, group, _ in rows:
        split = split_map.assignments.get(group)
        if split is None or group in split_map.withheld or sha in split_map.band_excluded:
            continue
        splits.setdefault(sha, set()).add(split)
    return {sha: frozenset(found) for sha, found in splits.items()}


def _exclude_phash_pairs(
    out: str | Path,
    rows: list[tuple[str, str, str]],
    selection: PhashSelection | None,
    prior: frozenset[str],
) -> tuple[dict[str, BandHit], int]:
    """Drop one image of every pHash-64 pair the written map ships across splits and record it
    under the map's ``phash_excluded`` (schema 2); one pass, see
    :func:`~marinedata.neardup_phash.phash_exclusions`. Returns the exclusions and the count of
    frozen prior-prior pairs."""
    written = load_split_map(out) if selection is not None else None
    if written is None or selection is None:
        return {}, 0
    excluded, frozen = phash_exclusions(
        selection.pairs, _image_splits(rows, written), protected=prior
    )
    if excluded:
        save_split_map(
            out,
            replace(
                written,
                schema_version=max(written.schema_version, SCHEMA_VERSION),
                phash_excluded=excluded,
            ),
        )
    return excluded, frozen
