"""Generating the frozen SPLIT_MAP.json: dedup links and the near-duplicate guards."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from ..builder import SUPERVISED_DEFAULT_RATIOS, SplitName
from ..checksums import file_digest
from ..licence_class import release_excluded
from ..neardup import NearDupChainError, NearDupConfig, compute_dhashes, near_dup_record, near_pairs
from ..registry import Registry
from ..splitmap import MergeInfo, SplitMap, load_split_map, resolve_splits, rows_to_counts
from ..splitmap_inherit import resolve_inherited_splits
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
    ood_rules: RulesConfig | None = None,
    ood_metadata: Mapping[str, Mapping[str, object]] | None = None,
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
    images (dHash links chain; the thresholds are policy, never auto-retuned).

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
    hashes: dict[str, int] = {}
    if near_dup is not None:
        eval_shas = sorted({sha for sha, _, source in rows if source not in never_eval_sources})
        hashes = compute_dhashes(
            {sha: paths[sha] for sha in eval_shas},
            cache_dir=near_dup.cache_dir,
            workers=near_dup.workers,
        )
        links = [(a, b) for a, b, _ in near_pairs(hashes, near_dup.union_max)]
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
    plan = (
        plan_inheritance(
            rows,
            inherit=inherit,
            never_eval_sources=never_eval_sources,
            canonical=merge_info.canonical,
            hashes=hashes,
            near_dup=near_dup,
            ood_rules=ood_rules,
            ood_metadata=ood_metadata,
        )
        if inherit is not None
        else None
    )
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

    all_ratios = ratios or dict(SUPERVISED_DEFAULT_RATIOS)
    near_dup_value = {
        **(near_dup_record(near_dup) if near_dup is not None else {}),
        **(near_dup_header or {}),
    }
    inherited = None
    if inherit is None or plan is None:
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
            prior=inherit,
            prior_sha256=inherit_sha256,
            strata=strata or {},
            merge_canonical=merge_info.canonical,
            seed=seed,
            now=now,
            release=release,
            min_groups=min_groups,
            forced=forced,
            withhold=plan.withhold,
            ood=plan.ood,
            ood_header=plan.ood_header,
            near_dup=near_dup_value or None,
        )

    members_of: dict[str, list[str]] = {}
    for group, canon in merge_info.canonical.items():
        members_of.setdefault(canon, []).append(group)
    merged_cross_partition = sum(
        1
        for members in members_of.values()
        if len(members) > 1
        and len({split for g in members for split in upstream_splits.get(g, set())}) > 1
    )

    return SplitMapStats(
        merged_components=merge_info.merged_components,
        merged_cross_partition=merged_cross_partition,
        near_dup_pairs=len(links),
        near_dup_unions=merge_info.near_dup_unions,
        near_dup_max_component=near_dup_max_component,
        upstream_test_groups=len(test_groups),
        upstream_test_components=sum(1 for c in upstream_test_components if forced.get(c) == TEST),
        dedup_group_links=len(dedup_links),
        **(
            {
                "inherited_groups": inherited.inherited_groups,
                "inherit_adopted_groups": inherited.adopted_groups,
                "inherit_conflict_components": inherited.conflict_components,
                "inherit_band_pairs": plan.band_pairs if plan is not None else 0,
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
