"""Release driver — ``marinedata release build`` (WS-D S7k).

Loads every admitted source already resolved to a local ``roots`` path, builds every
task declared under ``registry/tasks/*.yaml`` against one frozen ``SPLIT_MAP.json``, and
writes the resulting per-task split manifests plus a ``RELEASE.json`` under
``<out>/releases/<id>/``. Nothing is uploaded.

Resolving ``roots`` (fetching a pinned staged tree per admitted source, or accepting a
local override for a tree that has not been uploaded yet) is the CLI's job — see
:mod:`marinedata.cli_release` — so this module stays pure and directly testable: hand it
a registry and a ``roots`` mapping and it never touches the network.

Splitting under a frozen map, not allocating one, is the point: ``frozen=True`` raises if
a group an admitted source contributes is missing from the map (see
:mod:`marinedata.splitmap`), so a release can never silently grow the shared map or
disagree with a split some other task, or a previous release, already committed to.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from .builder import SUPERVISED_DEFAULT_RATIOS, DatasetBuilder, PartialAbstainExclusion, SplitName
from .checksums import file_digest
from .flavours import check_profile, flavour_source_ids, release_record, sample_ships
from .gate import evaluate
from .labelcheck import release_label_gate
from .licence_class import release_excluded, require_flavour, spec_for
from .loaders.generic import IMAGE_SUFFIXES
from .metadata_norm.base import NormContext
from .metadata_norm.default import attribution_text
from .metadata_norm.split_group import derive_split_group
from .models import Source
from .neardup import (
    NearDupChainError,
    NearDupConfig,
    compute_dhashes,
    near_dup_record,
    near_pairs,
)
from .registry import Registry
from .sample_schema import staged_partition
from .splitmap import MergeInfo, Row, load_split_map, resolve_splits, rows_to_counts
from .strata import DEFAULT_MIN_GROUPS, TEST, TRAIN
from .tables import _require_pyarrow
from .upstream_split import honoured_test_groups

logger = logging.getLogger(__name__)

GUARD_MIN_ROWS = 100
"""A source with at least this many rows ..."""
GUARD_MIN_GROUPS = 10
"""... and fewer final split groups than this falls back to one group per image (WP-R2d)."""

DEFAULT_SCHEMA_ID = "rs-benthic-v1"


@dataclass(frozen=True)
class TaskManifest:
    """One task's split manifest — sha256/split pairs, sorted for a stable file."""

    task_id: str
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReleaseResult:
    release: str
    out_dir: Path
    sources: tuple[str, ...]
    tasks: tuple[TaskManifest, ...]
    skipped_tasks: tuple[tuple[str, str], ...]
    """``(task_id, reason)`` for a task with no permitted source to build from — not an
    error: a partial ``roots`` (a dry run, or a source temporarily unfetchable) simply
    leaves that task out of this release rather than failing the whole build."""
    never_eval_excluded: int = 0
    """Rows dropped from a non-``train`` split manifest because their source carries the
    ``never-eval`` tag (WS-D S15c) — a mixed merged component (see
    :mod:`marinedata.splitmap`) lands where its non-never-eval members send it, but a
    machine-generated row must never enter an evaluation split. Counted, never moved to
    ``train`` — that would leak the same content the split was built to hold out."""
    partial_abstain_excluded: tuple[PartialAbstainExclusion, ...] = ()
    """Sources dropped entirely from one task's axis (WS-D S26) because their own native
    labels were a mix of resolved and coarser-abstaining — see
    ``Dataset.partial_abstain_excluded``."""
    never_eval_near_dup_excluded: tuple[str, ...] = ()
    """Sorted sha256 of every never-eval image within ``exclude_max`` dHash Hamming of
    any eval-capable admitted row (WS-D S47 rule A) — dropped from every split of every
    task, train included: a re-encoded twin of an evaluation image must not train."""
    never_eval_near_dup_rows: int = 0
    """Task-manifest rows those images would have contributed, summed over tasks."""
    task_layer_configs: dict[str, int] = field(default_factory=dict)
    """``{config_id: n_images}`` for the 5 WP-8c v2 configs — empty unless ``tasks="v2"``
    was passed to :func:`build_release` (D-X: v1 default leaves this empty, always)."""


@dataclass(frozen=True)
class SplitMapStats:
    """What :func:`generate_split_map` did beyond writing the file — for the CLI/build
    result to report (WS-D S15c)."""

    merged_components: int
    """Connected components of 2+ ``split_group`` values merged for sharing an image
    digest — see :class:`marinedata.splitmap.MergeInfo`."""
    merged_cross_partition: int
    """Of those, how many combined members whose rows carry more than one distinct
    ``upstream_split`` value — report-only evidence of upstream leakage, never a
    blocker."""
    near_dup_pairs: int = 0
    """Eval-capable image pairs within the union threshold (WS-D S47 rule B)."""
    near_dup_unions: int = 0
    """Of those pairs, how many joined two still-separate components."""
    near_dup_max_component: int = 0
    """Images in the largest component a near-dup union formed — the chain guard's input."""
    upstream_test_groups: int = 0
    """Raw split groups holding an upstream-test row that the "honour upstream test" rule
    (WP-R2c) pulled to ``test``; their merged components are counted below."""
    upstream_test_components: int = 0
    """Merged components forced to ``test`` by that rule (a never-eval-only one stays train)."""
    dedup_group_links: int = 0
    """Links added from the ``dedup run`` groups (WP-R6): every image of a dedup group in the
    enumeration is tied to the group's first image, so a near-dup group never straddles splits."""


def _never_eval_source_ids(registry: Registry, source_ids: Iterable[str]) -> set[str]:
    """Admitted sources tagged ``never-eval`` in the registry (WS-D S15c) — their rows
    may only ever land in the ``train`` split."""
    return {
        source_id for source_id in source_ids if "never-eval" in registry.source(source_id).tags
    }


def _admitted_source_ids(
    registry: Registry, roots: dict[str, str | Path], profile: str
) -> list[str]:
    """Sources with a resolved root that also pass the profile's licence gate.

    ``roots`` already answers "has a pinned staged tree" (the caller's job — see the
    module docstring); this only re-applies the licence gate, so a caller cannot smuggle
    a denied source in by handing it a root directly.
    """
    prof = registry.profile(profile)
    admitted = []
    for source_id in sorted(roots):
        source = registry.source(source_id)
        if evaluate(source, prof).allowed:
            admitted.append(source_id)
    return admitted


def _partition_stem_index(partition_dir: Path) -> dict[str, list[Path]]:
    """``{stem: sorted candidate paths}`` for every file directly under ``partition_dir``,
    built with one directory scan rather than one ``glob(f"{stem}.*")`` per row.

    Keyed by ``Path(name).stem`` (split on the *last* dot) rather than a first-dot split:
    staging always appends exactly one extension to a stem that may itself contain dots
    (``Path(...).stem`` at staging time — see ``fetchers_remote``/``ingest_image_labels``),
    so this reconstructs the same key the glob-per-row pattern ``f"{stem}.*"`` matched
    against. A stem with two candidate files (e.g. ``a0.jpg`` and ``a0.png``) keeps the
    same tie-break as before: the caller sorts and takes the first.
    """
    index: dict[str, list[Path]] = {}
    if partition_dir.is_dir():
        for path in partition_dir.iterdir():
            if path.is_file():
                index.setdefault(path.stem, []).append(path)
        for paths in index.values():
            paths.sort()
    return index


def _release_group(source: Source, record: dict, *, stem: str, partition: str, sha256: str) -> str:
    """One row's split group at release time (never written back to the staged tree).

    The staged ``split_group`` column wins. Else a source with a registry pattern uses it.
    A source with NO pattern never gets the single-group ``<source>/<partition>`` default
    (WP-R2d): it takes the ``metadata_norm`` chain (spatio-temporal, row ids, upstream split
    + folder, sha256), computed here so a pinned tree with a null column needs no re-stage.
    """
    group = record.get("split_group")
    if group:
        return group
    upstream_path = str(record.get("upstream_path") or "")
    if source.split_group.pattern is not None:
        return source.split_group_for(stem=stem, upstream_path=upstream_path, partition=partition)
    values = {**record, "stem": stem, "upstream_path": upstream_path, "image_sha256": sha256}
    return derive_split_group(source.id, record, values, None)[0]


def _guard_degenerate_grouping(
    source_id: str, entries: list[tuple[str, str, str | None, Path]]
) -> list[tuple[str, str, str | None, Path]]:
    """A source with >= ``GUARD_MIN_ROWS`` rows but < ``GUARD_MIN_GROUPS`` final groups cannot
    be split sanely (a lottery over 1-2 groups gives 0% or 100% test): warn and group per
    image (``<source>/sha:<sha256>``) instead."""
    n_groups = len({group for _, group, _, _ in entries})
    if len(entries) < GUARD_MIN_ROWS or n_groups >= GUARD_MIN_GROUPS:
        return entries
    logger.warning(
        "%s: %d rows fall into only %d split group(s) (< %d); grouping per image (sha256) instead",
        source_id,
        len(entries),
        n_groups,
        GUARD_MIN_GROUPS,
    )
    return [(sha, f"{source_id}/sha:{sha}", split, path) for sha, _, split, path in entries]


WALK_LAYOUTS = ("flat-images", "image-mask-pairs")
"""Layouts whose image sub-directory defaults to ``images`` (pairs) or the root when the tree has
no staged ``metadata.parquet``. Since WP-R2f every layout is enumerated
(:func:`source_release_entries`). A pair's mask/depth file is never its own row, so a pair always
shares its primary image's group."""


def source_attribution(source: Source) -> str | None:
    """Attribution for ``source`` from the metadata_norm default normaliser: the ingest-spec
    attribution + registry citation + upstream URL, joined; ``None`` when all are empty."""
    spec = spec_for(source.id) or {}
    ctx = NormContext(
        attribution=str(spec.get("attribution") or ""),
        citation=source.citation or "",
        homepage=source.homepage or "",
    )
    return attribution_text(ctx)


def release_skip_reason(source: Source) -> str | None:
    """Why the split-map enumerator leaves ``source`` out, or ``None`` when it is enumerated.

    The split map assigns IMAGES, so it never depends on the label layout (WP-R2f): every
    ``loader.layout`` is enumerated, from the staged ``metadata.parquet`` or by walking the image
    directory. Skipped only for an explicit reason: a registry ``release_skip_reason``, or a
    ``needs-attribution`` source that still has no attribution after the metadata_norm default
    (registry citation + upstream URL). :func:`build_release` asks the same question, so it never
    roots a source the map cannot cover (WP-R2d/R2e)."""
    if source.release_skip_reason:
        return f"registry: {source.release_skip_reason}"
    if "needs-attribution" in source.tags and source_attribution(source) is None:
        return "needs-attribution: no attribution (registry citation, upstream URL) to take"
    return None


@dataclass(frozen=True)
class ReleaseEntry:
    """One primary image of a source as the split map sees it."""

    sha256: str
    group: str
    upstream_split: str | None
    path: Path
    key: str
    """Path relative to the source root — what ``Sample.key`` holds for a loader-read sample."""


def _staged_meta(path: Path) -> bool:
    """True when ``path`` is a staged sample index (``stem`` + ``partition`` or ``image_path``)."""
    if not path.is_file():
        return False
    import pyarrow.parquet as pq

    names = set(pq.read_schema(path).names)
    return "stem" in names and bool(names & {"partition", "image_path"})


def _staged_entries(
    source: Source, root: Path, digest: Callable[[Path], str]
) -> list[ReleaseEntry]:
    import pyarrow.parquet as pq

    metadata_path = root / "metadata.parquet"
    if not metadata_path.is_file():
        raise ValueError(
            f"{source.id}: no metadata.parquet under {root} — the release enumerator "
            "reads the staging pipeline's sample index, independent of the source's "
            "own loader layout"
        )
    partition_indexes: dict[str, dict[str, list[Path]]] = {}
    entries: list[tuple[str, str, str | None, Path]] = []
    for record in pq.read_table(metadata_path).to_pylist():
        partition, stem = staged_partition(record), record["stem"]
        index = partition_indexes.get(partition)
        if index is None:
            index = _partition_stem_index(root / "images" / partition)
            partition_indexes[partition] = index
        matches = index.get(stem, [])
        if not matches:
            raise ValueError(
                f"{source.id}: metadata.parquet references image {stem!r} (partition "
                f"{partition!r}) not found under {root / 'images' / partition}"
            )
        sha256 = digest(matches[0])
        group = _release_group(source, record, stem=stem, partition=partition, sha256=sha256)
        entries.append((sha256, group, record.get("upstream_split") or None, matches[0]))
    entries = _guard_degenerate_grouping(source.id, entries)
    return [ReleaseEntry(sha, g, u, p, str(p.relative_to(root))) for sha, g, u, p in entries]


def _walk_entries(source: Source, root: Path, digest: Callable[[Path], str]) -> list[ReleaseEntry]:
    """Primary images of an unstaged tree of any layout, grouped with the
    same chain as a staged row (registry pattern, else the metadata_norm chain: the upstream
    path is the only per-row evidence, so it usually ends at the image sha256)."""
    params = source.loader.params if source.loader is not None else {}
    layout = source.loader.layout if source.loader is not None else ""
    sub = params.get("images_dir", "images" if layout == "image-mask-pairs" else "")
    base = root / str(sub) if sub else root
    entries: list[tuple[str, str, str | None, Path]] = []
    for path in sorted(base.rglob("*")) if base.is_dir() else ():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            sha256 = digest(path)
            record = {"upstream_path": str(path.relative_to(root))}
            group = _release_group(source, record, stem=path.stem, partition="", sha256=sha256)
            entries.append((sha256, group, None, path))
    entries = _guard_degenerate_grouping(source.id, entries)
    return [ReleaseEntry(sha, g, u, p, str(p.relative_to(root))) for sha, g, u, p in entries]


def source_release_entries(
    source: Source, root: str | Path, digest: Callable[[Path], str] = file_digest
) -> list[ReleaseEntry]:
    """Every primary image of ``source`` with its release group: the one enumeration the split
    map generator and the build's group hook share, so they cannot disagree."""
    root = Path(root)
    layout = source.loader.layout if source.loader is not None else None
    if layout == "staged-tree" or _staged_meta(root / "metadata.parquet"):
        return _staged_entries(source, root, digest)
    return _walk_entries(source, root, digest)


def enumerate_release_rows(
    registry: Registry,
    roots: dict[str, str | Path],
    profile: str = "research",
    *,
    skipped: dict[str, str] | None = None,
    upstream_splits: dict[str, set[str]] | None = None,
    upstream_splits_by_source: dict[str, dict[str, set[str]]] | None = None,
    paths: dict[str, Path] | None = None,
    digest: Callable[[Path], str] = file_digest,
) -> Iterator[Row]:
    """``(image_sha256, split_group, stratum)`` for every admitted source's staged tree.

    ``stratum`` is always the source id — the enumerator a release needs to generate a
    stratified ``SPLIT_MAP.json`` without a hand-built TSV (the WS-D 7l scratch
    ``build_tsv.py`` did this once, read-only, outside the repo; this is that made real).

    Reads each source's ``metadata.parquet`` directly rather than through its registered
    loader: ``split_group`` is a staging-time property (D-group, WS-D step 3), recorded
    for every staged tree regardless of what layout the source's own loader later
    declares for training reads (e.g. ``labelbox-rgb`` reads stitched masks the
    staged-tree convention never touches, but every staged tree still carries the same
    sample index). Going through a per-annotation-kind loader here would demand
    layout-specific params this enumerator has no business needing.

    ``upstream_splits``, if given, is filled in-place with ``{split_group: {upstream_split,
    ...}}`` — every distinct, non-empty ``upstream_split`` value seen for a group, keyed
    by the *raw* (pre-merge) group id. :func:`generate_split_map` uses it to flag a
    merged component whose members carry more than one upstream partition — report-only
    evidence of upstream leakage (WS-D S15c), not a blocker.

    ``upstream_splits_by_source``, if given, is filled the same way but one level deeper,
    ``{source_id: {split_group: {upstream_split, ...}}}`` — the per-source input the
    "honour upstream test" rule needs so one source can opt out (WP-R2c).

    ``paths``, if given, is filled in-place with ``{image_sha256: file}`` (first file seen
    for each digest) — what the near-duplicate check (WS-D S47) hashes.

    A source whose ``loader.layout`` is not ``staged-tree`` (e.g. ``metadata-only``:
    v3i's bespoke ``path``/``class``/``upstream_split``/``width``/``height``/
    ``split_group`` schema, no ``partition``/``stem`` columns at all) is *not* a
    staged-tree sample index and is skipped rather than crashing on the missing
    columns — whether classification sources like these join a release is still
    Yohan's open question (WS-D S12b/S12c), so until it is answered they are left
    out, never silently dropped: pass ``skipped`` to collect ``{source_id: reason}``
    for the caller to report. The guard checks the registry's declared layout, not
    a caught ``KeyError`` — a *staged-tree* source missing ``partition`` is real
    corruption and must still raise (below, unchanged).

    A source tagged ``needs-attribution`` (WS-D S23) is skipped the same way, and for
    the same reason a denied-licence source never reaches here at all: its citation is
    still unconfirmed, so shipping its rows in a release would attribute a real
    creator's work incorrectly rather than not at all. Checked before the layout guard
    so a ``needs-attribution`` source gets this specific reason even once it is staged
    as a ``staged-tree`` — until now the tag was declared in the registry but never
    read anywhere in ``src/`` (``grep -rn needs-attribution src`` = 0).
    """
    _require_pyarrow()

    for source_id in _admitted_source_ids(registry, roots, profile):
        source = registry.source(source_id)
        reason = release_skip_reason(source)
        if reason is not None:
            if skipped is not None:
                skipped[source_id] = reason
            continue
        for entry in source_release_entries(source, roots[source_id], digest):
            if entry.upstream_split:
                if upstream_splits is not None:
                    upstream_splits.setdefault(entry.group, set()).add(entry.upstream_split)
                if upstream_splits_by_source is not None:
                    by_group = upstream_splits_by_source.setdefault(source_id, {})
                    by_group.setdefault(entry.group, set()).add(entry.upstream_split)
            if paths is not None:
                paths.setdefault(entry.sha256, entry.path)
            yield entry.sha256, entry.group, source_id


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
    # WP-R2c: a group with an upstream-test row pulls its whole merged component to test.
    # `merge_info.canonical` already joins digest and near-dup components; the never-eval
    # train rule above wins over it.
    test_groups = honoured_test_groups(
        upstream_by_source, honour=honour_upstream_test, off=upstream_test_off
    )
    upstream_test_components = {merge_info.canonical.get(g, g) for g in test_groups}
    for canon in sorted(upstream_test_components):
        forced.setdefault(canon, TEST)

    resolve_splits(
        out,
        counts,
        ratios or dict(SUPERVISED_DEFAULT_RATIOS),
        seed=seed,
        by="group",
        now=now,
        release=release,
        strata=strata,
        stratify="source",
        min_groups=min_groups,
        merge_canonical=merge_info.canonical,
        forced=forced,
        near_dup=(
            {
                **(near_dup_record(near_dup) if near_dup is not None else {}),
                **(near_dup_header or {}),
            }
            or None
        ),
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
        upstream_test_components=sum(1 for c in upstream_test_components if forced[c] == TEST),
        dedup_group_links=len(dedup_links),
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


class ReleaseSkipError(ValueError):
    """A releasable source with rows would be silently left out of a flavour build (WP-R2e)."""


def _loader_layout(source: Source) -> str | None:
    return source.loader.layout if source.loader is not None else None


def _memoized(digest: Callable[[Path], str]) -> Callable[[Path], str]:
    """``digest`` computed once per path: the group hook, the near-dup check and the manifest
    loop all hash the same files, and a 26 GB source must not be read three times."""
    seen: dict[Path, str] = {}

    def cached(path: Path) -> str:
        key = Path(path)
        if key not in seen:
            seen[key] = digest(key)
        return seen[key]

    return cached


def _with_release_group(  # type: ignore[no-untyped-def]
    sample, groups: dict[str, dict[str, str]], roots: dict[str, Path]
):
    """``sample`` with ``meta["split_group"]`` set from the enumeration the split map was
    generated from (a loader that is not ``staged-tree`` never sets one). A pair is one
    sample: its mask/depth file rides along with the primary image's group. The lookup is by
    image path relative to the source root, because ``Sample.key`` is layout-specific (a COCO
    ``file_name`` or a CSV image name is relative to ``images_dir``, not to the root); the key
    stays the fallback."""
    by_path = groups.get(sample.source_id)
    if by_path is None or sample.meta.get("split_group"):
        return sample
    group = None
    if sample.image is not None and sample.source_id in roots:
        try:
            group = by_path.get(str(Path(sample.image).relative_to(roots[sample.source_id])))
        except ValueError:
            group = None
    if group is None:
        group = by_path.get(sample.key)
    return sample if group is None else replace(sample, meta={**sample.meta, "split_group": group})


def _has_rows(root: Path) -> bool:
    """True when the staged tree holds at least one row (``metadata.parquet``) or, without one,
    at least one image file."""
    index = root / "metadata.parquet"
    if index.is_file():
        import pyarrow.parquet as pq

        return pq.ParquetFile(index).metadata.num_rows > 0
    return root.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in root.rglob("*")
    )


def _skipped_sources(
    registry: Registry,
    admitted: list[str],
    roots: dict[str, str | Path],
    allow_skip: frozenset[str],
) -> dict[str, tuple[str, str]]:
    """``{source_id: (reason, allowed_by)}`` for every admitted source the build leaves out.

    The hard gate (WP-R2e): a source the split map cannot cover that has rows > 0 raises unless
    ``--allow-skip`` names it or its registry carries ``release_skip_reason``; a source with no
    rows is recorded (``allowed_by`` = ``no-rows``, reason ``no-staged-images`` when the split map
    would have covered it) and never blocks."""
    skipped: dict[str, tuple[str, str]] = {}
    blocked: list[str] = []
    for sid in admitted:
        source = registry.source(sid)
        reason = release_skip_reason(source)
        if sid in allow_skip:
            skipped[sid] = (reason or "--allow-skip: skipped on request", "allow-skip")
        elif reason is None:
            if not _has_rows(Path(roots[sid])):  # WP-R4: no rows to build, so nothing to judge
                skipped[sid] = (f"no-staged-images: no staged rows under {roots[sid]}", "no-rows")
            continue
        elif source.release_skip_reason:
            skipped[sid] = (reason, "registry")
        elif not _has_rows(Path(roots[sid])):
            skipped[sid] = (reason, "no-rows")
        else:
            blocked.append(sid)
    if blocked:
        raise ReleaseSkipError(
            "releasable source(s) with rows the split map cannot cover would be silently "
            "dropped: "
            + "; ".join(f"{sid} ({release_skip_reason(registry.source(sid))})" for sid in blocked)
            + " -- cover them, set a registry release_skip_reason, or pass --allow-skip SOURCE_ID"
        )
    return skipped


def build_release(
    registry: Registry,
    *,
    release: str,
    split_map: str | Path,
    roots: dict[str, str | Path],
    out_dir: str | Path,
    profile: str = "research",
    allow_unmapped: bool = False,
    near_dup: NearDupConfig | None = None,
    dedup_v2_groups: str | Path | None = None,
    decon: bool = False,
    dedup_crop: bool = False,
    split_v2: bool = False,
    tasks: str = "v1",
    v2: bool = False,
    tasklabels_root: str | Path | None = None,
    digest: Callable[[Path], str] = file_digest,
    flavour: str | None = None,
    allow_skip: Iterable[str] = (),
) -> ReleaseResult:
    """Build every registry task against a frozen split map and write the release.

    ``flavour`` (WP-L1b, ``open`` | ``nc``): ship only that flavour's sources, under one of
    its shipping profiles, into ``<out_dir>/releases/<release>/<flavour>/``; RELEASE.json
    records the flavour, class counts and every excluded source. ``None`` (default) is the
    unfiltered v1 build, byte-identical to before.

    ``tasks`` (WP-8c, charter D-X): ``"v1"`` (default) never touches the task-layer
    configs below and produces byte-identical output to before this parameter existed.
    ``"v2"`` additionally builds the 5 WP-8 task-layer configs (``points``, ``vqa``,
    ``semseg``, ``benthic-coarse``, ``benthic-cover`` — see
    :mod:`marinedata.task_layers.configs`) from ``<tasklabels_root>/_tasklabels/**``
    (D-Z2 producers) and writes them as parquet under
    ``<out_dir>/releases/<release>/task_layers/``. Additive only: no v1 file changes
    shape or content when ``tasks="v2"`` is passed, so v1 byte-identity holds regardless
    of this flag. ``tasklabels_root`` is the directory holding ``_tasklabels/`` (default:
    ``<repo>/data``, derived from the registry's location, never the cwd — INT-core3c);
    see :func:`marinedata.task_layers.configs.resolve_tasklabels_root`.

    ``decon`` (WP-12 P2) and ``dedup_crop`` (WP-10c) are independent switches: the latter
    only changes decon's S5 patch/crop stage when ``decon=True`` (INT-core2, D-T2).
    ``split_v2`` (WP-11/12 P3, INT-core2) validates the split-v2 config — ``registry/
    splits/v2.yaml`` plus ``registry/benchmarks.yaml`` load and hash cleanly — and records
    the hashes in ``RELEASE.json``. The full per-sample split-v2 gate needs WP-2's
    per-sample geo columns, not yet in ``metadata.parquet`` (``docs/split-v2-dry-run.md``),
    so it is deferred to the v2 build; this switch is a config-only smoke check, off by
    default, that changes nothing when off (D-X).

    ``v2=True`` is a preset (INT-core2): it turns on ``decon``, ``dedup_crop``,
    ``split_v2`` and ``tasks="v2"`` together. It does not turn on the WP-10
    ``dedup_v2_groups`` split-leak gate, which needs an explicit groups.parquet path.
    Default off, so passing no flags stays byte-identical to before this preset existed.

    Raises if ``split_map`` does not exist yet (a release never allocates one — see
    :mod:`marinedata.splitmap`), or if any admitted source contributes a group absent
    from it (``SplitMapError``, propagated from ``Dataset.split(frozen=True)``) — both
    left uncaught deliberately, since either means the release is not reproducible yet.

    ``near_dup`` (WS-D S47 rule A): every image of a ``never-eval`` source within
    ``near_dup.exclude_max`` dHash Hamming of any row of an eval-capable admitted source
    (any split) is dropped from every task manifest, and recorded in ``RELEASE.json`` as
    a count plus the sorted sha256 list, alongside the check's own parameters.

    Hard gate (WP-R2e, flavour builds): a releasable source with rows > 0 that the split map
    cannot cover FAILS the build, unless it is named in ``allow_skip`` (``--allow-skip``) or
    carries a registry ``release_skip_reason``. Every skipped source is recorded, with its
    reason and what allowed it, under ``skipped_sources`` in RELEASE.json.
    """
    digest = _memoized(digest)
    if v2:
        decon = True
        dedup_crop = True
        split_v2 = True
        tasks = "v2"
    tasks_mode = tasks
    split_map_path = Path(split_map)
    if load_split_map(split_map_path) is None:
        raise ValueError(
            f"{split_map_path} does not exist — generate it first with "
            "`marinedata splitmap generate`."
        )

    require_flavour(flavour, "build_release")
    if flavour is not None:
        check_profile(flavour, profile)
    admitted = _admitted_source_ids(registry, roots, profile)
    if flavour is not None:
        admitted = flavour_source_ids(registry, admitted, flavour)
    else:  # test escape only: release-excluded sources never ship
        admitted = [s for s in admitted if not release_excluded(s)]
    not_in_map: dict[str, tuple[str, str]] = {}
    if flavour is not None:  # the map enumerator never saw these (WP-R2d): don't root them
        not_in_map = _skipped_sources(registry, admitted, roots, frozenset(allow_skip))
        admitted = [sid for sid in admitted if sid not in not_in_map]
    if not admitted:
        raise ValueError(
            f"no admitted source under profile {profile!r}"
            + (f" in flavour {flavour!r}" if flavour else "")
            + " has a resolved root"
        )
    # WP-7: the label gate — 0 silent drops, >= 95% mapped (or a documented exception),
    # every canonical taxon node anchored. Skipped for in-memory registries (no root).
    unstaged = (  # flavour builds already dropped rowless sources in _skipped_sources
        [] if flavour is not None else [sid for sid in admitted if not _has_rows(Path(roots[sid]))]
    )
    taxonomy_stamp = (
        {} if allow_unmapped else release_label_gate(registry, admitted, empty=unstaged)
    )
    admitted_roots = {source_id: Path(roots[source_id]) for source_id in admitted}
    never_eval_sources = _never_eval_source_ids(registry, admitted)
    walk_groups = {
        sid: {e.key: e.group for e in source_release_entries(registry.source(sid), root, digest)}
        for sid, root in admitted_roots.items()
        if _loader_layout(registry.source(sid)) != "staged-tree"
    }

    release_root = Path(out_dir) / "releases" / release / (flavour or "")
    manifests_dir = release_root / "tasks"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    map_bytes = split_map_path.read_bytes()
    (release_root / "SPLIT_MAP.json").write_bytes(map_bytes)
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()

    tasks: list[TaskManifest] = []
    skipped: list[tuple[str, str]] = []
    never_eval_excluded = 0
    near_dup_excluded: list[str] = []
    if near_dup is not None and never_eval_sources:
        near_dup_excluded = _never_eval_near_dups(
            registry, admitted_roots, profile, never_eval_sources, near_dup, digest=digest
        )
    near_dup_excluded_set = frozenset(near_dup_excluded)
    near_dup_rows = 0
    partial_abstain_excluded: list[PartialAbstainExclusion] = []

    for task in sorted(registry.tasks, key=lambda t: t.id):
        builder = DatasetBuilder(
            registry,
            profile=profile,
            roots=admitted_roots,
            schema_id=task.schema_id or DEFAULT_SCHEMA_ID,
            task_id=task.id,
            allow_unmapped=allow_unmapped,
        )
        try:
            dataset = builder.build()
        except ValueError as exc:
            # No admitted source can serve this task (e.g. a partial dry-run roots) —
            # not a defect in the release, just an empty task this time.
            skipped.append((task.id, str(exc)[:200]))
            continue

        partial_abstain_excluded.extend(dataset.partial_abstain_excluded)
        dataset.samples = [
            _with_release_group(sample, walk_groups, admitted_roots) for sample in dataset.samples
        ]
        dataset.split(by="group", split_map=split_map_path, frozen=True, tolerance=None)

        rows: list[tuple[str, str]] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None or not sample_ships(registry, sample, flavour):
                    continue
                sha256 = digest(Path(sample.image))
                if sample.source_id in never_eval_sources and sha256 in near_dup_excluded_set:
                    near_dup_rows += 1
                    continue
                if split_name != TRAIN and sample.source_id in never_eval_sources:
                    # A never-eval-only merged component is already forced to `train`
                    # (see generate_split_map); this is the mixed-component case — some
                    # of the group's images are real evaluation data, so the group still
                    # lands wherever they send it, but the machine-generated rows must
                    # never enter the resulting evaluation split. Counted, never moved
                    # to train — that would leak the same content the split holds out.
                    never_eval_excluded += 1
                    continue
                rows.append((sha256, split_name))
        rows.sort()

        manifest_path = manifests_dir / f"{task.id}.tsv"
        with manifest_path.open("w", newline="") as fh:
            fh.write("image_sha256\tsplit\n")
            for sha256, split_name in rows:
                fh.write(f"{sha256}\t{split_name}\n")

        tasks.append(TaskManifest(task_id=task.id, rows=tuple(rows)))

    release_json = {
        "release": release,
        **taxonomy_stamp,
        "split_map_sha256": map_sha256,
        "sources": [
            {
                "id": source_id,
                "version": registry.source(source_id).version,
                "root_digest": (
                    registry.source(source_id).checksums.root_digest
                    if registry.source(source_id).checksums is not None
                    else ""
                ),
            }
            for source_id in admitted
        ],
        "tasks": [task.task_id for task in tasks],
        "skipped_tasks": [{"task": task_id, "reason": reason} for task_id, reason in skipped],
        "partial_abstain_excluded": [
            {
                "source": exclusion.source_id,
                "task": exclusion.task_id,
                "abstaining_labels": list(exclusion.abstaining_labels),
                "rows_dropped": exclusion.rows_dropped,
            }
            for exclusion in partial_abstain_excluded
        ],
    }
    if not_in_map:
        release_json["skipped_sources"] = [
            {"id": sid, "reason": reason, "allowed_by": allowed_by}
            for sid, (reason, allowed_by) in sorted(not_in_map.items())
        ]
    if near_dup is not None:
        release_json["near_dup"] = near_dup_record(near_dup)
        release_json["never_eval_near_dup_excluded"] = {
            "count": len(near_dup_excluded),
            "sha256": near_dup_excluded,
        }
    if dedup_v2_groups is not None:
        # WP-10 split-leak gate, OFF by default: when off nothing below changes, so v1
        # outputs stay byte-identical. The gate report is written even on failure, and a
        # failing release never gets a RELEASE.json.
        from .dedup.groups import DedupGateError, gate_release, write_gate_report

        gate = gate_release(release_root, dedup_v2_groups)
        write_gate_report(gate, release_root / "DEDUP_GATE.json")
        if not gate.ok:
            raise DedupGateError(
                f"split-leak gate failed: {len(gate.spanning)} group(s) span splits, "
                f"{len(gate.upstream_test_in_train)} upstream-test group(s) in train, "
                f"{len(gate.ungrouped)} ungrouped image(s) — see {release_root / 'DEDUP_GATE.json'}"
            )
        release_json["dedup_v2"] = {
            "groups_sha256": file_digest(Path(dedup_v2_groups)),
            "gate": "pass",
            "groups": gate.groups,
        }
    if split_v2:  # WP-11/12 P3 hook, off by default (INT-core2); config-only smoke
        # check — see the build_release docstring for why the full per-sample gate
        # is deferred to the v2 build.
        from .benchmarks import BenchmarkRegistry, benchmarks_sha256
        from .splitv2.rules import load_rules, rules_sha256

        rules = load_rules("registry/splits/v2.yaml")
        bench = BenchmarkRegistry.load()
        release_json["split_v2"] = {
            "rules_sha256": rules_sha256(rules),
            "benchmarks_sha256": benchmarks_sha256(bench),
            "status": "config-validated-only; per-sample gate deferred to the v2 build",
        }
    if decon:  # WP-12 P2 hook, off by default; the integrator flips it at the v2 build
        from .decon import run_decon_gate

        run_decon_gate(release_root, registry, admitted_roots, dedup_crop=dedup_crop)
    if flavour is not None:
        release_json.update(release_record(registry, flavour, profile, admitted))
    release_json_text = json.dumps(release_json, indent=2, sort_keys=True) + "\n"
    (release_root / "RELEASE.json").write_text(release_json_text)

    task_layer_configs: dict[str, int] = {}
    if tasks_mode == "v2":
        from .task_layers.configs import build_all_configs, resolve_tasklabels_root, write_configs

        # INT-core3c: never the cwd — explicit root, else derived from the registry.
        base_dir = resolve_tasklabels_root(registry, tasklabels_root)
        results = build_all_configs(registry, base_dir, flavour)
        write_configs(results, release_root / "task_layers")
        task_layer_configs = {config_id: result.n_images for config_id, result in results.items()}

    return ReleaseResult(
        release=release,
        out_dir=release_root,
        sources=tuple(admitted),
        tasks=tuple(tasks),
        skipped_tasks=tuple(skipped),
        never_eval_excluded=never_eval_excluded,
        partial_abstain_excluded=tuple(partial_abstain_excluded),
        never_eval_near_dup_excluded=tuple(near_dup_excluded),
        never_eval_near_dup_rows=near_dup_rows,
        task_layer_configs=task_layer_configs,
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
