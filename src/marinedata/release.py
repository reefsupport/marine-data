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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .builder import SUPERVISED_DEFAULT_RATIOS, DatasetBuilder, PartialAbstainExclusion, SplitName
from .checksums import file_digest
from .gate import evaluate
from .neardup import (
    NearDupChainError,
    NearDupConfig,
    compute_dhashes,
    near_dup_record,
    near_pairs,
)
from .registry import Registry
from .splitmap import MergeInfo, Row, load_split_map, resolve_splits, rows_to_counts
from .strata import DEFAULT_MIN_GROUPS, TRAIN
from .tables import _require_pyarrow

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


def _never_eval_source_ids(registry: Registry, source_ids: Iterable[str]) -> set[str]:
    """Admitted sources tagged ``never-eval`` in the registry (WS-D S15c) — their rows
    may only ever land in the ``train`` split."""
    return {
        source_id
        for source_id in source_ids
        if "never-eval" in registry.source(source_id).tags
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


def enumerate_release_rows(
    registry: Registry,
    roots: dict[str, str | Path],
    profile: str = "research",
    *,
    skipped: dict[str, str] | None = None,
    upstream_splits: dict[str, set[str]] | None = None,
    paths: dict[str, Path] | None = None,
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
    import pyarrow.parquet as pq

    for source_id in _admitted_source_ids(registry, roots, profile):
        source = registry.source(source_id)
        if "needs-attribution" in source.tags:
            if skipped is not None:
                skipped[source_id] = (
                    "needs-attribution: attribution target unconfirmed, not in release"
                )
            continue
        layout = source.loader.layout if source.loader is not None else None
        if layout != "staged-tree":
            if skipped is not None:
                skipped[source_id] = (
                    f"{layout if layout is not None else 'no loader'}: not in release "
                    "until labels format decided"
                )
            continue
        root = Path(roots[source_id])
        metadata_path = root / "metadata.parquet"
        if not metadata_path.is_file():
            raise ValueError(
                f"{source_id}: no metadata.parquet under {root} — the release enumerator "
                "reads the staging pipeline's sample index, independent of the source's "
                "own loader layout"
            )
        partition_indexes: dict[str, dict[str, list[Path]]] = {}
        for record in pq.read_table(metadata_path).to_pylist():
            group = record.get("split_group")
            if not group:
                raise ValueError(
                    f"{source_id}: metadata.parquet row for stem={record['stem']!r} has no "
                    "split_group — the registry's per-source split_group rule must run "
                    "before staging"
                )
            partition, stem = record["partition"], record["stem"]
            index = partition_indexes.get(partition)
            if index is None:
                index = _partition_stem_index(root / "images" / partition)
                partition_indexes[partition] = index
            matches = index.get(stem, [])
            if not matches:
                raise ValueError(
                    f"{source_id}: metadata.parquet references image {stem!r} (partition "
                    f"{partition!r}) not found under {root / 'images' / partition}"
                )
            if upstream_splits is not None:
                upstream_split = record.get("upstream_split")
                if upstream_split:
                    upstream_splits.setdefault(group, set()).add(upstream_split)
            digest = file_digest(matches[0])
            if paths is not None:
                paths.setdefault(digest, matches[0])
            yield digest, group, source_id


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
    """
    if load_split_map(out) is not None:
        raise ValueError(f"{out} already exists — remove it first to regenerate")
    never_eval_sources = _never_eval_source_ids(
        registry, _admitted_source_ids(registry, roots, profile)
    )
    upstream_splits: dict[str, set[str]] = {}
    paths: dict[str, Path] = {}
    rows = list(
        enumerate_release_rows(
            registry,
            roots,
            profile,
            skipped=skipped,
            upstream_splits=upstream_splits,
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
        near_dup=near_dup_record(near_dup) if near_dup is not None else None,
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
) -> ReleaseResult:
    """Build every registry task against a frozen split map and write the release.

    Raises if ``split_map`` does not exist yet (a release never allocates one — see
    :mod:`marinedata.splitmap`), or if any admitted source contributes a group absent
    from it (``SplitMapError``, propagated from ``Dataset.split(frozen=True)``) — both
    left uncaught deliberately, since either means the release is not reproducible yet.

    ``near_dup`` (WS-D S47 rule A): every image of a ``never-eval`` source within
    ``near_dup.exclude_max`` dHash Hamming of any row of an eval-capable admitted source
    (any split) is dropped from every task manifest, and recorded in ``RELEASE.json`` as
    a count plus the sorted sha256 list, alongside the check's own parameters.
    """
    split_map_path = Path(split_map)
    if load_split_map(split_map_path) is None:
        raise ValueError(
            f"{split_map_path} does not exist — generate it first with "
            "`marinedata splitmap generate`."
        )

    admitted = _admitted_source_ids(registry, roots, profile)
    if not admitted:
        raise ValueError(f"no admitted source under profile {profile!r} has a resolved root")
    admitted_roots = {source_id: Path(roots[source_id]) for source_id in admitted}
    never_eval_sources = _never_eval_source_ids(registry, admitted)

    release_root = Path(out_dir) / "releases" / release
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
            registry, admitted_roots, profile, never_eval_sources, near_dup
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
        dataset.split(by="group", split_map=split_map_path, frozen=True, tolerance=None)

        rows: list[tuple[str, str]] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None:
                    continue
                sha256 = file_digest(Path(sample.image))
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
    release_json_text = json.dumps(release_json, indent=2, sort_keys=True) + "\n"
    (release_root / "RELEASE.json").write_text(release_json_text)

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
    )


def _never_eval_near_dups(
    registry: Registry,
    roots: dict[str, Path],
    profile: str,
    never_eval_sources: set[str],
    config: NearDupConfig,
) -> list[str]:
    """Sorted sha256 of never-eval images within ``config.exclude_max`` of any
    eval-capable admitted image (rule A). Enumerates the same staged-tree rows the split
    map is generated from, so every admitted row counts, whatever split or task."""
    paths: dict[str, Path] = {}
    never_eval: set[str] = set()
    eval_capable: set[str] = set()
    for sha, _, source_id in enumerate_release_rows(registry, roots, profile, paths=paths):
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
