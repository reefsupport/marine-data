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

from .builder import SUPERVISED_DEFAULT_RATIOS, DatasetBuilder, SplitName
from .checksums import file_digest
from .gate import evaluate
from .registry import Registry
from .splitmap import Row, load_split_map, resolve_splits, rows_to_counts
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


def enumerate_release_rows(
    registry: Registry,
    roots: dict[str, str | Path],
    profile: str = "research",
    *,
    skipped: dict[str, str] | None = None,
    upstream_splits: dict[str, set[str]] | None = None,
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
    """
    _require_pyarrow()
    import pyarrow.parquet as pq

    for source_id in _admitted_source_ids(registry, roots, profile):
        source = registry.source(source_id)
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
        for record in pq.read_table(metadata_path).to_pylist():
            group = record.get("split_group")
            if not group:
                raise ValueError(
                    f"{source_id}: metadata.parquet row for stem={record['stem']!r} has no "
                    "split_group — the registry's per-source split_group rule must run "
                    "before staging"
                )
            partition, stem = record["partition"], record["stem"]
            matches = sorted((root / "images" / partition).glob(f"{stem}.*"))
            if not matches:
                raise ValueError(
                    f"{source_id}: metadata.parquet references image {stem!r} (partition "
                    f"{partition!r}) not found under {root / 'images' / partition}"
                )
            if upstream_splits is not None:
                upstream_split = record.get("upstream_split")
                if upstream_split:
                    upstream_splits.setdefault(group, set()).add(upstream_split)
            yield file_digest(matches[0]), group, source_id


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
    """
    if load_split_map(out) is not None:
        raise ValueError(f"{out} already exists — remove it first to regenerate")
    never_eval_sources = _never_eval_source_ids(
        registry, _admitted_source_ids(registry, roots, profile)
    )
    upstream_splits: dict[str, set[str]] = {}
    rows = enumerate_release_rows(
        registry, roots, profile, skipped=skipped, upstream_splits=upstream_splits
    )
    counts, strata, merge_info = rows_to_counts(rows, stratified=True)

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
) -> ReleaseResult:
    """Build every registry task against a frozen split map and write the release.

    Raises if ``split_map`` does not exist yet (a release never allocates one — see
    :mod:`marinedata.splitmap`), or if any admitted source contributes a group absent
    from it (``SplitMapError``, propagated from ``Dataset.split(frozen=True)``) — both
    left uncaught deliberately, since either means the release is not reproducible yet.
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

        dataset.split(by="group", split_map=split_map_path, frozen=True, tolerance=None)

        rows: list[tuple[str, str]] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None:
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
                rows.append((file_digest(Path(sample.image)), split_name))
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
    )
