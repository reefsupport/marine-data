"""Enumerating the rows of a release: admitted sources, staged or walked entries, grouping guard."""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from ..checksums import file_digest
from ..gate import evaluate
from ..licence_class import spec_for
from ..loaders.generic import IMAGE_SUFFIXES
from ..metadata_norm.base import NormContext
from ..metadata_norm.default import attribution_text
from ..metadata_norm.split_group import derive_split_group
from ..models import Source
from ..registry import Registry
from ..sample_schema import staged_partition
from ..splitmap import Row
from ..tables import _require_pyarrow

logger = logging.getLogger(__name__)

GUARD_MIN_ROWS = 100
"""A source with at least this many rows ..."""

GUARD_MIN_GROUPS = 10
"""... and fewer final split groups than this falls back to one group per image (WP-R2d)."""


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
    """``{stem: sorted candidate paths}`` for every file under ``partition_dir`` at any depth
    (a partition may nest its images, e.g. Seaview's ``labelled_data/segments/``), built with
    one directory walk rather than one ``glob(f"{stem}.*")`` per row.

    Keyed by ``Path(name).stem`` (split on the *last* dot) rather than a first-dot split:
    staging always appends exactly one extension to a stem that may itself contain dots
    (``Path(...).stem`` at staging time — see ``fetchers_remote``/``ingest_image_labels``),
    so this reconstructs the same key the glob-per-row pattern ``f"{stem}.*"`` matched
    against. A stem with two candidate files (e.g. ``a0.jpg`` and ``a0.png``) keeps the
    same tie-break as before: the candidates are sorted and the caller takes the first.
    """
    index: dict[str, list[Path]] = {}
    if partition_dir.is_dir():
        for path in partition_dir.rglob("*"):
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


def _image_path_file(root: Path, record: dict) -> Path | None:
    """The image file a staged record's own ``image_path`` names under ``root``, else ``None``
    (the caller then falls back to the ``images/<partition>/`` stem index).

    ``image_path`` is relative to the source root and names the file exactly, however deep its
    partition nests it. Only an image file counts: in shard layout ``image_path`` is the shard
    ``.tar`` (``image_member`` names the member), and digesting that would give every row of a
    shard the same sha256.
    """
    rel = record.get("image_path")
    if not rel:
        return None
    path = root / str(rel)
    return path if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file() else None


def _staged_entries(
    source: Source, root: Path, digest: Callable[[Path], str]
) -> list[ReleaseEntry]:
    """One entry per ``metadata.parquet`` row. A row's image is the file its own ``image_path``
    names when that exists; else the partition's stem index finds it; a row neither finds raises."""
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
        image = _image_path_file(root, record)
        if image is None:
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
            image = matches[0]
        sha256 = digest(image)
        group = _release_group(source, record, stem=stem, partition=partition, sha256=sha256)
        entries.append((sha256, group, record.get("upstream_split") or None, image))
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
