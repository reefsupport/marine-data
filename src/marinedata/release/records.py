"""Result records of a release build: task manifest, build result, split-map statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..builder import PartialAbstainExclusion

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
    prior_band_excluded: tuple[str, ...] = ()
    """Sorted sha256 of every image the split map's ``band_excluded`` lists (schema 2): a new image
    inside the prior-release near-duplicate band, on the side its component did not keep. Dropped
    from every split of every config; ``RELEASE.json`` records them as ``prior_band_excluded``."""
    prior_band_rows: int = 0
    """Task-manifest rows those images would have contributed, summed over tasks."""
    task_layer_configs: dict[str, int] = field(default_factory=dict)
    """``{config_id: n_images}`` for the 5 WP-8c v2 configs — empty unless ``tasks="v2"``
    was passed to :func:`build_release` (D-X: v1 default leaves this empty, always)."""
    staged_layers: tuple[object, ...] = ()
    """``--v11`` only: one ``task_layers.staged_annotations.StagedLayer`` tally per (table, source)
    built from the staged trees (or reused from the tasklabels root); also in
    ``V11_TABLES.json``."""


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
    near_dup_candidates: int = 0
    """dHash pairs within the union radius before rule B's confirm-hash (v1.1.0)."""
    near_dup_rejected_phash: int = 0
    """Candidates the confirm-hash rejected: pHash-256 too far apart (different scenes)."""
    near_dup_rejected_entropy: int = 0
    """Candidates the confirm-hash rejected: a flat frame, where no hash proves anything."""
    near_dup_phash_pairs: int = 0
    """Eval-capable image pairs within the gates' DCT pHash-64 radius, unioned like a dHash pair
    (v1.1.0: the dedup and splits gates' near-duplicate rule)."""
    upstream_test_groups: int = 0
    """Raw split groups holding an upstream-test row that the "honour upstream test" rule
    (WP-R2c) pulled to ``test``; their merged components are counted below."""
    upstream_test_components: int = 0
    """Merged components forced to ``test`` by that rule (a never-eval-only one stays train)."""
    dedup_group_links: int = 0
    """Links added from the ``dedup run`` groups (WP-R6): every image of a dedup group in the
    enumeration is tied to the group's first image, so a near-dup group never straddles splits."""
    inherited_groups: int = 0
    """Groups of the inherited prior map: all kept verbatim (D1)."""
    inherit_adopted_groups: int = 0
    """New groups that took a prior split because a prior group shares their component."""
    inherit_conflict_components: int = 0
    """Components whose prior groups the prior map had split apart (new groups withheld)."""
    inherit_band_pairs: int = 0
    """New x prior image pairs inside the near-duplicate band (5..8): never-eval / never-train."""
    inherit_band_excluded: int = 0
    """Images the band excludes one by one (``SPLIT_MAP.json`` ``band_excluded``, schema 2)."""
    withheld_groups: int = 0
    """New groups with no entry, because no split can honour every prior neighbour."""
    ood_components: int = 0
    """Components routed to ``test_ood`` by the ``splits/v2.yaml`` holdouts (D2)."""
    ood_groups: int = 0
    """Split groups in those components."""
    ood_blocked_components: int = 0
    """Components an OOD rule matched but that precedence placed elsewhere (prior, band, ...)."""
    pins: dict[str, object] | None = None
    """The SPLIT_MAP ``pins`` header (benchmark pins, v1.0 conflicts); ``None`` without pins."""
