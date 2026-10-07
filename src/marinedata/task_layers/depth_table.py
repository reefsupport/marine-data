"""Depth maps into the unified ``depth`` table (WP-U11).

One row per (image, depth map): ``image_sha256`` is the RGB frame, ``depth_ref`` the bucket key of the
depth file, ``units`` (``m`` / ``disparity_px`` / ``relative``), ``gt_type`` (``sensor`` / ``stereo`` /
``sfm`` / ``synthetic`` / ``estimated``) and an optional ``valid_mask_ref``. The brief's
``depth_kind`` (``gt`` / ``sparse`` / ``disparity`` / ``pseudo``) is kept in ``attrs.depth_kind``.

Depth files that sit inside ``images/`` (usod10k, sonarsweep, viame-public) are paired to their
RGB frame by stem; the row's ``attrs`` marks them ``non_image: true`` and names their sha256. No
bucket object is moved. A frame whose sha256 is unknown (no CHECKSUMS on the staged tree) is a
*pending* row (``image_key`` + a TODO to resolve once the tree is checksummed), as for pending boxes.

``attrs.licence_class`` (always set) is the lic-A/lic-B proposed class, else the registry tier,
else ``unknown``. Sources and their producers are bound in :data:`DEPTH_PAIR_SOURCES` (shared with
:mod:`marinedata.task_layers.pairs_table`); :data:`DEPTH_PAIR_GAPS` lists the sources with depth or
pair labels that cannot be built from what is staged, with the reason.
"""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..annotation_schema import annotation_path, write_annotations
from .vqa_table import OPEN, VqaResult, free_row

INTERNAL_ONLY = "internal-only"
DEPTH_KINDS = frozenset({"gt", "sparse", "disparity", "pseudo"})
__all__ = [
    "DEPTH_KINDS", "DEPTH_PAIR_GAPS", "DEPTH_PAIR_SOURCES", "DepthPairResult", "DepthPairSource",
    "depth_row", "staged_depth_pairs", "write_depth",
]  # fmt: skip


@dataclass(frozen=True)
class DepthPairSource:
    """One staged source with depth maps and/or image pairs, and how its labels were produced."""

    source_id: str
    version: str
    producer: str  # key of ``sources.depth_pairs.PRODUCERS``
    annotator: str  # label-origin vocabulary
    annotator_detail: str
    ann_license: str | None
    licence_class: str
    tables: tuple[str, ...]  # which of ``depth`` / ``pairs`` the source can feed
    lang: str = "en"  # unused by these tables; keeps the free_row duck type of VqaSource


@dataclass
class DepthPairResult:
    depth: VqaResult = field(default_factory=VqaResult)
    pairs: VqaResult = field(default_factory=VqaResult)
    gaps: list[str] = field(default_factory=list)


DEPTH_PAIR_SOURCES: dict[str, DepthPairSource] = {
    s.source_id: s
    for s in (
        # lic-A: internal-only (research-only, no redistribution). Depth is an estimate (no sensor).
        DepthPairSource("usod10k", "zip-f82114057c27", "usod10k", "pseudo",
                        "estimated-depth:unverified (USOD10K does not document a depth sensor)",
                        None, INTERNAL_ONLY, ("depth",)),
        # lic-A: internal-only (no licence stated). Simulated sweeps: pairs and depth come from the simulator.
        DepthPairSource("sonarsweep", "rev-350b20a9acaf", "sonarsweep", "derived_rule",
                        "synthetic:simulator-aligned (same frame id, cam/sonar/depth renders)",
                        None, INTERNAL_ONLY, ("depth", "pairs")),
        # registry spec: CC-BY-4.0, in neither lic TSV -> registry tier open. Camera and sonar are time-aligned.
        DepthPairSource("auv-flc-fls-sunboat", "record-10544811", "auv_sunboat", "derived_rule",
                        "time-aligned camera/sonar frame pair (same timestamp prefix and index)",
                        None, OPEN, ("pairs",)),
        # lic-A: internal-only. Reference selected by volunteers upstream (unverified here).
        DepthPairSource("lsui", "zip-e15c4a2203f4", "lsui", "human",
                        "reference chosen upstream from enhancement methods (unverified here)",
                        None, INTERNAL_ONLY, ("pairs",)),
        # registry licence US-GOV-PD (open), in neither lic TSV. No CHECKSUMS -> pending rows.
        DepthPairSource("viame-public", "girder-pinned", "viame_habcam", "derived_rule",
                        "stereo rig: HabCam2019 -left/-right/-disp triple (same stem)",
                        None, OPEN, ("depth", "pairs")),
        # lic-A: internal-only. One image per parquet row, reference column not staged: gap only.
        DepthPairSource("euvp", "rev-e3c6d08a8b07", "euvp", "human",
                        "paired reference not staged", None, INTERNAL_ONLY, ("pairs",)),
    )
}  # fmt: skip

# Sources the audit / U15 triage list with depth or pairs that cannot be built from what is staged.
DEPTH_PAIR_GAPS: dict[str, str] = {
    "flsea": "22,451 images = the pair count: one file per parquet row, the paired depth column is not staged",
    "uw-stereodepth-40k": "only image_left staged (15,676 png), no CHECKSUMS, right image and depth not staged",
    "uwstereo-syn": "nothing staged (metadata.parquet has 0 rows)",
    "estonefish-scenes-real": "nothing staged (metadata.parquet has 0 rows)",
    "uxo-acoustic-optical": "3 preview images only, no depth or pair files staged",
    "koi-rgb-sonar": "images are 65 tar archives, never extracted: no per-frame keys, no CHECKSUMS",
    "elliott-bay-benthic": "500 jpg <-> 500 tif pair by timestamp, but the tif's role is unverified and is in no pair_role",
    "pingmapper-sss-seg": "depth declared by the spec but no depth files are staged next to the 31,014 png",
}  # fmt: skip


def depth_row(
    *,
    spec: DepthPairSource,
    ordinal: int,
    sha: str | None,
    depth_ref: str,
    units: str,
    gt_type: str,
    depth_kind: str,
    split: str | None = None,
    valid_mask_ref: str | None = None,
    attrs: Mapping[str, object] | None = None,
) -> dict:
    """One ``depth`` row; ``sha`` None = pending (the caller adds ``image_key``)."""
    if depth_kind not in DEPTH_KINDS:
        raise ValueError(f"depth_kind {depth_kind!r} not in {sorted(DEPTH_KINDS)}")
    return free_row(
        spec=spec,
        ordinal=ordinal,
        sha=sha,
        split=split,
        attrs={**dict(attrs or {}), "depth_kind": depth_kind},
        payload={
            "depth_ref": depth_ref,
            "units": units,
            "gt_type": gt_type,
            "valid_mask_ref": valid_mask_ref,
        },
    )


def write_depth(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    path = annotation_path(data_dir, "depth", source_id, source_version)
    write_annotations(path, "depth", rows)
    return path


def staged_depth_pairs(
    spec: DepthPairSource,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes] | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> DepthPairResult:
    """Run the producer bound to ``spec`` (imported lazily: producers import this module)."""
    from .sources.depth_pairs import PRODUCERS

    return PRODUCERS[spec.producer](spec, limit=limit, fetch=fetch, lister=lister)
