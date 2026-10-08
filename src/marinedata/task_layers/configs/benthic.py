"""Task-layer configs: benthic (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

import json
from pathlib import Path

from ...registry import Registry
from ..rollup import MIXED, UNKNOWN, rollup_counts
from .common import ConfigResult, _canonical_taxon, _SourceUnmapped
from .constants import BENTHIC_COARSE_TASK, ROLLUP_MASK_SOURCES, ROLLUP_POINT_SOURCES
from .masks import _mask_records
from .points import _point_records
from .tasklabels import _load_label_status


def _coarse_counts_from_points(
    registry: Registry, projector, source_id: str, points: list[dict]
) -> dict[str, dict[str, int]]:
    """``{sha256: {coarse_class_or_unknown: n}}`` for one source's point rows."""
    per_image: dict[str, dict[str, int]] = {}
    for point in points:
        counts = per_image.setdefault(point["image_sha256"], {})
        canonical = point.get("taxon_node_id") or _canonical_taxon(
            registry, source_id, point["label_native"]
        )  # noqa: E501, RUF100
        target = projector.project(canonical).target_class if canonical is not None else None
        cls = target or UNKNOWN
        counts[cls] = counts.get(cls, 0) + 1
    return per_image


def _coarse_counts_from_masks(
    registry: Registry, projector, source_id: str, masks: list[dict]
) -> dict[str, dict[str, int]]:
    """``{sha256: {coarse_class_or_unknown: n}}`` for one source's mask rows."""
    per_image: dict[str, dict[str, int]] = {}
    for record in masks:
        native_counts: dict[str, int] = json.loads(record["class_counts"])
        counts: dict[str, int] = {}
        for native_label, n in native_counts.items():
            canonical = _canonical_taxon(registry, source_id, native_label)
            target = projector.project(canonical).target_class if canonical is not None else None
            cls = target or UNKNOWN
            counts[cls] = counts.get(cls, 0) + n
        per_image[record["sha256"]] = counts
    return per_image


def _build_benthic_rollup(
    registry: Registry, base_dir: Path
) -> tuple[dict[str, tuple[str, dict, object]], dict[str, float]]:
    """Shared D-Y rollup for both benthic configs: ``{sha256: (source_id, origin,
    rollup_result)}`` plus per-source unmapped fractions."""
    projector = registry.projector_for(BENTHIC_COARSE_TASK)
    origins: dict[str, str] = {}
    per_image_counts: dict[str, dict[str, int]] = {}
    unmapped: dict[str, _SourceUnmapped] = {}
    source_of: dict[str, str] = {}

    def add(source_id: str, per_source: dict[str, dict[str, int]]) -> None:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for sha256, counts in per_source.items():
            per_image_counts.setdefault(sha256, {})
            for cls, n in counts.items():
                per_image_counts[sha256][cls] = per_image_counts[sha256].get(cls, 0) + n
                if cls == UNKNOWN:
                    tally.unmapped += n
                else:
                    tally.known += n
            origins.setdefault(sha256, "human")
            source_of.setdefault(sha256, source_id)

    for source_id in ROLLUP_POINT_SOURCES:
        points = _point_records(base_dir, source_id)
        add(source_id, _coarse_counts_from_points(registry, projector, source_id, points))
    for source_id in ROLLUP_MASK_SOURCES:
        masks = _mask_records(base_dir, source_id)
        add(source_id, _coarse_counts_from_masks(registry, projector, source_id, masks))

    result = {
        sha256: (source_of[sha256], origins[sha256], rollup_counts(counts))
        for sha256, counts in per_image_counts.items()
    }
    return result, {sid: t.fraction() for sid, t in unmapped.items()}


def build_benthic_coarse_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``benthic-coarse``: single-label view (D-Z). ``mixed`` -> null."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    per_image, unmapped_by_source = _build_benthic_rollup(registry, base_dir)
    rows = []
    for sha256, (source_id, origin, result) in per_image.items():
        dominant = None if result.dominant in (None, MIXED) else result.dominant
        rows.append(
            {
                "sha256": sha256,
                "source_id": source_id,
                "label_origin": origin,
                "benthic_dominant": dominant,
                "label_status": label_status.get(sha256, "ok"),
            }
        )
    return ConfigResult("benthic-coarse", tuple(rows), unmapped_by_source)


def build_benthic_cover_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``benthic-cover``: the full D-Y multi-label rollup (cover/dominant/present)."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    per_image, unmapped_by_source = _build_benthic_rollup(registry, base_dir)
    rows = []
    for sha256, (source_id, origin, result) in per_image.items():
        rows.append(
            {
                "sha256": sha256,
                "source_id": source_id,
                "label_origin": origin,
                "benthic_cover": json.dumps(result.cover, sort_keys=True),
                "benthic_dominant": result.dominant,
                "benthic_present": json.dumps(sorted(result.present)),
                "label_status": label_status.get(sha256, "ok"),
            }
        )
    return ConfigResult("benthic-cover", tuple(rows), unmapped_by_source)
