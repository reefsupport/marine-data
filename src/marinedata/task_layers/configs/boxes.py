"""Task-layer configs: boxes (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

from pathlib import Path

from ...registry import Registry
from .common import ConfigResult, _annotation_boxes, _attrs
from .constants import BOX_CONFIG_SOURCES
from .tasklabels import _load_label_status


def build_boxes_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``boxes``: every unified ``boxes`` row (xyxy normalised + pixel columns, native label,
    resolve fields, ``annotator_type``, ``ann_license``, ``attrs.licence_class`` lifted to a
    ``licence_class`` column so the open / restricted split is a filter, not a re-parse).
    ``unmapped_by_source`` is the fraction of boxes with ``match_type == unmapped``."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    rows: list[dict] = []
    unmapped: dict[str, float] = {}
    for source_id in BOX_CONFIG_SOURCES:
        records = _annotation_boxes(base_dir, source_id)
        for r in records:
            rows.append(
                {
                    **r,
                    "licence_class": _attrs(r).get("licence_class"),
                    "label_status": label_status.get(r["image_sha256"], r["label_status"]),
                }
            )
        if records:
            unmapped[source_id] = sum(r["match_type"] == "unmapped" for r in records) / len(records)
    return ConfigResult("boxes", tuple(rows), unmapped)
