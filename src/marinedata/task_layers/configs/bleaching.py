"""Task-layer configs: bleaching (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

from pathlib import Path

from ...registry import Registry
from ...schema import Axis
from .common import ConfigResult, _annotation_image_labels, _attrs, _canonical, _SourceUnmapped
from .constants import BLEACHING_SOURCES, BLEACHING_TASK, HEALTH_TASK
from .tasklabels import _load_label_status, _read_tasklabels


def _bleaching_records(base_dir: Path, source_id: str) -> list[dict]:
    """Bleaching rows of one source in the legacy shape (``sha256``, ``label_origin``,
    ``native_label``, ``evidence``, ``pixel_count``, ``confidence``) plus the unified columns
    (``image_sha256, ann_id, condition_node_id, match_type, annotator_type, licence_class``).
    The unified ``image_labels`` table wins (WP-U5); else ``_tasklabels/<source>/bleaching.parquet``."""  # noqa: E501
    unified = _annotation_image_labels(base_dir, source_id)
    if not unified:
        return _read_tasklabels(base_dir, source_id, "bleaching")
    out = []
    for r in unified:
        extra = _attrs(r)
        out.append(
            {
                "sha256": r["image_sha256"], "label_origin": r.get("annotator_type"),
                "native_label": r["label_native"], "evidence": extra.get("evidence", "image"),
                "pixel_count": extra.get("pixel_count"), "confidence": r.get("confidence"),
                "image_sha256": r["image_sha256"], "ann_id": r.get("ann_id"),
                "condition_node_id": r.get("condition_node_id"), "unified": True,
                "match_type": r.get("match_type"), "annotator_type": r.get("annotator_type"),
                "licence_class": extra.get("licence_class"),
            }
        )  # fmt: skip
    return out


def build_bleaching_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``bleaching``: one row per (image, native condition label) from every
    :data:`BLEACHING_SOURCES` file, with the canonical ``condition`` node and its
    projection onto both condition tasks — ``bleaching-condition`` (6-class; abstains,
    i.e. null, on a label too coarse to split, e.g. Roboflow "Unhealthy") and
    ``coral-health-binary`` (HEALTHY/UNHEALTHY). A label with no condition edge is
    unmapped (tallied per source by row), never guessed."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    fine = registry.projector_for(BLEACHING_TASK)
    binary = registry.projector_for(HEALTH_TASK)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []
    for source_id in BLEACHING_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for record in _bleaching_records(base_dir, source_id):
            node = (
                record.get("condition_node_id")
                if record.get("unified")
                else _canonical(registry, source_id, record["native_label"], Axis.CONDITION)
            )
            if node is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            rows.append(
                {
                    "sha256": record["sha256"],
                    "image_sha256": record["sha256"],
                    "source_id": source_id,
                    "ann_id": record.get("ann_id"),
                    "label_origin": record["label_origin"],
                    "annotator_type": record.get("annotator_type"),
                    "licence_class": record.get("licence_class"),
                    "native_label": record["native_label"],
                    "label_native": record["native_label"],
                    "canonical_condition": node,
                    "match_type": record.get("match_type"),
                    "bleaching_condition": fine.project(node).target_class if node else None,
                    "coral_health": binary.project(node).target_class if node else None,
                    "evidence": record.get("evidence"),
                    "pixel_count": record.get("pixel_count"),
                    "confidence": record.get("confidence"),
                    "label_status": label_status.get(record["sha256"], "ok"),
                }
            )
    return ConfigResult(
        "bleaching", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()}
    )
