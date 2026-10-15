"""Task-layer configs: masks (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

import json
from pathlib import Path

from ...registry import Registry
from ..mask_classmap import coarse_of as _coarse_of
from ..mask_classmap import hf_class_map, json_or_none
from .common import ConfigResult, _annotation_masks, _attrs, _canonical_taxon, _SourceUnmapped
from .constants import INSTANCE_CONFIG_SOURCES, SEMSEG_SOURCES
from .tasklabels import _load_label_status, _read_tasklabels


def _mask_records(base_dir: Path, source_id: str) -> list[dict]:
    """Semantic mask rows of one source in the legacy semseg shape (``sha256``, ``label_origin``,
    ``mask_key``, ``class_counts``) plus the unified columns. The unified table wins; else the
    legacy ``_tasklabels/<source>/semseg.parquet`` is read as before."""
    unified = _annotation_masks(base_dir, source_id)
    if not unified:
        return _read_tasklabels(base_dir, source_id, "semseg")
    return [
        {
            "sha256": r["image_sha256"],
            "label_origin": r.get("annotator_type"),
            "mask_key": r["mask_ref"],
            "class_counts": r["class_counts"],
            "image_sha256": r["image_sha256"],
            "ann_id": r.get("ann_id"),
            "mask_kind": r.get("mask_kind"),
            "match_type": r.get("match_type"),
            "annotator_type": r.get("annotator_type"),
            "ann_license": r.get("ann_license"),
            "licence_class": _attrs(r).get("licence_class"),
            "taxon_by_label": {
                k: v["taxon_node_id"] for k, v in _attrs(r).get("class_resolution", {}).items()
            },  # noqa: E501, RUF100
            "class_map": r.get("class_map"),
            "ignore_value": r.get("ignore_value"),
            "class_resolution": _attrs(r).get("class_resolution", {}),
            "mask_encoding": _attrs(r).get("mask_encoding"),
            "mask_parts": _attrs(r).get("mask_parts"),
        }
        for r in unified
    ]


def build_semseg_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``semseg``: Coralscapes (today) per-image mask rows, native + canonical pixel
    counts (unmapped pixels — e.g. Coralscapes has no soft-coral class — kept out of the
    canonical histogram, tallied in ``unmapped_by_source``)."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []
    coarse = _coarse_of(registry)

    for source_id in SEMSEG_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for record in _mask_records(base_dir, source_id):
            native_counts: dict[str, int] = json.loads(record["class_counts"])
            canonical_counts: dict[str, int] = {}
            resolved = record.get("taxon_by_label")  # unified rows carry U2's per-class resolve
            for native_label, n in native_counts.items():
                canonical = (
                    resolved.get(native_label)
                    if resolved is not None
                    else _canonical_taxon(registry, source_id, native_label)
                )
                if canonical is None:
                    tally.unmapped += n
                    continue
                tally.known += n
                canonical_counts[canonical] = canonical_counts.get(canonical, 0) + n
            rows.append(
                {
                    "sha256": record["sha256"],
                    "source_id": source_id,
                    "label_origin": record["label_origin"],
                    "mask_key": record["mask_key"],
                    "class_counts": record["class_counts"],
                    "canonical_class_counts": json.dumps(canonical_counts, sort_keys=True),
                    "label_status": label_status.get(record["sha256"], "ok"),
                    "hf_class_map": hf_class_map(record, coarse),
                    "ignore_value": record.get("ignore_value"),
                    "mask_encoding": record.get("mask_encoding"),
                    "mask_parts": json_or_none(record.get("mask_parts")),
                    **{
                        k: record[k]  # unified columns, present only on unified rows
                        for k in (
                            "image_sha256",
                            "ann_id",
                            "mask_kind",
                            "match_type",
                            "annotator_type",
                            "ann_license",
                            "licence_class",
                        )  # noqa: E501, RUF100
                        if k in record
                    },
                }
            )

    return ConfigResult("semseg", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()})


def build_instance_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``instances``: every unified ``masks`` row with ``mask_kind == instance`` of
    :data:`INSTANCE_CONFIG_SOURCES` (``instance_id`` column, ``polygon`` / ``rle``, native label,
    resolve fields, ``annotator_type``, ``ann_license``; ``licence_class`` and ``modality`` lifted
    from ``attrs``). ``unmapped_by_source`` is the fraction of instances with
    ``match_type == unmapped``."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    rows: list[dict] = []
    unmapped: dict[str, float] = {}
    coarse = _coarse_of(registry)
    for source_id in INSTANCE_CONFIG_SOURCES:
        masks = _annotation_masks(base_dir, source_id)
        records = [r for r in masks if r["mask_kind"] == "instance"]
        for r in records:
            attrs = _attrs(r)
            rows.append(
                {
                    **r,
                    "sha256": r["image_sha256"],
                    "licence_class": attrs.get("licence_class"),
                    "modality": attrs.get("modality"),
                    "coarse": coarse(r.get("taxon_node_id")),
                    "label_status": label_status.get(r["image_sha256"], r["label_status"]),
                }
            )
        if records:
            unmapped[source_id] = sum(r["match_type"] == "unmapped" for r in records) / len(records)
    return ConfigResult("instances", tuple(rows), unmapped)
