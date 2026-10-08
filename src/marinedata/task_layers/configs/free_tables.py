"""Task-layer configs: free tables (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

from pathlib import Path

from .common import ConfigResult, _annotation_free, _free_config_rows
from .constants import (
    DEPTH_CONFIG_SOURCES,
    LEGACY_VQA_SOURCES,
    PAIRS_CONFIG_SOURCES,
    VQA_CONFIG_SOURCES,
)
from .tasklabels import _load_label_status, _read_tasklabels


def build_vqa_config(base_dir: str | Path) -> ConfigResult:
    """``vqa``: every unified ``vqa`` row (``_annotations/vqa/<source>``: question, answer,
    ``qa_type``, ``annotator_type``, ``licence_class`` lifted from ``attrs``), registry-driven over
    :data:`VQA_CONFIG_SOURCES`. A legacy source with no unified file still reads its D-Z2
    ``_tasklabels`` rows unchanged. No crosswalk applies to free-text VQA (D-Y note: "treat as a
    language/UX asset, not a measurement one")."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    rows: list[dict] = []
    for source_id in VQA_CONFIG_SOURCES:
        if records := _annotation_free(base_dir, "vqa", source_id):
            rows.extend(_free_config_rows(records, label_status))
        elif source_id in LEGACY_VQA_SOURCES:
            for record in _read_tasklabels(base_dir, source_id, "vqa"):
                rows.append({**record, "label_status": label_status.get(record["sha256"], "ok")})
    return ConfigResult("vqa", tuple(rows), {})


def build_captions_config(base_dir: str | Path) -> ConfigResult:
    """``captions``: every unified ``captions`` row of every source that has a
    ``_annotations/captions/<source>`` directory (text, ``lang``, ``caption_type``,
    ``annotator_type`` / ``annotator_detail``, ``licence_class`` lifted from ``attrs``)."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    root = base_dir / "_annotations" / "captions"
    rows: list[dict] = []
    for source_dir in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        rows.extend(
            _free_config_rows(_annotation_free(base_dir, "captions", source_dir.name), label_status)
        )
    return ConfigResult("captions", tuple(rows), {})


def _build_free_table_config(config_id: str, base_dir: str | Path, sources: tuple[str, ...]):
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    rows: list[dict] = []
    for source_id in sources:
        records = _annotation_free(base_dir, config_id, source_id)
        rows.extend(_free_config_rows(records, label_status))
    return ConfigResult(config_id, tuple(rows), {})


def build_depth_config(base_dir: str | Path) -> ConfigResult:
    """``depth``: every unified ``depth`` row (``depth_ref``, ``units``, ``gt_type``,
    ``licence_class`` lifted from ``attrs``) of :data:`DEPTH_CONFIG_SOURCES`; pending rows (no
    ``image_sha256``) never reach the config. No crosswalk: depth carries no label."""
    return _build_free_table_config("depth", base_dir, DEPTH_CONFIG_SOURCES)


def build_pairs_config(base_dir: str | Path) -> ConfigResult:
    """``pairs``: every unified ``pairs`` row (``pair_role``, ``ref_image_sha256``,
    ``licence_class`` lifted from ``attrs``) of :data:`PAIRS_CONFIG_SOURCES`."""
    return _build_free_table_config("pairs", base_dir, PAIRS_CONFIG_SOURCES)
