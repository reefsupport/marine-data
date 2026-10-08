"""Task-layer configs: common (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from ...registry import Registry
from ...schema import Axis
from .tasklabels import _read_parquet, _row_sha


@dataclass(frozen=True)
class ConfigResult:
    """One config's built rows plus the bookkeeping the card/report needs."""

    config_id: str
    rows: tuple[dict, ...]
    unmapped_by_source: dict[str, float]
    """Fraction of native labels/pixels that had no crosswalk edge, per source_id —
    reported so a lossy crosswalk (e.g. Coralscapes' missing soft-coral row) is visible
    on the card, not just silently dropped."""

    @property
    def n_images(self) -> int:
        return len({_row_sha(row) for row in self.rows})


def _annotation_points(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``points`` rows of ``source_id`` (``_annotations/points/<source>/<version>.parquet``,
    newest version), or ``[]`` when none were written. Rows without an ``image_sha256`` are skipped."""  # noqa: E501
    root = base_dir / "_annotations" / "points" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_masks(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``masks`` rows of ``source_id`` (newest ``_annotations/masks/<source>/<version>.parquet``)."""  # noqa: E501
    root = base_dir / "_annotations" / "masks" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_image_labels(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``image_labels`` rows of ``source_id`` (newest ``_annotations/image_labels/<source>/<version>.parquet``)."""  # noqa: E501
    root = base_dir / "_annotations" / "image_labels" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_boxes(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``boxes`` rows of ``source_id`` (newest ``_annotations/boxes/<source>/<version>.parquet``)."""  # noqa: E501
    root = base_dir / "_annotations" / "boxes" / source_id
    files = sorted(p for p in root.glob("*.parquet") if not p.name.endswith(".pending.parquet"))
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_free(base_dir: Path, table: str, source_id: str) -> list[dict]:
    """The unified ``captions`` / ``vqa`` rows of ``source_id`` (newest non-pending parquet); rows
    that are pending (no ``image_sha256``) never reach a config."""
    root = base_dir / "_annotations" / table / source_id
    files = sorted(p for p in root.glob("*.parquet") if not p.name.endswith(".pending.parquet"))
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _free_config_rows(records: list[dict], label_status: Mapping[str, str]) -> list[dict]:
    return [
        {
            **r,
            "sha256": r["image_sha256"],
            "licence_class": _attrs(r).get("licence_class"),
            "label_status": label_status.get(r["image_sha256"], r["label_status"]),
        }
        for r in records
    ]


def _attrs(row: Mapping) -> dict:
    return json.loads(row["attrs"]) if row.get("attrs") else {}


@dataclass
class _SourceUnmapped:
    known: int = 0
    unmapped: int = 0

    def fraction(self) -> float:
        total = self.known + self.unmapped
        return self.unmapped / total if total else 0.0


@cache
def _harmonizer(registry: Registry, source_id: str):
    return registry.harmonizer_for(source_id)


@lru_cache(maxsize=1 << 16)
def _canonical(registry: Registry, source_id: str, native_label: str, axis: Axis) -> str | None:
    """Native label -> canonical node id on ``axis``, or ``None`` if unmappable (memoised:
    a full-scale points file repeats a few hundred labels across ~0.5M rows)."""
    harmonizer = _harmonizer(registry, source_id)
    if harmonizer is None:
        return None
    value = harmonizer.map_label(native_label).labels.get(axis)
    return value.node_id if value is not None else None


def _canonical_taxon(registry: Registry, source_id: str, native_label: str) -> str | None:
    """Native label -> canonical ``taxon`` node id, or ``None`` if unmappable."""
    return _canonical(registry, source_id, native_label, Axis.TAXON)
