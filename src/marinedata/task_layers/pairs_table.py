"""Image pairs into the unified ``pairs`` table (WP-U11).

One row per (image, reference image): ``image_sha256`` is the anchor frame, ``ref_image_sha256``
its partner, ``pair_role`` the **role of the reference** relative to the anchor:

* ``enhanced``: the reference is the enhanced / clean version of the anchor (anchor = degraded input);
* ``degraded``: the reference is a degraded version of the anchor;
* ``stereo_right``: the anchor is the left view and the reference the right view;
* ``sonar``: the reference is the sonar frame time-aligned with the (camera) anchor.

``attrs.pair_kind`` (``enhancement`` / ``restoration`` / ``stereo`` / ``cross-modal`` / ``temporal``)
names the pairing family and ``attrs.licence_class`` is always set. A pair whose sha256 is unknown on
either side is a *pending* row: ``image_key`` / ``ref_image_key`` name the staged stems and both
sha256 columns are left out (as for pending boxes). A set whose reference is not staged builds no
row at all and is logged as a gap (euvp: the reference column was lost in staging).
"""

# ruff: noqa: E501

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from ..annotation_schema import annotation_path, arrow_schema, validate_row, write_annotations
from .depth_table import DepthPairSource
from .vqa_table import free_row

PAIR_KINDS = frozenset({"enhancement", "restoration", "stereo", "cross-modal", "temporal"})
PENDING_PAIR_COLUMNS = ("image_key", "ref_image_key")
_PLACEHOLDER_SHA = "0" * 64
__all__ = [
    "PAIR_KINDS", "PENDING_PAIR_COLUMNS", "pair_row", "pending_pair_path", "validate_pending_pairs",
    "write_pairs", "write_pending_pairs",
]  # fmt: skip


def pair_row(
    *,
    spec: DepthPairSource,
    ordinal: int,
    sha: str | None,
    ref_sha: str | None,
    pair_role: str,
    pair_kind: str,
    split: str | None = None,
    attrs: Mapping[str, object] | None = None,
) -> dict:
    """One ``pairs`` row; a None sha on either side = pending (the caller adds the keys)."""
    if pair_kind not in PAIR_KINDS:
        raise ValueError(f"pair_kind {pair_kind!r} not in {sorted(PAIR_KINDS)}")
    return free_row(
        spec=spec,
        ordinal=ordinal,
        sha=sha,
        split=split,
        attrs={**dict(attrs or {}), "pair_kind": pair_kind},
        payload={"pair_role": pair_role, "ref_image_sha256": ref_sha},
    )


def validate_pending_pairs(rows: Iterable[Mapping[str, object]]) -> list[str]:
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_PAIR_COLUMNS}
        core |= {"image_sha256": _PLACEHOLDER_SHA, "ref_image_sha256": _PLACEHOLDER_SHA}
        errs += [f"{row['ann_id']}: {e}" for e in validate_row("pairs", core)]
        errs += [f"{row['ann_id']}: {c} missing" for c in PENDING_PAIR_COLUMNS if not row.get(c)]
    return errs


def write_pairs(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    path = annotation_path(data_dir, "pairs", source_id, source_version)
    write_annotations(path, "pairs", rows)
    return path


def pending_pair_path(data_dir: Path, source_id: str, source_version: str) -> Path:
    return annotation_path(data_dir, "pairs", source_id, source_version).with_suffix(
        ".pending.parquet"
    )


def write_pending_pairs(path: Path, rows: list[dict]) -> int:
    """The table's columns minus both sha256 columns plus ``image_key`` / ``ref_image_key``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if errs := validate_pending_pairs(rows):
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema("pairs")
    dropped = {"image_sha256", "ref_image_sha256"}
    fields = [f for f in base if f.name not in dropped] + [
        pa.field(c, pa.string()) for c in PENDING_PAIR_COLUMNS
    ]
    tbl = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in fields},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, path, compression="zstd", write_statistics=True)
    return len(rows)
