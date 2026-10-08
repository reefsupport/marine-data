"""Task-layer configs: points (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

from pathlib import Path

from ...annotation_schema import annotator_from_origin
from ...registry import Registry
from .common import ConfigResult, _annotation_points, _attrs, _canonical_taxon, _SourceUnmapped
from .constants import POINT_SOURCES
from .tasklabels import _load_label_status, _read_tasklabels, _row_sha


def _point_records(base_dir: Path, source_id: str) -> list[dict]:
    """Point rows of one source in the unified column names. The unified table wins; else the
    legacy ``_tasklabels/<source>/points.parquet`` is renamed on read (``native_label`` ->
    ``label_native``, ``label_origin`` -> ``annotator_type``, ``sha256`` -> ``image_sha256``)
    with ``match_type`` left null (legacy rows were never resolved through ``Crosswalk.resolve``)."""  # noqa: E501
    unified = _annotation_points(base_dir, source_id)
    if unified:
        return unified
    out: list[dict] = []
    for i, point in enumerate(_read_tasklabels(base_dir, source_id, "points")):
        origin = point.get("label_origin")
        out.append(
            {
                "image_sha256": _row_sha(point),
                "source_id": source_id,
                "ann_id": f"{source_id}:{i}",
                "label_native": point.get("label_native") or point["native_label"],
                "annotator_type": point.get("annotator_type")
                or annotator_from_origin(origin or "human")[0],  # noqa: E501, RUF100
                "x": point["x"],
                "y": point["y"],
            }
        )
    return out


def build_points_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``points``: every :data:`POINT_SOURCES` point in the unified column names (WP-U3):
    ``image_sha256, source_id, ann_id, label_native, label_set, taxon_node_id, match_type,
    worms_aphia_id, rs_benthic_code, annotator_type, x, y, x_px, y_px, label_status``, plus the
    provenance and join columns ``source_version, ann_license, ann_attribution, quadrat_id,
    label_native_group, x_norm, y_norm``. ``x``/``y`` stay normalised 0-1 as in the unified
    schema and ``x_norm``/``y_norm`` repeat them under the explicit name; ``quadrat_id`` and
    ``label_native_group`` come from ``attrs`` and are null for sources that do not set them."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []

    for source_id in POINT_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for point in _point_records(base_dir, source_id):
            native = point["label_native"]
            taxon = point.get("taxon_node_id") or _canonical_taxon(registry, source_id, native)
            if taxon is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            attrs = _attrs(point)
            rows.append(
                {
                    "image_sha256": point["image_sha256"],
                    "source_id": source_id,
                    "source_version": point.get("source_version"),
                    "ann_id": point["ann_id"],
                    "quadrat_id": attrs.get("quadrat_id"),
                    "label_native": native,
                    "label_native_group": attrs.get("func_group"),
                    "label_set": point.get("label_set"),
                    "taxon_node_id": taxon,
                    "match_type": point.get("match_type")
                    or ("unmapped" if taxon is None else None),  # noqa: E501, RUF100
                    "worms_aphia_id": point.get("worms_aphia_id"),
                    "rs_benthic_code": point.get("rs_benthic_code"),
                    "annotator_type": point["annotator_type"],
                    "ann_license": point.get("ann_license"),
                    "ann_attribution": point.get("ann_attribution"),
                    "x": point["x"],
                    "y": point["y"],
                    "x_norm": point["x"],
                    "y_norm": point["y"],
                    "x_px": point.get("x_px"),
                    "y_px": point.get("y_px"),
                    "label_status": label_status.get(point["image_sha256"], "ok"),
                }
            )

    return ConfigResult("points", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()})
