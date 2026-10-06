"""Labelbox export rasteriser: Labelbox ``export-result.ndjson`` polygons to per-image class
counts, keyed by the already-staged sources' sha256 (D-Z2, WP-8d). Renamed on 2026-10-06 (RB-3)
from ``rs_labelled`` (legacy: module name); the ``rs-labelled-masks`` id is retired (D7), its
ndjson exports now live under ``labels/exports/`` of ``reef-support-seaview-labels`` and
``reef-support-benthic-own``.

The module has no pixel-mask copy of its own: every image it annotates already has a known sha256
in one of those two staged sources' ``metadata.parquet`` (both D-D ``staged-tree``), so a row here
is keyed by looking its ``data_row.external_id`` filename up in that already-staged metadata — no
image bytes are read.

**Real-data finding (WP-8d, verified 2026-09-25):** every annotation object sampled
across ``SEAFLOWER_BOLIVAR/export-result.ndjson`` (246 rows, 2,578 objects) has
``annotation_kind: "ImageSegmentationMask"`` with only a ``mask.url`` pointing at
``api.labelbox.com`` — never inline polygon geometry. The brief's "rasterise the
polygons" premise does not hold for the live export: there is no local geometry to
rasterise, and the Do-Not-list forbids fetching that Labelbox-hosted mask URL. This
module still implements :func:`rasterize_record` for the inline-polygon case (Labelbox
ndjson *does* support a ``polygon``/``line`` object shape with an inline point list, per
its export schema, and the fixed columns/tests below cover it), but
:func:`class_counts_for_site` reports every ``ImageSegmentationMask``-only record as
*unusable* rather than fabricating counts from the class name alone (D-Z2's
``class_counts`` field is pixel-fraction, not presence). See
``docs/task-labels-producers.md`` for the resume path once a polygon export exists (or
D-Z2 is revisited to allow model-side mask fetch under a different rule).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCE_ID = "reef-support-seaview-labels"
LABEL_ORIGIN = "human"  # no label-origin.yaml exists in this repo yet; see docs note.


@dataclass(frozen=True)
class RasterResult:
    external_id: str
    class_counts: dict[str, int] | None
    """``None`` when the record has no inline polygon geometry (mask-URL-only)."""
    usable: bool


def _polygon_points(obj: dict[str, Any]) -> list[tuple[float, float]] | None:
    """Labelbox's inline polygon shape: ``{"polygon": [{"x":..,"y":..}, ...]}``."""
    poly = obj.get("polygon")
    if not poly:
        return None
    try:
        return [(float(p["x"]), float(p["y"])) for p in poly]
    except (KeyError, TypeError, ValueError):
        return None


def rasterize_record(record: dict[str, Any]) -> RasterResult:
    """One ndjson row (one image) -> its per-class pixel counts, or ``usable=False``.

    Iterates every project's every label's ``annotations.objects``; an
    ``ImageSegmentationMask`` object (mask hosted on Labelbox) makes the whole record
    unusable (never fetched). A ``polygon`` object is filled at the image's native
    resolution (``media_attributes.width``/``height``) and its pixels added to that
    class's count; overlapping polygons of different classes double-count their overlap
    (each layer is independent, matching how Labelbox stores them) — a documented
    approximation, not a compositing bug.
    """
    from PIL import Image, ImageDraw

    external_id = str(record.get("data_row", {}).get("external_id", ""))
    media = record.get("media_attributes", {}) or {}
    width, height = media.get("width"), media.get("height")
    counts: dict[str, int] = {}
    saw_object = False
    for proj in (record.get("projects") or {}).values():
        for label in proj.get("labels", []) or []:
            for obj in label.get("annotations", {}).get("objects", []) or []:
                saw_object = True
                if obj.get("annotation_kind") == "ImageSegmentationMask" or "mask" in obj:
                    return RasterResult(external_id, None, usable=False)
                points = _polygon_points(obj)
                if points is None or not width or not height:
                    return RasterResult(external_id, None, usable=False)
                name = str(obj.get("name", "unknown"))
                canvas = Image.new("L", (int(width), int(height)), 0)
                ImageDraw.Draw(canvas).polygon(points, fill=1)
                counts[name] = counts.get(name, 0) + int(canvas.histogram()[1])
    if not saw_object:
        return RasterResult(external_id, {}, usable=True)
    return RasterResult(external_id, counts, usable=True)


def sha256_lookup(metadata_paths: list[Path]) -> dict[str, str]:
    """``upstream_id``/basename -> ``image_sha256``, from one or more staged
    ``metadata.parquet`` files (the ``images_from`` sources)."""
    import pyarrow.parquet as pq

    out: dict[str, str] = {}
    for path in metadata_paths:
        table = pq.read_table(path, columns=["upstream_id", "image_sha256"])
        for upstream_id, sha in zip(
            table.column("upstream_id").to_pylist(),
            table.column("image_sha256").to_pylist(),
            strict=True,
        ):
            if upstream_id:
                out[Path(str(upstream_id)).name] = sha
    return out


def build_semseg_rows(
    ndjson_lines: list[str], sha_by_basename: dict[str, str]
) -> tuple[list[dict[str, Any]], int, int]:
    """Returns (rows, usable_count, unusable_count). Rows unmatched to a known sha256
    are dropped (can't be keyed) and counted separately by the caller if needed."""
    rows: list[dict[str, Any]] = []
    usable = unusable = 0
    for line in ndjson_lines:
        if not line.strip():
            continue
        record = json.loads(line)
        result = rasterize_record(record)
        if not result.usable:
            unusable += 1
            continue
        usable += 1
        sha = sha_by_basename.get(Path(result.external_id).name)
        if sha is None or not result.class_counts:
            continue
        rows.append(
            {
                "sha256": sha,
                "source_id": SOURCE_ID,
                "label_origin": LABEL_ORIGIN,
                "mask_key": None,
                "class_counts": json.dumps(result.class_counts, sort_keys=True),
            }
        )
    return rows, usable, unusable


def write_semseg_parquet(rows: list[dict[str, Any]], out_path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path, compression="zstd")
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ndjson", type=Path, nargs="+", required=True)
    p.add_argument(
        "--metadata", type=Path, nargs="+", required=True, help="staged metadata.parquet files"
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    sha_by_basename = sha256_lookup(args.metadata)
    lines: list[str] = []
    for path in args.ndjson:
        lines.extend(path.read_text().splitlines())
    rows, usable, unusable = build_semseg_rows(lines, sha_by_basename)
    n = write_semseg_parquet(rows, args.out)
    print(json.dumps({"rows": n, "usable_records": usable, "unusable_records": unusable}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
