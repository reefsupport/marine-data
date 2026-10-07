"""Hackathon fish-boxes driver (HK-4c): staged local mirror -> ``boxes`` parquet per source.

    python scripts/hk_fish_boxes.py boxes --stage STAGE --data-dir DIR uiis uiis10k usis10k ...

Reads ``STAGE/<source_id>/`` (a mirror of ``sources/<id>/<version>/``) instead of the bucket (the
public fetch caps files at 64 MB; the COCO documents are 100 MB+), writes
``DIR/_annotations/boxes/<id>/<version>.parquet`` and an empty ``DIR/_tasklabels/`` so that
``marinedata release build --tasklabels-root DIR`` builds the ``boxes`` config.

    python scripts/hk_fish_boxes.py package --hf HF_EXPORT --data-dir DIR --out OUT

``package`` (see :func:`package`) joins an ``hf_export`` dir (``data/images`` shards +
``data/metadata`` v2 tables) with the boxes tables into the unified hackathon layout, the same
column names as ``out/<flavour>/coral-points``: ``OUT/data/<split>-NNNNN-of-NNNNN.parquet`` (one per
image shard, image bytes embedded, per-row licence/attribution/split/split_group, ``boxes`` =
list of normalised 0-1 ``x_min, y_min, x_max, y_max`` + ``label``, ``points`` empty), plus the flat
``OUT/annotations/boxes.parquet`` and ``OUT/build_stats.json``.
"""

from __future__ import annotations

import argparse
import collections
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.registry import Registry, _default_root
from marinedata.task_layers.boxes_table import BOX_SOURCES, staged_boxes, write_boxes
from marinedata.task_layers.s3_keyed import StagedTree


def local_fetch(stage: Path, source_id: str, version: str):
    prefix = f"sources/{source_id}/{version}/"

    def fetch(key: str) -> bytes:
        return (stage / source_id / key.removeprefix(prefix)).read_bytes()

    return fetch


def write_source_boxes(stage: Path, data_dir: Path, source_id: str, registry) -> dict:
    spec = BOX_SOURCES[source_id]
    tree = StagedTree(spec.tree, local_fetch(stage, source_id, spec.version))
    res = staged_boxes(spec, registry, tree=tree)
    write_boxes(data_dir, source_id, spec.version, list(res.rows))
    classes = {r["label_native"] for r in res.rows}
    return {
        "source": source_id, "images": res.images, "boxes": len(res.rows),
        "classes": len(classes), "orphan": res.counts.orphan, "unparsable": list(res.unparsable),
    }  # fmt: skip


def unified_features():
    """The ``out/<flavour>/coral-points`` feature set: one schema for every hackathon config."""
    import datasets as ds

    v = ds.Value
    point = {"x": v("float32"), "y": v("float32"), "label": v("string"), "t1_class": v("string")}
    box = {k: v("float32") for k in ("x_min", "y_min", "x_max", "y_max")} | {"label": v("string")}
    cols = {"image_sha256": v("string"), "image": ds.Image()}
    cols |= {
        c: v("string") for c in ("source_id", "split_group", "split", "license", "licence_class")
    }
    cols |= {c: v("string") for c in ("attribution", "upstream_id", "upstream_url")}
    cols |= {"lat": v("float64"), "lon": v("float64"), "depth_m": v("float64")}
    cols |= {"meow_realm": v("string"), "habitat": v("string")}
    cols |= {"width": v("int32"), "height": v("int32"), "points": [point], "boxes": [box]}
    return ds.Features(cols)


def unified_schema() -> pa.Schema:
    feats = unified_features()
    meta = {b"huggingface": json.dumps({"info": {"features": feats.to_dict()}}).encode()}
    return feats.arrow_schema.with_metadata(meta)


META_COLUMNS = ("split_group", "split", "license", "licence_class", "attribution", "upstream_id")
META_COLUMNS += ("upstream_url", "lat", "lon", "depth_m", "meow_realm", "habitat")
ANN_COLUMNS = ("image_sha256", "source_id", "ann_id", "label_native", "x_min", "y_min", "x_max")
ANN_COLUMNS += ("y_max", "x_min_px", "y_min_px", "x_max_px", "y_max_px", "is_crowd", "confidence")


def read_boxes(data_dir: Path) -> dict[tuple[str, str], list[dict]]:
    """``{(image_sha256, source_id): [box rows]}`` from ``DIR/_annotations/boxes/*/*.parquet``."""
    out: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for path in sorted((data_dir / "_annotations" / "boxes").glob("*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            out[(row["image_sha256"], row["source_id"])].append(row)
    return out


def package(hf_dir: Path, data_dir: Path, out_dir: Path) -> dict:
    """Join ``hf_dir/data/{images,metadata}`` with the boxes into ``out_dir`` (see module doc).

    An image with no box is left out (``images_without_boxes``): the config is "photos with boxes".
    The pair ``(image_sha256, source_id)`` keys the join, so a pixel-identical image that two
    sources both ship never inherits the other source's boxes."""
    from PIL import Image

    schema = unified_schema()
    meta = {}
    for path in sorted((hf_dir / "data" / "metadata").glob("*.parquet")):
        for row in pq.read_table(path).to_pylist():
            meta[(row["image_sha256"], row["source_id"])] = row
    boxes = read_boxes(data_dir)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "annotations").mkdir(parents=True, exist_ok=True)
    stats: collections.Counter = collections.Counter()
    per_split: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ann_rows: list[dict] = []
    for shard in sorted((hf_dir / "data" / "images").glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        writer = pq.ParquetWriter(out_dir / "data" / shard.name, schema, compression="zstd")
        for g in range(pf.num_row_groups):
            rows = []
            for r in pf.read_row_group(g).to_pylist():
                key = (r["image_sha256"], r["source_id"])
                found, m = boxes.get(key), meta[key]
                if not found:
                    stats["images_without_boxes"] += 1
                    continue
                try:
                    width, height = Image.open(io.BytesIO(r["image"]["bytes"])).size
                except Exception:  # an unreadable header leaves the size null
                    width = height = None
                packed = [{k: b[k] for k in ("x_min", "y_min", "x_max", "y_max")} for b in found]
                packed = [
                    p | {"label": b["label_native"]} for p, b in zip(packed, found, strict=True)
                ]
                row = {"image_sha256": r["image_sha256"], "image": r["image"], "source_id": key[1]}
                row |= {c: m[c] for c in META_COLUMNS} | {"width": width, "height": height}
                rows.append(row | {"points": [], "boxes": packed})
                ann_rows += [
                    {c: b[c] for c in ANN_COLUMNS}
                    | {c: m[c] for c in ("split", "split_group")}
                    | {
                        "license": m["license"],
                        "attribution": m["attribution"],
                        "label": b["label_native"],
                    }
                    for b in found
                ]
                stats["images"] += 1
                stats["boxes"] += len(found)
                stats["bytes"] += len(r["image"]["bytes"])
                per_split[m["split"]]["images"] += 1
                per_split[m["split"]]["boxes"] += len(found)
                per_split[m["split"]][f"source:{key[1]}"] += 1
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        writer.close()
    pq.write_table(
        pa.Table.from_pylist(ann_rows),
        out_dir / "annotations" / "boxes.parquet",
        compression="zstd",
    )
    classes = sorted({a["label"] for a in ann_rows})
    summary = {
        "config": "fish-boxes",
        **stats,
        "classes": classes,
        "per_split": {k: dict(v) for k, v in per_split.items()},
    }
    (out_dir / "build_stats.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["boxes", "package"])
    ap.add_argument("--stage", type=Path)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--hf", type=Path, help="package: the hf_export directory")
    ap.add_argument("--out", type=Path, help="package: output directory (data/, annotations/)")
    ap.add_argument("sources", nargs="*")
    a = ap.parse_args(argv)
    if a.cmd == "package":
        if a.hf is None or a.out is None:
            ap.error("package needs --hf and --out")
        print(json.dumps(package(a.hf, a.data_dir, a.out)))
        return 0
    if a.stage is None or not a.sources:
        ap.error("boxes needs --stage and at least one source id")
    (a.data_dir / "_tasklabels").mkdir(parents=True, exist_ok=True)
    reg = Registry.load(_default_root())
    stats = [write_source_boxes(a.stage, a.data_dir, s, reg) for s in a.sources]
    (a.data_dir / "boxes_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
