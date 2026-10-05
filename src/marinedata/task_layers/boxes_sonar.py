"""Sonar (and synthetic-debris) boxes into the unified ``boxes`` table (WP-U6c).

Binds three staged sources to the format readers; :func:`staged_boxes` in :mod:`boxes_table`
dispatches here for the readers in :data:`SONAR_READERS`:

* ``sss-mine-detection`` (``yolo-parquet``): YOLO text per image in ``labels/image_labels.parquet``
  (``key == "txt"``). The class names are not staged, so the native label is the class id (0 / 1);
* ``swdd-sss-wall`` (``coco-doc-frames``): one COCO document
  (``labels/files/unresolved-0.json``) for
  the 1570 x 390 ``SSS-Video-frames`` images, matched to staged stems by the upstream file name;
* ``synthetic-seabed-debris`` (``synthetic-json``): per-image JSON (:mod:`sources.boxes_synthetic`)
  bound to the ``final`` composite of each image. The set is generated *optical* imagery, not sonar.

Side-scan sources carry ``attrs.modality = sonar``. ``aquascan-1k-sss`` has no boxes staged (its
only
label file lists the category ``Human``), so it has no reader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from .boxes_table import BoxResult, BoxSource, _Emit, _meta_int
from .image_labels_table import _stride
from .s3_keyed import StagedTree
from .sources.boxes_coco import read_coco
from .sources.boxes_common import BoxCounts, BoxFormatError
from .sources.boxes_synthetic import read_synthetic_json
from .sources.boxes_yolo import read_yolo

SONAR_READERS = ("yolo-parquet", "coco-doc-frames", "synthetic-json")
SONAR = {"modality": "sonar", "sonar_type": "side-scan"}
OPTICAL_SYNTHETIC = {"modality": "optical", "synthetic": True}
WALL_FRAMES = "SSS-Video-frames/Images/"


def _yolo_parquet(spec: BoxSource, registry, tree: StagedTree, limit) -> BoxResult:
    out = _Emit(spec, registry)
    rows = tree.table(spec.names_rel or "labels/image_labels.parquet")
    texts = {r["stem"]: r["value"] or "" for r in rows
             if r.get("key") == "txt"}  # fmt: skip
    meta = sorted(tree.table("metadata.parquet"), key=lambda m: m["stem"])
    meta = [
        m
        for m in _stride(meta, limit)
        if m["stem"] in texts and ("default", m["stem"]) in tree.shas
    ]
    for m in meta:
        try:
            boxes, counts = read_yolo(texts[m["stem"]], img_w=_meta_int(m.get("width")),
                                      img_h=_meta_int(m.get("height")))  # fmt: skip
        except BoxFormatError as exc:
            out.reject(m["stem"], exc)
            continue
        out.image(tree.shas[("default", m["stem"])], boxes, counts, attrs=SONAR)
    return out.result()


def _coco_frames(spec: BoxSource, registry, tree: StagedTree, limit) -> BoxResult:
    out = _Emit(spec, registry)
    doc = json.loads(tree.get("labels/files/unresolved-0.json"))
    meta = tree.table("metadata.parquet")
    by_name = {PurePosixPath(m["upstream_id"]).name: m["stem"] for m in meta
               if WALL_FRAMES in str(m.get("upstream_id"))}  # fmt: skip
    chosen = {
        im["id"] for im in _stride(sorted(doc["images"], key=lambda i: i["file_name"]), limit)
    }
    part = {**doc, "images": [i for i in doc["images"] if i["id"] in chosen],
            "annotations": [a for a in doc["annotations"] if a.get("image_id") in chosen],
            }  # fmt: skip
    images, counts = read_coco(part)
    out.counts = out.counts + counts
    for img in images:
        stem = by_name.get(PurePosixPath(img.file_name or "").name)
        sha = tree.shas.get(("default", stem)) if stem else None
        if sha is None:
            out.counts = out.counts + BoxCounts(orphan=1)
            continue
        out.image(sha, img.boxes, BoxCounts(), attrs=SONAR)
    return out.result()


def _synthetic(spec: BoxSource, registry, tree: StagedTree, limit, workers: int = 16) -> BoxResult:
    out = _Emit(spec, registry)
    final = {m["stem"].split("_final_")[-1]: m for m in tree.table("metadata.parquet")
             if "_final_img_" in m["stem"]}  # fmt: skip
    rels = _stride(sorted(r for r in tree.checksums if r.startswith("labels/files/")), limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for data in pool.map(tree.get_or_skip, rels):
            if data is None:
                out.counts = out.counts + BoxCounts(orphan=1)
                continue
            doc: Mapping = json.loads(data)
            m = final.get(PurePosixPath(str(doc.get("image"))).stem)
            sha = tree.shas.get(("default", m["stem"])) if m else None
            if sha is None:
                out.counts = out.counts + BoxCounts(orphan=1)
                continue
            try:
                boxes, counts = read_synthetic_json(doc, img_w=_meta_int(m.get("width")),
                                                    img_h=_meta_int(m.get("height")))  # fmt: skip
            except BoxFormatError as exc:
                out.reject(str(doc.get("image")), exc)
                continue
            out.image(
                sha, boxes, counts, attrs={**OPTICAL_SYNTHETIC, "generator": doc.get("source")}
            )
    return out.result()


def staged_sonar(spec: BoxSource, registry, tree: StagedTree, limit: int | None) -> BoxResult:
    return {"yolo-parquet": _yolo_parquet, "coco-doc-frames": _coco_frames,
            "synthetic-json": _synthetic}[spec.reader](spec, registry, tree, limit)  # fmt: skip
