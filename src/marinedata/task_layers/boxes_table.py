"""COCO / YOLO / FathomNet boxes into the unified ``boxes`` table (WP-U6a).

One row per box in :mod:`marinedata.annotation_schema`'s ``boxes`` table (normalised ``xyxy`` float32
in [0, 1] plus the pixel columns). The format readers live in ``sources/boxes_*.py`` and know no
source; this module binds five staged sources (:data:`BOX_SOURCES`) to them:

* ``fathomnet``: per-image JSON (``labels/files/<uuid>.json``), the image sha256 from the ingest
  stream index (``_stream/<tree>/part-*.json``); images the ingest could not fetch are *pending*;
* ``fathomnet-fgvc23``: COCO ``objects`` columns in ``labels/files/metadata.jsonl`` (category ids
  only: the id -> name table is not staged, so the native label is the id);
* ``fathomnet-fgvc25``: COCO ``annotations_json`` cells in ``labels/image_labels.parquet``;
* ``rf100-coral-lwptl``, ``roboflow-aquarium``: YOLO ``.txt`` + ``data.yaml``.

A source without a crosswalk (everything but rf100-coral-lwptl today; U8 adds the fauna ones)
records ``match_type = unmapped`` with the native label kept.

Licence per row: ``ann_license`` is the box's own licence (else the image's, else the source's);
``attrs.licence_class`` is ``open`` (CC0 / CC-BY / PD), ``restricted-nc`` or ``restricted-nd`` from
the strictest of the box and image licences, and the source's class when no row licence is known.
"""  # noqa: E501

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    arrow_schema,
    normalise_split,
    validate_row,
    write_annotations,
)
from ..registry import Registry
from .image_labels_table import Resolved, _splits, _stride, axes_resolver_for
from .masks_table import INTERNAL_ONLY, drop_internal_only, is_internal_only
from .points_table import _taxon_rank
from .s3_keyed import StagedTree, fetch_small
from .sources.boxes_coco import coco_annotation_boxes, coco_columnar_boxes
from .sources.boxes_common import UNLABELLED, BoxCounts, BoxFormatError, RawBox, pixel_columns
from .sources.boxes_fathomnet import read_fathomnet_json
from .sources.boxes_yolo import read_yolo, yolo_names

OPEN, NC, ND = "open", "restricted-nc", "restricted-nd"
_RANK = {OPEN: 0, NC: 1, ND: 2}
PENDING_COLUMNS = ("image_key",)
_PLACEHOLDER_SHA = "0" * 64
_CC = re.compile(r"^cc[- ]?(?P<rest>(?:by|0).*)$", re.I)


def normalise_licence(text: str | None) -> str | None:
    """A recognised Creative Commons / public-domain licence as an SPDX-style id
    (``CC-BY-NC-ND-4.0``), else ``None`` (``"FathomNet"`` is not a licence)."""
    if not text:
        return None
    raw = str(text).strip()
    if raw.upper() in {"PD", "PUBLIC DOMAIN", "US-GOV-PD"}:
        return "US-GOV-PD" if raw.upper() == "US-GOV-PD" else "PD"
    m = _CC.match(raw)
    if m is None:
        return None
    rest = m["rest"].lower().replace("_", "-").replace(" ", "-")
    if rest.startswith("0"):
        return "CC0-1.0"
    tokens = [t for t in rest.split("-") if t]
    mods = [t for t in tokens if t in {"by", "nc", "nd", "sa"}]
    version = next((t for t in tokens if re.fullmatch(r"\d(\.\d)?", t)), None)
    out = "CC-" + "-".join(m.upper() for m in mods)
    return f"{out}-{version if version and '.' in version else (version or '4') + '.0'}"


def licence_class_of(*licences: str | None, default: str) -> str:
    """The strictest class among the recognised ``licences`` (ND > NC > open), else ``default``."""
    found = [normalise_licence(x) for x in licences]
    classes = []
    for spdx in filter(None, found):
        parts = spdx.split("-")
        classes.append(ND if "ND" in parts else NC if "NC" in parts else OPEN)
    return max(classes, key=_RANK.__getitem__) if classes else default


@dataclass(frozen=True)
class BoxSource:
    source_id: str
    version: str
    reader: str  # fathomnet | coco-columnar | coco-fragments | yolo
    annotator: str
    ann_license: str | None  # the source's licence (fallback for ann_license)
    licence_class: str  # the source's class (fallback for attrs.licence_class)
    crosswalk_id: str | None = None
    label_set: str = "dataset-native"
    names_rel: str | None = None  # data.yaml of the YOLO sets
    stream_parts: int = 0  # fathomnet: number of _stream part files

    @property
    def tree(self) -> str:
        return f"{self.source_id}/{self.version}"


# Source classes: fathomnet = lic-A.tsv (restricted-nd, per-contributor); the other four state
# CC-BY-4.0 in their LICENSE and are in neither lic TSV, so the registry tier (T1) = open.
BOX_SOURCES: dict[str, BoxSource] = {
    s.source_id: s
    for s in (
        BoxSource("fathomnet", "fathomnet-f4f9b794691e", "fathomnet", "human", None, ND,
                  label_set="fathomnet-concepts", stream_parts=188),
        BoxSource("fathomnet-fgvc23", "rev-4636bed20b3b", "coco-columnar", "human", "CC-BY-4.0",
                  OPEN, label_set="fgvc23-category-id"),
        BoxSource("fathomnet-fgvc25", "2025", "coco-fragments", "human", "CC-BY-4.0", OPEN,
                  label_set="fgvc25-categories"),
        BoxSource("rf100-coral-lwptl", "rev-83f0a33679b0", "yolo", "human_crowd", "CC-BY-4.0",
                  OPEN, crosswalk_id="rf100-coral-lwptl", label_set="rf100-coral-lwptl",
                  names_rel="labels/files/data.yaml"),
        BoxSource("roboflow-aquarium", "zip-5d30fed3dd5a", "yolo", "human_crowd", "CC-BY-4.0",
                  OPEN, label_set="aquarium-combined",
                  names_rel="docs/aquarium_pretrain/data.yaml"),
    )
}  # fmt: skip


def resolver_for_source(registry: Registry, spec: BoxSource) -> Callable[[str], Resolved]:
    if spec.crosswalk_id is not None:
        return axes_resolver_for(registry, spec.crosswalk_id)
    unmapped = Resolved(None, None, None, "unmapped", None, None)
    return lambda _native: unmapped


def box_row(
    *,
    spec: BoxSource,
    registry: Registry,
    resolve: Callable[[str], Resolved],
    ordinal: int,
    image_sha256: str | None,
    box: RawBox,
    split: str | None = None,
    extra_attrs: Mapping[str, object] | None = None,
    image_licence: str | None = None,
) -> dict:
    """One ``boxes`` row; ``attrs.licence_class`` always set (a missing native label is the
    ``__unlabelled`` sentinel, unmapped)."""
    # no name but an id (fgvc23 ships ids only): the id is the native label; neither: sentinel
    native = box.native or box.native_id or UNLABELLED
    got = resolve(native)
    typ, detail = annotator_from_origin(spec.annotator)
    row_licence = normalise_licence(box.licence) or normalise_licence(image_licence)
    attrs = {**dict(extra_attrs or {}), **dict(box.attrs)}
    if box.native is None:
        attrs["no_category" if box.native_id is None else "name_unavailable"] = True
    if image_licence and normalise_licence(image_licence) != row_licence:
        attrs["image_licence"] = image_licence
    elif image_licence:
        attrs["image_licence"] = normalise_licence(image_licence)
    attrs["licence_class"] = licence_class_of(
        box.licence, image_licence, default=spec.licence_class
    )
    return {
        "image_sha256": image_sha256,
        "source_id": spec.source_id,
        "source_version": spec.version,
        "ann_id": f"{spec.source_id}:{ordinal}",
        "label_native": native,
        "label_native_id": box.native_id,
        "label_set": spec.label_set,
        "taxon_node_id": got.taxon,
        "form_node_id": got.form,
        "condition_node_id": got.condition,
        "taxon_rank": _taxon_rank(registry, got.taxon),
        "worms_aphia_id": got.worms_aphia_id,
        "rs_benthic_code": got.rs_benthic_code,
        "match_type": got.match_type,
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": row_licence or spec.ann_license,
        "confidence": box.confidence,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(attrs, sort_keys=True),
        "x_min": box.x_min,
        "y_min": box.y_min,
        "x_max": box.x_max,
        "y_max": box.y_max,
        **pixel_columns(box),
        "is_crowd": box.is_crowd,
        "img_w": box.img_w,
        "img_h": box.img_h,
    }


@dataclass(frozen=True)
class BoxResult:
    rows: tuple[dict, ...]
    pending: tuple[dict, ...]  # boxes of images that are not staged (no image_sha256)
    counts: BoxCounts
    images: int  # staged images with at least one row
    unparsable: tuple[str, ...]  # label files that raised BoxFormatError (first few reasons)
    unlabelled_images: int = 0  # staged images whose label file holds no box


class _Emit:
    """Row ordinals and tallies shared by one source run."""

    def __init__(self, spec: BoxSource, registry: Registry) -> None:
        self.spec, self.registry = spec, registry
        self.resolve = resolver_for_source(registry, spec)
        self.rows: list[dict] = []
        self.pending: list[dict] = []
        self.counts = BoxCounts()
        self.images = 0
        self.empty = 0
        self.bad: list[str] = []

    def image(
        self,
        sha: str | None,
        boxes: Iterable[RawBox],
        counts: BoxCounts,
        *,
        split=None,
        attrs=None,
        image_licence=None,
        image_key=None,
    ) -> None:
        boxes = list(boxes)
        self.counts = self.counts + counts
        if not boxes:
            self.empty += 1
            return
        target = self.rows if sha is not None else self.pending
        for b in boxes:
            row = box_row(
                spec=self.spec, registry=self.registry, resolve=self.resolve,
                ordinal=len(self.rows) + len(self.pending), image_sha256=sha, box=b,
                split=split, extra_attrs=attrs, image_licence=image_licence,
            )  # fmt: skip
            if sha is None:
                row["image_key"] = image_key
            target.append(row)
        self.images += sha is not None

    def reject(self, what: str, exc: Exception) -> None:
        self.bad.append(f"{what}: {exc}"[:160])

    def result(self) -> BoxResult:
        return BoxResult(tuple(self.rows), tuple(self.pending), self.counts, self.images,
                         tuple(self.bad[:10]), self.empty)  # fmt: skip


def _meta_int(v: object) -> int | None:
    try:
        return int(v) if v not in (None, "", "None") else None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def _label_refs(raw: object) -> list[str]:
    if isinstance(raw, str):
        raw = ast.literal_eval(raw) if raw.startswith("[") else [raw]
    return list(raw or [])


def fathomnet_stream_index(
    spec: BoxSource, fetch: Callable[[str], bytes] = fetch_small, workers: int = 16
) -> tuple[dict[str, str], list[str]]:
    """``({uuid: image sha256}, [uuid of images the ingest could not fetch])`` from the stream
    part files (``index[].member`` = ``<uuid>.<ext>``; ``missing[][0]`` = the unfetched member)."""

    def part(n: int) -> dict:
        return json.loads(
            fetch(f"sources/{spec.source_id}/_stream/{spec.version}/part-{n:05d}.json")
        )

    staged: dict[str, str] = {}
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for doc in pool.map(part, range(spec.stream_parts)):
            for e in doc.get("index") or []:
                staged[e["member"].rsplit(".", 1)[0]] = e["sha256"]
            missing += [m[0].rsplit(".", 1)[0] for m in doc.get("missing") or []]
    return staged, missing


def _fathomnet(spec, registry, fetch, limit, workers=16) -> BoxResult:
    staged, missing = fathomnet_stream_index(spec, fetch, workers)
    keys = [(u, staged[u]) for u in _stride(sorted(staged), limit)]
    pend = _stride(sorted(set(missing)), limit) if missing else []
    out = _Emit(spec, registry)

    def get(uuid: str) -> dict | None:
        try:
            return json.loads(fetch(f"sources/{spec.tree}/labels/files/{uuid}.json"))
        except OSError:
            return None

    docs = [(u, s) for u, s in keys] + [(u, None) for u in pend]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (uuid, sha), doc in zip(docs, pool.map(get, [u for u, _ in docs]), strict=True):
            if doc is None:
                out.reject(uuid, OSError("label JSON not readable"))
                continue
            try:
                img, counts = read_fathomnet_json(doc)
            except BoxFormatError as exc:
                out.reject(uuid, exc)
                continue
            attrs = {} if sha is None or img.sha256 in (None, sha) else {"upstream_sha_differs": 1}
            out.image(sha, img.boxes, counts, attrs=attrs, image_licence=img.licence,
                      image_key=f"{uuid}")  # fmt: skip
    return out.result()


def _fgvc23(spec, registry, tree, limit) -> BoxResult:
    out = _Emit(spec, registry)
    meta = tree.table("metadata.parquet")
    stem_of = {m["upstream_id"].rsplit("/", 1)[-1]: m for m in meta}
    splits = _splits(tree)
    lines = [
        json.loads(x) for x in tree.get("labels/files/metadata.jsonl").decode().splitlines() if x
    ]
    for rec in _stride(sorted(lines, key=lambda r: r["file_name"]), limit):
        m = stem_of.get(rec["file_name"])
        sha = None if m is None else tree.shas.get(("default", m["stem"]))
        if sha is None:
            out.counts = out.counts + BoxCounts(orphan=1)
            continue
        try:
            boxes, counts = coco_columnar_boxes(
                rec.get("objects") or [], img_w=_meta_int(rec.get("width")),
                img_h=_meta_int(rec.get("height")),
            )  # fmt: skip
        except BoxFormatError as exc:
            out.reject(rec["file_name"], exc)
            continue
        out.image(sha, boxes, counts, split=splits.get(("default", m["stem"])))
    return out.result()


def _fgvc25(spec, registry, tree, limit) -> BoxResult:
    out = _Emit(spec, registry)
    by_stem: dict[str, dict[str, str]] = {}
    for r in tree.table("labels/image_labels.parquet"):
        by_stem.setdefault(r["stem"], {})[r["key"]] = r["value"]
    meta = {m["stem"]: m for m in tree.table("metadata.parquet")}
    for stem in _stride(sorted(by_stem), limit):
        sha, cells = tree.shas.get(("default", stem)), by_stem[stem]
        if sha is None or "annotations_json" not in cells:
            out.counts = out.counts + BoxCounts(orphan=1)
            continue
        m = meta.get(stem, {})
        try:
            boxes, counts = coco_annotation_boxes(
                json.loads(cells["annotations_json"]),
                img_w=_meta_int(m.get("width") or cells.get("width")),
                img_h=_meta_int(m.get("height") or cells.get("height")),
            )  # fmt: skip
        except (BoxFormatError, ValueError) as exc:
            out.reject(stem, exc)
            continue
        out.image(sha, boxes, counts, split=cells.get("split"))
    return out.result()


def _yolo(spec, registry, tree, limit, workers=16) -> BoxResult:
    out = _Emit(spec, registry)
    names = yolo_names(tree.get(spec.names_rel).decode())
    meta = sorted(tree.table("metadata.parquet"), key=lambda m: m["stem"])
    meta = [m for m in _stride(meta, limit) if ("default", m["stem"]) in tree.shas]

    def read(m: Mapping) -> bytes | None:
        refs = _label_refs(m.get("label_refs"))
        return tree.get_or_skip(refs[0]) if refs else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for m, data in zip(meta, pool.map(read, meta), strict=True):
            if data is None:
                out.counts = out.counts + BoxCounts(orphan=1)
                continue
            try:
                boxes, counts = read_yolo(data.decode(), names=names,
                                          img_w=_meta_int(m.get("width")),
                                          img_h=_meta_int(m.get("height")))  # fmt: skip
            except (BoxFormatError, UnicodeDecodeError) as exc:
                out.reject(m["stem"], exc)
                continue
            split = m.get("upstream_split") or m.get("split_hint")
            out.image(tree.shas[("default", m["stem"])], boxes, counts, split=split)
    return out.result()


def staged_boxes(
    spec: BoxSource,
    registry: Registry,
    *,
    limit: int | None = None,
    tree: StagedTree | None = None,
    fetch: Callable[[str], bytes] = fetch_small,
) -> BoxResult:
    """Read one source's staged labels, at most ``limit`` images (evenly spaced, deterministic)."""
    if spec.reader == "fathomnet":
        return _fathomnet(spec, registry, fetch, limit)
    tree = tree or StagedTree(spec.tree, fetch)
    return {"coco-columnar": _fgvc23, "coco-fragments": _fgvc25, "yolo": _yolo}[spec.reader](
        spec, registry, tree, limit
    )


def write_boxes(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    """Write rows to ``<data_dir>/_annotations/boxes/<source_id>/<version>.parquet``."""
    path = annotation_path(data_dir, "boxes", source_id, source_version)
    write_annotations(path, "boxes", rows)
    return path


def validate_pending(rows: Iterable[Mapping[str, object]]) -> list[str]:
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_COLUMNS}
        core["image_sha256"] = _PLACEHOLDER_SHA
        errs += [f"{row['ann_id']}: {e}" for e in validate_row("boxes", core)]
        if not row.get("image_key"):
            errs.append(f"{row['ann_id']}: image_key missing")
    return errs


def write_pending_boxes(path: Path, rows: list[dict]) -> int:
    """Boxes of images that are not staged: ``boxes`` columns minus ``image_sha256`` plus
    ``image_key`` (the upstream image id), validated against a placeholder sha."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if errs := validate_pending(rows):
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema("boxes")
    fields = [f for f in base if f.name != "image_sha256"] + [
        pa.field(c, pa.string()) for c in PENDING_COLUMNS
    ]
    table = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in fields},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return len(rows)


def licence_split(rows: Iterable[Mapping[str, object]]) -> dict[str, int]:
    return dict(Counter(json.loads(r["attrs"])["licence_class"] for r in rows))


__all__ = [
    "BOX_SOURCES", "INTERNAL_ONLY", "BoxResult", "BoxSource", "box_row", "drop_internal_only",
    "is_internal_only", "licence_class_of", "licence_split", "normalise_licence", "staged_boxes",
    "write_boxes", "write_pending_boxes",
]  # fmt: skip
