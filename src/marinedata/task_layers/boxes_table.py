"""COCO / YOLO / FathomNet / MOT boxes into the unified ``boxes`` table (WP-U6a, WP-U6b).

One row per box in :mod:`marinedata.annotation_schema`'s ``boxes`` table (normalised ``xyxy`` float32
in [0, 1] plus the pixel columns). The format readers live in ``sources/boxes_*.py`` and know no
source; this module binds five staged sources (:data:`BOX_SOURCES`) to them:

* ``fathomnet``: per-image JSON (``labels/files/<uuid>.json``), the image sha256 from the ingest
  stream index (``_stream/<tree>/part-*.json``); images the ingest could not fetch are *pending*;
* ``fathomnet-fgvc23``: COCO ``objects`` columns in ``labels/files/metadata.jsonl`` (category ids
  only: the id -> name table is not staged, so the native label is the id);
* ``fathomnet-fgvc25``: COCO ``annotations_json`` cells in ``labels/image_labels.parquet``;
* ``rf100-coral-lwptl``, ``roboflow-aquarium``: YOLO ``.txt`` + ``data.yaml``;
* ``ruod``: two COCO documents (``labels/files/coco_annotations_instances_<split>.json``);
* ``brackishmot``: MOT ``gt.txt`` per sequence (CSV, :data:`MOT_GT`) + ``seqinfo.ini`` sizes;
* ``obsea-fish``: flat YOLO ``.txt`` (``labels/files``) with no ``metadata.parquet`` / CHECKSUMS in
  the staged tree: every box is *pending* (``image_key`` = the upstream image stem, no sha).

The two FGVC sets get a per-image licence: the FathomNet ``imageLicense`` of the image with the same
uuid (:mod:`boxes_licence_join`); an unmatched image is ``restricted-nd``, never ``open``.

A source without a crosswalk (everything but rf100-coral-lwptl today; U8 adds the fauna ones)
records ``match_type = unmapped`` with the native label kept.

Licence per row: ``ann_license`` is the box's own licence (else the image's, else the source's);
``attrs.licence_class`` is ``open`` (CC0 / CC-BY / PD), ``restricted-nc`` or ``restricted-nd`` from
the strictest of the box and image licences, and the source's class when no row licence is known.
"""  # noqa: E501

from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    arrow_schema,
    normalise_split,
    validate_row,
    write_annotations,
)
from ..registry import Registry, _default_root
from .boxes_licence_join import FathomnetLicenceJoin, JoinedLicence, image_uuid
from .image_labels_table import Resolved, _splits, _stride, axes_resolver_for
from .masks_table import INTERNAL_ONLY, drop_internal_only, is_internal_only
from .points_table import _taxon_rank
from .s3_keyed import StagedTree, fetch_small
from .sources.boxes_coco import coco_annotation_boxes, coco_columnar_boxes, read_coco
from .sources.boxes_common import UNLABELLED, BoxCounts, BoxFormatError, RawBox, pixel_columns
from .sources.boxes_csv import MOT_GT, parse_seqinfo, read_csv_boxes
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
    reader: str  # fathomnet | coco-columnar | coco-fragments | yolo | coco-docs | mot | yolo-flat
    annotator: str
    ann_license: str | None  # the source's licence (fallback for ann_license)
    licence_class: str  # the source's class (fallback for attrs.licence_class)
    crosswalk_id: str | None = None
    label_set: str = "dataset-native"
    names_rel: str | None = None  # data.yaml of the YOLO sets
    stream_parts: int = 0  # fathomnet: number of _stream part files
    licence_join: bool = False  # image licence = the matching FathomNet image's (FGVC sets)
    id_key: str | None = None  # registry-relative CSV (category_id,name): ids-only sets get names

    @property
    def tree(self) -> str:
        return f"{self.source_id}/{self.version}"


FGVC23_KEY = "taxonomy/vocab/fathomnet-fgvc23-category-key.csv"  # category_id -> concept (U8a)


# Source classes: fathomnet = lic-A.tsv (restricted-nd, per-contributor); rf100-coral-lwptl and
# roboflow-aquarium state CC-BY-4.0 in their LICENSE and are in neither lic TSV, so the registry tier  # noqa: E501
# (T1) = open; obsea-fish is T1 CC-BY-4.0 (open); the FGVC sets' LICENSE covers the annotations
# only, so their class is the per-image FathomNet one and restricted-nd where there is none;
# ruod (NOASSERTION, lic-A internal-only) and brackishmot (licence unknown) are internal-only.
BOX_SOURCES: dict[str, BoxSource] = {
    s.source_id: s
    for s in (
        BoxSource("fathomnet", "fathomnet-f4f9b794691e", "fathomnet", "human", None, ND,
                  crosswalk_id="fathomnet-concepts", label_set="fathomnet-concepts",
                  stream_parts=188),
        BoxSource("fathomnet-fgvc23", "rev-4636bed20b3b", "coco-columnar", "human", "CC-BY-4.0",
                  ND, crosswalk_id="fathomnet-concepts", label_set="fathomnet-concepts",
                  licence_join=True, id_key=FGVC23_KEY),
        BoxSource("fathomnet-fgvc25", "2025", "coco-fragments", "human", "CC-BY-4.0", ND,
                  crosswalk_id="fathomnet-concepts", label_set="fathomnet-concepts",
                  licence_join=True),
        BoxSource("rf100-coral-lwptl", "rev-83f0a33679b0", "yolo", "human_crowd", "CC-BY-4.0",
                  OPEN, crosswalk_id="rf100-coral-lwptl", label_set="rf100-coral-lwptl",
                  names_rel="labels/files/data.yaml"),
        BoxSource("roboflow-aquarium", "zip-5d30fed3dd5a", "yolo", "human_crowd", "CC-BY-4.0",
                  OPEN, crosswalk_id="roboflow-aquarium", label_set="aquarium-combined",
                  names_rel="docs/aquarium_pretrain/data.yaml"),
        BoxSource("ruod", "rev-c22094e45b7f", "coco-docs", "human", "NOASSERTION",
                  INTERNAL_ONLY, crosswalk_id="ruod", label_set="ruod-categories"),
        BoxSource("brackishmot", "zip-e717dc1438aa", "mot", "human", "NOASSERTION",
                  INTERNAL_ONLY, crosswalk_id="brackishmot-class-id",
                  label_set="brackishmot-class-id"),
        BoxSource("obsea-fish", "v1", "yolo-flat", "human", "CC-BY-4.0", OPEN,
                  crosswalk_id="obsea-fish", label_set="obsea-fish-species",
                  names_rel="labels/files/23sp_4120img_34945annots_2688res_data.yaml"),
    )
}  # fmt: skip


def load_id_names(rel: str) -> dict[str, str]:
    """``{category_id: concept name}`` from a registry-relative key CSV (``#`` header lines)."""
    path = Path(_default_root()) / rel
    rows = [x for x in path.read_text().splitlines() if x and not x.startswith("#")]
    return {r["category_id"]: r["name"] for r in csv.DictReader(rows)}


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
    own = normalise_licence(box.licence)
    # a joined set's LICENSE is the annotations' licence: the image's goes to attrs.image_licence only  # noqa: E501
    row_licence = own or (None if spec.licence_join else normalise_licence(image_licence))
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
        self.id_names = load_id_names(spec.id_key) if spec.id_key else {}
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
        boxes = [self._named(b) for b in boxes]
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

    def _named(self, box: RawBox) -> RawBox:
        """An ids-only box (FGVC 2023) gets its concept name from the source's category key; the id
        stays in ``label_native_id``. An id the key lacks stays ids-only (and unmapped)."""
        name = self.id_names.get(box.native_id or "") if box.native is None else None
        return replace(box, native=name) if name else box

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


def _joined_attrs(j: JoinedLicence) -> tuple[dict[str, str], str | None]:
    return {"licence_join": j.status}, j.licence


def _fgvc23(spec, registry, tree, limit, join) -> BoxResult:
    out = _Emit(spec, registry)
    meta = tree.table("metadata.parquet")
    stem_of = {m["upstream_id"].rsplit("/", 1)[-1]: m for m in meta}
    splits = _splits(tree)
    lines = [
        json.loads(x) for x in tree.get("labels/files/metadata.jsonl").decode().splitlines() if x
    ]
    recs = _stride(sorted(lines, key=lambda r: r["file_name"]), limit)
    joined = join.lookup(image_uuid(r["file_name"]) for r in recs)
    for rec in recs:
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
        attrs, lic = _joined_attrs(joined[image_uuid(rec["file_name"])])
        out.image(sha, boxes, counts, split=splits.get(("default", m["stem"])), attrs=attrs,
                  image_licence=lic)  # fmt: skip
    return out.result()


def _fgvc25(spec, registry, tree, limit, join) -> BoxResult:
    out = _Emit(spec, registry)
    by_stem: dict[str, dict[str, str]] = {}
    for r in tree.table("labels/image_labels.parquet"):
        by_stem.setdefault(r["stem"], {})[r["key"]] = r["value"]
    meta = {m["stem"]: m for m in tree.table("metadata.parquet")}
    stems = _stride(sorted(by_stem), limit)
    joined = join.lookup(image_uuid(s) for s in stems)
    for stem in stems:
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
        attrs, lic = _joined_attrs(joined[image_uuid(stem)])
        out.image(sha, boxes, counts, split=cells.get("split"), attrs=attrs, image_licence=lic)
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


def _coco_docs(spec, registry, tree, limit) -> BoxResult:
    """``ruod``: one COCO document per split; ``coco_<split>_<file stem>`` is the staged stem."""
    out = _Emit(spec, registry)
    docs = {
        sp: json.loads(tree.get(f"labels/files/coco_annotations_instances_{sp}.json"))
        for sp in ("train", "val")
    }

    def stem(split: str, img: Mapping) -> str:
        return f"coco_{split}_{PurePosixPath(img['file_name']).stem}"

    pairs = sorted(((sp, im) for sp, d in docs.items() for im in d["images"]),
                   key=lambda p: stem(*p))  # fmt: skip
    for sp, doc in docs.items():
        ids = {im["id"] for s, im in _stride(pairs, limit) if s == sp}
        part = {  # only the chosen images, so the counts are theirs
            **doc,
            "images": [im for im in doc["images"] if im["id"] in ids],
            "annotations": [a for a in doc["annotations"] if a.get("image_id") in ids],
        }
        images, counts = read_coco(part)
        out.counts = out.counts + counts
        for img in images:
            sha = tree.shas.get(("default", stem(sp, {"file_name": img.file_name})))
            if sha is None:
                out.counts = out.counts + BoxCounts(orphan=1)
                continue
            out.image(sha, img.boxes, BoxCounts(), split=sp)
    return out.result()


def _mot(spec, registry, tree, limit, workers=16) -> BoxResult:
    """``brackishmot``: ``<seq>_gt_gt.txt`` (MOT CSV) + ``<seq>_seqinfo.ini``; frame ``n`` is the
    image ``<seq>_img1_<n:06d>``."""
    out = _Emit(spec, registry)
    meta = {m["stem"]: m for m in tree.table("metadata.parquet")
            if ("default", m["stem"]) in tree.shas}  # fmt: skip
    stems = _stride(sorted(meta), limit)
    seqs: dict[str, list[tuple[str, int]]] = {}
    for stem in stems:
        seq, _, frame = stem.rpartition("_img1_")
        if not seq or not frame.isdigit():
            out.counts = out.counts + BoxCounts(orphan=1)
            continue
        seqs.setdefault(seq, []).append((stem, int(frame)))

    def read(seq: str) -> tuple[dict | None, tuple[int | None, int | None]]:
        raw = tree.get_or_skip(f"labels/files/{seq}_gt_gt.txt")
        info = tree.get_or_skip(f"labels/files/{seq}_seqinfo.ini")
        size = parse_seqinfo(info.decode()) if info else (None, None)
        first = meta[seqs[seq][0][0]]
        size = (size[0] or _meta_int(first.get("width")), size[1] or _meta_int(first.get("height")))
        if raw is None:
            return None, size
        return {int(k): v for k, v in read_csv_boxes(raw.decode(), MOT_GT, img_w=size[0],
                                                     img_h=size[1]).items()}, size  # fmt: skip

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for seq, (frames, _size) in zip(seqs, pool.map(read, seqs), strict=True):
            for stem, frame in seqs[seq]:
                if frames is None:
                    out.counts = out.counts + BoxCounts(orphan=1)
                    continue
                boxes, counts = frames.get(frame, ((), BoxCounts()))
                m = meta[stem]
                out.image(tree.shas[("default", stem)], boxes, counts,
                          split=m.get("upstream_split") or m.get("split_hint"))  # fmt: skip
    return out.result()


_OBSEA_LABEL = re.compile(
    r"^(?P<head>.+)_(?P<split>train|valid|val|test)_labels_(?P<name>.+)\.txt$"
)


def bucket_lister(bucket: str = "rs-storage-open") -> Callable[[str], list[str]]:
    """Flat, paginated ``list_objects_v2`` of a prefix (credentials by configparser, never echoed)."""  # noqa: E501
    import configparser
    import os

    import boto3

    cfg = configparser.ConfigParser()
    cfg.read(os.path.expanduser("~/.config/rclone/rclone.conf"))
    sec = cfg["rs-hel1"]
    endpoint = sec["endpoint"]
    client = boto3.client(
        "s3",
        aws_access_key_id=sec["access_key_id"],
        aws_secret_access_key=sec["secret_access_key"],
        endpoint_url=endpoint if endpoint.startswith("http") else f"https://{endpoint}",
    )

    def list_keys(prefix: str) -> list[str]:
        pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix)
        return [o["Key"] for page in pages for o in page.get("Contents", [])]

    return list_keys


def _yolo_flat(spec, registry, limit, fetch, lister, workers=16) -> BoxResult:
    """``obsea-fish``: YOLO ``.txt`` files in ``labels/files`` and no metadata / CHECKSUMS in the
    staged tree, so no image sha: every box is pending, keyed by the upstream image stem
    (``<head>_<split>_images_<name>``)."""
    base = f"sources/{spec.tree}/"
    names = yolo_names(fetch(base + spec.names_rel).decode())
    keys = sorted(k for k in lister(base + "labels/files/") if k.endswith(".txt"))
    out = _Emit(spec, registry)
    keys = _stride(keys, limit)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, data in zip(keys, pool.map(lambda k: fetch(k).decode(), keys), strict=True):
            stem = key.rsplit("/", 1)[-1]
            m = _OBSEA_LABEL.match(stem)
            if m is None:
                out.reject(stem, BoxFormatError("not a <head>_<split>_labels_<name>.txt name"))
                continue
            try:
                boxes, counts = read_yolo(data, names=names)
            except BoxFormatError as exc:
                out.reject(stem, exc)
                continue
            image_key = f"{m['head']}_{m['split']}_images_{m['name']}"
            out.image(None, boxes, counts, split=m["split"], image_key=image_key)
    return out.result()


def staged_boxes(
    spec: BoxSource,
    registry: Registry,
    *,
    limit: int | None = None,
    tree: StagedTree | None = None,
    fetch: Callable[[str], bytes] = fetch_small,
    join: FathomnetLicenceJoin | None = None,
    lister: Callable[[str], list[str]] | None = None,
) -> BoxResult:
    """Read one source's staged labels, at most ``limit`` images (evenly spaced, deterministic).

    ``join`` (FGVC sets) defaults to one over the staged ``fathomnet`` source; ``lister`` (obsea-fish,
    which has no manifest) defaults to :func:`bucket_lister`."""  # noqa: E501
    if spec.reader == "fathomnet":
        return _fathomnet(spec, registry, fetch, limit)
    if spec.reader == "yolo-flat":
        return _yolo_flat(spec, registry, limit, fetch, lister or bucket_lister())
    tree = tree or StagedTree(spec.tree, fetch)
    if spec.licence_join:
        join = join or fathomnet_join(fetch)
        return {"coco-columnar": _fgvc23, "coco-fragments": _fgvc25}[spec.reader](
            spec, registry, tree, limit, join
        )
    return {"coco-docs": _coco_docs, "mot": _mot, "yolo": _yolo}[spec.reader](
        spec, registry, tree, limit
    )


def fathomnet_join(fetch: Callable[[str], bytes] = fetch_small) -> FathomnetLicenceJoin:
    """The FGVC licence join over the staged ``fathomnet`` source (index of the staged uuids
    from the ingest stream parts, then the per-image JSON of each uuid asked for)."""
    fn = BOX_SOURCES["fathomnet"]
    return FathomnetLicenceJoin(
        lambda: fathomnet_stream_index(fn, fetch)[0],
        lambda uuid: fetch(f"sources/{fn.tree}/labels/files/{uuid}.json"),
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
    "BOX_SOURCES", "INTERNAL_ONLY", "BoxResult", "BoxSource", "box_row", "bucket_lister",
    "drop_internal_only", "fathomnet_join", "is_internal_only", "licence_class_of", "licence_split", "normalise_licence", "staged_boxes",  # noqa: E501
    "write_boxes", "write_pending_boxes",
]  # fmt: skip
