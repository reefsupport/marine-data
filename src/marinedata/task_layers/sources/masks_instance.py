"""General marine *instance* masks into the unified ``masks`` table (WP-U12).

One ``mask_kind=instance`` row per COCO annotation of the Lian/Li underwater instance-segmentation
sets (:data:`INSTANCE_SOURCES`: ``usis10k``, ``uiis``, ``uiis10k``; all three share the ``usis-uiis``
crosswalk and an Apache-2.0 licence). The geometry is the COCO polygon (``polygon``, a JSON list of
rings) or, for a dict segmentation, the COCO RLE (``rle``); ``mask_ref`` stays null (nothing is
rasterised). Each instance's own category name is ``label_native`` and resolves through
:meth:`marinedata.schema.Crosswalk.resolve` exactly like WP-U4 (same ``match_type`` vocabulary).

``instance_id`` (the upstream COCO annotation id), the crowd flag, the box and the image size live in
``attrs`` (the ``masks`` table has no column for them; U1). ``attrs.modality`` is ``optical`` here and
``sonar`` for the acoustic sets, so a later reader can split them without a join.

USIS10K ships a class-agnostic ``foreground`` and a ``multi_class`` document over the *same* instances;
only ``multi_class`` is read, so no instance is counted twice. Sources with no staged labels (every
sonar set, usod, mas3k-hf, rmas-hf, atlantis, marineinst20m-web) are listed in :data:`U7_GAPS`: no
row is built for them.
"""  # noqa: E501

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath

from ...annotation_schema import annotator_from_origin, normalise_split
from ...registry import Registry
from ..points_table import _taxon_rank, resolver_for
from ..s3_keyed import StagedTree

LABEL_SET = "usis-uiis-categories"
OPEN = "open"
Resolver = Callable[[str], tuple[str | None, str, int | None, str | None]]


@dataclass(frozen=True)
class InstanceSource:
    """One staged COCO instance-segmentation set (documents, stem rule)."""

    source_id: str
    version: str
    docs: tuple[tuple[str, str], ...]  # (rel path of the COCO json, split)
    stem_prefix: str  # prepended to ``[<split>_]<file stem>`` by :func:`staged_stem`
    crosswalk_id: str = "usis-uiis"
    annotator: str = "human"
    ann_license: str | None = "Apache-2.0"
    licence_class: str = OPEN  # no lic-A/B row; the staged LICENSE is Apache-2.0
    label_set: str = LABEL_SET
    modality: str = "optical"

    @property
    def tree(self) -> str:
        return f"{self.source_id}/{self.version}"


_F = "labels/files/"
INSTANCE_SOURCES: dict[str, InstanceSource] = {
    s.source_id: s
    for s in (
        InstanceSource(
            "usis10k", "rev-b10f41ab819b",
            tuple((f"{_F}USIS10K_multi_class_annotations_multi_class_{sp}_annotations.json", sp)
                  for sp in ("train", "val", "test")),
            "USIS10K_zip_USIS10K_",
        ),
        InstanceSource(
            "uiis", "rev-44ca5db46599",
            ((f"{_F}annotations_train.json", "train"), (f"{_F}annotations_val.json", "val")),
            "",
        ),
        InstanceSource(
            "uiis10k", "rev-5c4316368a8e",
            ((f"{_F}UIIS10K_annotations_multiclass_train.json", "train"),
             (f"{_F}UIIS10K_annotations_multiclass_test.json", "test")),
            "UIIS10K_zip_UIIS10K_img_",
        ),
    )
}  # fmt: skip

# Sources whose labels are not staged (or whose images are not): no rows, a U7 staging gap.
U7_GAPS: dict[str, str] = {
    "pingmapper-sss-seg": "sonar (side-scan) masks: labels not staged; modality=sonar when staged",
    "aris-didson-fish-td": "sonar (ARIS/DIDSON) boxes: labels not staged; modality=sonar",
    "uatd": "sonar (UATD) boxes: labels not staged",
    "usod": "salient masks + points: labels not staged (IRVL page 403)",
    "mas3k-hf": "masks not staged (HF mirror, lic-A internal-only)",
    "rmas-hf": "masks not staged (HF mirror, lic-A internal-only)",
    "atlantis": "semantic masks not staged",
    "marineinst20m-web": "instance masks not staged (lic-B internal-only, scraped images)",
}  # fmt: skip


def staged_stem(spec: InstanceSource, split: str, file_name: str) -> list[str]:
    """Candidate staged stems of a COCO ``file_name`` (the split may or may not be in the name)."""
    stem = PurePosixPath(file_name).stem
    p = spec.stem_prefix
    return list(dict.fromkeys([f"{p}{split}_{stem}", f"{p}{stem}", f"{split}_{stem}", stem]))


def _geometry(seg: object) -> tuple[str | None, str | None]:
    """``(polygon JSON, rle JSON)``: a ring list is a polygon, a dict is a COCO RLE."""
    if isinstance(seg, Mapping):
        return None, json.dumps(seg, sort_keys=True)
    if isinstance(seg, list) and seg:
        return json.dumps(seg, separators=(",", ":")), None
    return None, None


def instance_row(
    *,
    spec: InstanceSource,
    resolve: Resolver,
    registry: Registry,
    ordinal: int,
    image_sha256: str,
    ann: Mapping[str, object],
    native: str,
    split: str | None,
    img_w: int | None = None,
    img_h: int | None = None,
) -> dict | None:
    """One instance ``masks`` row, or ``None`` when the annotation has no geometry."""
    polygon, rle = _geometry(ann.get("segmentation"))
    if polygon is None and rle is None:
        return None
    taxon, match, aphia, l2 = resolve(native)
    typ, detail = annotator_from_origin(spec.annotator)
    attrs = {
        "instance_id": str(ann.get("id")),
        "is_crowd": bool(ann.get("iscrowd")),
        "bbox_xywh": ann.get("bbox"),
        "area": ann.get("area"),
        "img_w": img_w,
        "img_h": img_h,
        "licence_class": spec.licence_class,
        "modality": spec.modality,
    }
    return {
        "image_sha256": image_sha256,
        "source_id": spec.source_id,
        "source_version": spec.version,
        "ann_id": f"{spec.source_id}:{ordinal}",
        "label_native": native,
        "label_native_id": str(ann.get("category_id")),
        "label_set": spec.label_set,
        "taxon_node_id": taxon,
        "form_node_id": None,
        "condition_node_id": None,
        "taxon_rank": _taxon_rank(registry, taxon),
        "worms_aphia_id": aphia,
        "rs_benthic_code": l2,
        "match_type": match,
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": spec.ann_license,
        "confidence": None,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(attrs, sort_keys=True),
        "mask_kind": "instance",
        "mask_ref": None,
        "polygon": polygon,
        "rle": rle,
        "class_map": None,
        "ignore_value": None,
        "class_counts": None,
        "canonical_class_counts": None,
    }


def rows_from_doc(
    spec: InstanceSource,
    doc: Mapping[str, object],
    split: str,
    shas: Mapping[tuple[str, str], str],
    resolve: Resolver,
    registry: Registry,
    *,
    start: int = 0,
) -> tuple[list[dict], Counter]:
    """Rows of one COCO document plus a tally (orphan_image, no_geometry, no_category)."""
    cats = {c["id"]: c["name"] for c in doc.get("categories", [])}  # type: ignore[union-attr]
    images = {im["id"]: im for im in doc.get("images", [])}  # type: ignore[union-attr]
    tally: Counter = Counter()
    rows: list[dict] = []
    for ann in sorted(doc.get("annotations", []), key=lambda a: a["id"]):  # type: ignore[union-attr]
        im = images.get(ann.get("image_id"))
        sha = None
        if im is not None:
            sha = next(
                (shas[("default", s)] for s in staged_stem(spec, split, im["file_name"])
                 if ("default", s) in shas), None)  # fmt: skip
        if sha is None:
            tally["orphan_image"] += 1
            continue
        native = cats.get(ann.get("category_id"))
        if native is None:
            tally["no_category"] += 1
            continue
        row = instance_row(
            spec=spec, resolve=resolve, registry=registry, ordinal=start + len(rows),
            image_sha256=sha, ann=ann, native=native, split=split,
            img_w=im.get("width"), img_h=im.get("height"),
        )  # fmt: skip
        if row is None:
            tally["no_geometry"] += 1
            continue
        rows.append(row)
    return rows, tally


def stride(rows: list[dict], limit: int | None) -> list[dict]:
    """``limit`` rows spread evenly over ``rows`` (all of them when ``limit`` is None or large)."""
    if limit is None or len(rows) <= limit:
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]


def staged_instance_rows(
    tree: StagedTree,
    spec: InstanceSource,
    registry: Registry,
    *,
    limit: int | None = None,
) -> tuple[list[dict], Counter]:
    """All instance rows of ``spec`` (an even ``limit`` sample), and the read tally."""
    resolve = resolver_for(registry, spec.crosswalk_id)
    rows: list[dict] = []
    tally: Counter = Counter()
    for rel, split in spec.docs:
        got, t = rows_from_doc(spec, json.loads(tree.get(rel)), split, tree.shas, resolve,
                               registry, start=len(rows))  # fmt: skip
        rows += got
        tally += t
    tally["rows_total"] = len(rows)
    return stride(rows, limit), tally


def mapped_pct(rows: Iterable[Mapping[str, object]]) -> float:
    """Share of rows whose ``match_type`` is not ``unmapped``."""
    rs = list(rows)
    return round(100 * sum(r["match_type"] != "unmapped" for r in rs) / max(len(rs), 1), 2)


__all__ = [  # noqa: RUF022
    "INSTANCE_SOURCES",
    "InstanceSource",
    "U7_GAPS",
    "instance_row",
    "mapped_pct",
    "rows_from_doc",
    "staged_instance_rows",
    "staged_stem",
    "stride",
]
