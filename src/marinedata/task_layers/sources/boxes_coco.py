"""COCO-style box reader (WP-U6): not tied to any source.

Three shapes of the same ``bbox = [x, y, w, h]`` (pixels, top-left origin) record are read:

* :func:`read_coco`: a full COCO document (``images`` / ``annotations`` / ``categories`` /
  ``licenses``); the per-image licence comes from ``images[].license`` -> ``licenses[]``;
* :func:`coco_annotation_boxes`: a bare list of COCO annotation dicts for one image (fragments kept
  in a kv table, e.g. an ``annotations_json`` cell);
* :func:`coco_columnar_boxes`: the Hugging Face ``objects`` layout, one dict of parallel columns
  (``id``, ``bbox``, ``category_id``, ...), as a dict or a one-element list.

``iscrowd`` becomes ``is_crowd``; ``ignore`` is kept in ``attrs`` (the box stays); boxes are
clipped onto the image (counted) and dropped when no area is left (counted).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box


@dataclass(frozen=True)
class CocoImage:
    image_id: object
    file_name: str | None
    width: int | None
    height: int | None
    licence: str | None
    boxes: tuple[RawBox, ...]


def _cid(v: object) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise BoxFormatError(f"bad category id {v!r}") from None


def coco_categories(doc: Mapping) -> dict[int, str]:
    return {int(c["id"]): str(c["name"]) for c in doc.get("categories") or [] if "name" in c}


def coco_licences(doc: Mapping) -> dict[int, str]:
    """``licence id -> name`` (the url when the entry has no name)."""
    return {
        int(lic["id"]): str(lic.get("name") or lic.get("url") or "")
        for lic in doc.get("licenses") or []
        if "id" in lic
    }


def _tally_annotations(
    tally: Tally,
    annotations: Sequence[Mapping],
    *,
    img_w: int | None,
    img_h: int | None,
    categories: Mapping[int, str] | None,
) -> list[RawBox]:
    out: list[RawBox] = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if bbox is None or len(bbox) != 4:
            raise BoxFormatError(f"COCO bbox needs 4 numbers, got {bbox!r}")
        x, y, w, h = (float(v) for v in bbox)
        cid = _cid(ann.get("category_id"))
        name = ann.get("category_name") or (categories or {}).get(cid)
        box = make_box(
            tally,
            (x, y, x + w, y + h),
            img_w=img_w,
            img_h=img_h,
            unit=False,
            is_crowd=bool(ann.get("iscrowd")),
            ignore=bool(ann.get("ignore")),
            native=None if name is None else str(name),
            native_id=None if cid is None else str(cid),
            confidence=None if ann.get("score") is None else float(ann["score"]),
        )
        if box is not None:
            out.append(box)
    return out


def coco_annotation_boxes(
    annotations: Sequence[Mapping],
    *,
    img_w: int | None,
    img_h: int | None,
    categories: Mapping[int, str] | None = None,
) -> tuple[tuple[RawBox, ...], BoxCounts]:
    tally = Tally()
    boxes = _tally_annotations(tally, annotations, img_w=img_w, img_h=img_h, categories=categories)
    return tuple(boxes), tally.freeze()


def coco_columnar_boxes(
    objects: Mapping | Sequence[Mapping],
    *,
    img_w: int | None,
    img_h: int | None,
    categories: Mapping[int, str] | None = None,
) -> tuple[tuple[RawBox, ...], BoxCounts]:
    """HF ``objects``: parallel lists ``bbox`` / ``category_id`` (and optional ``iscrowd``)."""
    parts = [objects] if isinstance(objects, Mapping) else list(objects)
    anns: list[dict] = []
    for part in parts:
        boxes = part.get("bbox") or []
        cats = part.get("category_id") or [None] * len(boxes)
        crowd = part.get("iscrowd") or [0] * len(boxes)
        if not (len(boxes) == len(cats) == len(crowd)):
            raise BoxFormatError("columnar objects have columns of different lengths")
        anns += [
            {"bbox": b, "category_id": c, "iscrowd": k}
            for b, c, k in zip(boxes, cats, crowd, strict=True)
        ]
    return coco_annotation_boxes(anns, img_w=img_w, img_h=img_h, categories=categories)


def read_coco(doc: Mapping) -> tuple[tuple[CocoImage, ...], BoxCounts]:
    """Every image of a COCO document with its boxes (images without annotations keep an empty
    tuple); annotations whose ``image_id`` is not in ``images`` are counted as ``orphan``."""
    categories, licences = coco_categories(doc), coco_licences(doc)
    by_image: dict[object, list[Mapping]] = {}
    for ann in doc.get("annotations") or []:
        by_image.setdefault(ann.get("image_id"), []).append(ann)
    known = {img.get("id") for img in doc.get("images") or []}
    tally = Tally()
    tally.bump("orphan", sum(len(v) for k, v in by_image.items() if k not in known))
    images: list[CocoImage] = []
    for img in doc.get("images") or []:
        w, h = img.get("width"), img.get("height")
        boxes = _tally_annotations(
            tally,
            by_image.get(img.get("id"), []),
            img_w=None if w is None else int(w),
            img_h=None if h is None else int(h),
            categories=categories,
        )
        lic = img.get("license")
        images.append(
            CocoImage(
                image_id=img.get("id"),
                file_name=img.get("file_name"),
                width=None if w is None else int(w),
                height=None if h is None else int(h),
                licence=(licences.get(int(lic)) or None) if lic is not None else None,
                boxes=tuple(boxes),
            )
        )
    return tuple(images), tally.freeze()
