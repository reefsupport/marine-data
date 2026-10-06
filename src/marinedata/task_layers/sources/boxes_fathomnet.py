"""FathomNet per-image JSON reader (WP-U6): the ``boundingBoxes`` of one image record.

The record (the FathomNet API image object) carries ``width``/``height``, ``imageLicense`` and a
list ``boundingBoxes`` of ``{concept, x, y, width, height, annotationLicense, groupOf, occluded,
truncated, reviewState, ...}`` in pixels, top-left origin. ``concept`` is the native label; the
licence is per box (``annotationLicense``) and per image (``imageLicense``). The ``observer`` and
``contributorsEmail`` fields are personal data and are never read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box


@dataclass(frozen=True)
class FathomnetImage:
    uuid: str | None
    sha256: str | None
    width: int | None
    height: int | None
    licence: str | None
    boxes: tuple[RawBox, ...]


def read_fathomnet_json(doc: Mapping) -> tuple[FathomnetImage, BoxCounts]:
    w, h = doc.get("width"), doc.get("height")
    w, h = (None if w is None else int(w)), (None if h is None else int(h))
    tally = Tally()
    boxes: list[RawBox] = []
    for bb in doc.get("boundingBoxes") or []:
        try:
            x, y, bw, bh = (float(bb[k]) for k in ("x", "y", "width", "height"))
        except (KeyError, TypeError, ValueError):
            raise BoxFormatError("FathomNet box without numeric x/y/width/height") from None
        attrs = {
            k: bb[v]
            for k, v in (
                ("review_state", "reviewState"),
                ("occluded", "occluded"),
                ("truncated", "truncated"),
                ("alt_concept", "altConcept"),
            )
            if bb.get(v) is not None
        }
        box = make_box(
            tally,
            (x, y, x + bw, y + bh),
            img_w=w,
            img_h=h,
            unit=False,
            is_crowd=bool(bb.get("groupOf")),
            attrs=attrs,
            native=bb.get("concept"),
            licence=bb.get("annotationLicense"),
        )
        if box is not None:
            boxes.append(box)
    image = FathomnetImage(
        uuid=doc.get("uuid"),
        sha256=doc.get("sha256"),
        width=w,
        height=h,
        licence=doc.get("imageLicense"),
        boxes=tuple(boxes),
    )
    return image, tally.freeze()
