"""Pascal VOC XML box reader (WP-U6b): not tied to any source.

One ``<annotation>`` per image: ``<filename>``, ``<size><width/><height/></size>`` and one
``<object>`` per box (``<name>``, ``<bndbox><xmin/><ymin/><xmax/><ymax/></bndbox>``, optional
``<truncated>``, ``<difficult>``, ``<occluded>``, ``<pose>``). Pixel corners are taken as written
(no 1-based adjustment). ``difficult`` / ``truncated`` / ``occluded`` are kept in ``attrs`` and the
box stays; boxes are clipped onto the image (counted) and dropped when no area is left (counted).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box

_FLAGS = ("difficult", "truncated", "occluded")


@dataclass(frozen=True)
class VocImage:
    filename: str | None
    width: int | None
    height: int | None
    boxes: tuple[RawBox, ...]


def _text(node: ET.Element | None, tag: str) -> str | None:
    child = None if node is None else node.find(tag)
    return None if child is None or child.text is None else child.text.strip() or None


def _int(v: str | None) -> int | None:
    try:
        return int(float(v)) if v else None
    except ValueError:
        raise BoxFormatError(f"VOC size is not a number: {v!r}") from None


def _flag(v: str | None) -> bool | None:
    return None if v is None else v not in ("0", "false", "False")


def read_voc(
    xml_text: str, *, img_w: int | None = None, img_h: int | None = None
) -> tuple[VocImage, BoxCounts]:
    """The boxes of one VOC annotation. ``img_w``/``img_h`` fill in a missing ``<size>``."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise BoxFormatError(f"not XML: {exc}") from None
    if root.tag != "annotation":
        raise BoxFormatError(f"VOC root is <{root.tag}>, expected <annotation>")
    size = root.find("size")
    w = _int(_text(size, "width")) or img_w
    h = _int(_text(size, "height")) or img_h
    tally = Tally()
    out: list[RawBox] = []
    for no, obj in enumerate(root.findall("object"), 1):
        bnd = obj.find("bndbox")
        try:
            coords = tuple(float(_text(bnd, k)) for k in ("xmin", "ymin", "xmax", "ymax"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise BoxFormatError(
                f"object {no}: bndbox without numeric xmin/ymin/xmax/ymax"
            ) from None
        attrs: dict[str, object] = {k: f for k in _FLAGS if (f := _flag(_text(obj, k))) is not None}
        if pose := _text(obj, "pose"):
            attrs["pose"] = pose
        box = make_box(
            tally,
            coords,  # type: ignore[arg-type]
            img_w=w,
            img_h=h,
            unit=False,
            attrs=attrs,
            native=_text(obj, "name"),
        )
        if box is not None:
            out.append(box)
    return VocImage(_text(root, "filename"), w, h, tuple(out)), tally.freeze()
