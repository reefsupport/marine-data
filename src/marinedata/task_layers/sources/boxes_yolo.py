"""YOLO ``.txt`` box reader (WP-U6): not tied to any source.

One line per object, normalised: ``class cx cy w h`` (centre xywh). Also read: a sixth number
(a confidence), and the segmentation form ``class x1 y1 x2 y2 ...`` (the polygon's bounding box,
``attrs.from_polygon``). Blank lines and ``#`` comments are skipped; an empty file is zero boxes.
Class names come from a ``data.yaml`` (:func:`yolo_names`: ``names`` as a list or an id map).
"""

from __future__ import annotations

from collections.abc import Mapping

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box


def yolo_names(yaml_text: str) -> dict[int, str]:
    import yaml

    names = (yaml.safe_load(yaml_text) or {}).get("names")
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    raise BoxFormatError("data.yaml has no `names` list or map")


def read_yolo(
    text: str,
    *,
    names: Mapping[int, str] | None = None,
    img_w: int | None = None,
    img_h: int | None = None,
) -> tuple[tuple[RawBox, ...], BoxCounts]:
    """The boxes of one label file. ``img_w``/``img_h`` (optional) feed the pixel columns."""
    tally = Tally()
    out: list[RawBox] = []
    for no, line in enumerate(text.splitlines(), 1):
        parts = line.split("#", 1)[0].split()
        if not parts:
            continue
        try:
            cls = int(parts[0])
            nums = [float(p) for p in parts[1:]]
        except ValueError:
            raise BoxFormatError(f"line {no}: not a YOLO row: {line[:60]!r}") from None
        conf = None
        extra: dict[str, object] = {}
        if len(nums) in (4, 5):
            cx, cy, w, h = nums[:4]
            conf = nums[4] if len(nums) == 5 else None
            coords = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            xs, ys = nums[0::2], nums[1::2]
            coords = (min(xs), min(ys), max(xs), max(ys))
            extra["from_polygon"] = True
        else:
            raise BoxFormatError(f"line {no}: {len(nums)} numbers after the class id")
        name = (names or {}).get(cls)
        box = make_box(
            tally,
            coords,
            img_w=img_w,
            img_h=img_h,
            unit=True,
            attrs=extra,
            native=name if name is not None else str(cls),
            native_id=str(cls),
            confidence=conf,
        )
        if box is not None:
            out.append(box)
    return tuple(out), tally.freeze()
