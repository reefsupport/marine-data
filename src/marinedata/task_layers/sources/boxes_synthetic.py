"""Per-image JSON boxes of the generated seabed-debris set (WP-U6c): not tied to any source.

One document per composite image: ``{"image": "img_0000.png", "placed": [{"class": ..., "box":
[x1, y1, x2, y2]}], "fauna": [{"box": [...]}]}``, boxes in pixels of the composite. ``placed`` boxes
carry their debris class; ``fauna`` boxes have none, so their native label is the generic ``Fauna``.
"""

from __future__ import annotations

from collections.abc import Mapping

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box

FAUNA = "Fauna"


def read_synthetic_json(
    doc: Mapping[str, object], *, img_w: int | None, img_h: int | None
) -> tuple[tuple[RawBox, ...], BoxCounts]:
    """The boxes of one document; ``img_w``/``img_h`` are the composite's pixel size."""
    if not isinstance(doc, Mapping) or "placed" not in doc:
        raise BoxFormatError("not a synthetic-seabed label document (no `placed` list)")
    tally, out = Tally(), []
    for layer, items in (("placed", doc.get("placed")), ("fauna", doc.get("fauna"))):
        for obj in items or []:
            try:
                coords = tuple(float(v) for v in obj["box"])
            except (KeyError, TypeError, ValueError):
                raise BoxFormatError(f"{layer}: box is not 4 numbers: {str(obj)[:60]!r}") from None
            if len(coords) != 4:
                raise BoxFormatError(f"{layer}: box has {len(coords)} numbers")
            box = make_box(
                tally, coords, img_w=img_w, img_h=img_h, unit=False,
                attrs={"layer": layer, "synthetic": True},
                native=str(obj.get("class") or FAUNA),
            )  # fmt: skip
            if box is not None:
                out.append(box)
    return tuple(out), tally.freeze()
