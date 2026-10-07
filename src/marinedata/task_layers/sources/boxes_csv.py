"""CSV box reader (WP-U6b): not tied to any source.

One row per box; a :class:`CsvLayout` names the columns, so the same reader reads a headed
annotation export and a headerless MOT ``gt.txt`` (:data:`MOT_GT`). Rows are grouped by the
``image`` column (one group, key ``""``, when the layout has none). Box formats: ``xywh`` (top-left
corner + size), ``xyxy`` and ``cxcywh``, in pixels or, with ``unit=True``, in [0, 1]. Trailing empty
cells (MOT writes a final comma) are ignored. A row whose ``ignore_if`` column holds the stated value
keeps its box with ``attrs.ignore`` (MOT's "do not consider" flag); boxes are clipped onto the image
(counted) and dropped when no area is left (counted).
"""  # noqa: E501

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from dataclasses import dataclass, field

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box

_FORMATS = ("xywh", "xyxy", "cxcywh")


@dataclass(frozen=True)
class CsvLayout:
    coords: tuple[str, str, str, str]  # the four column names, in the order of ``fmt``
    fmt: str = "xywh"
    unit: bool = False  # coordinates already in [0, 1]
    names: tuple[str, ...] | None = None  # header of a headerless file (None: first row is it)
    delimiter: str = ","
    image: str | None = None
    label: str | None = None
    label_id: str | None = None
    confidence: str | None = None
    ignore_if: tuple[str, str] | None = None  # (column, value) that marks a box to ignore
    crowd: str | None = None  # a column that is 1/true for a crowd box
    img_w: str | None = None
    img_h: str | None = None
    licence: str | None = None
    attrs: tuple[str, ...] = field(default_factory=tuple)  # columns kept as attrs[column]


# MOT Challenge ground truth: frame, track id, bb_left, bb_top, bb_width, bb_height, flag
# (1 = considered, 0 = ignore; NOT a confidence), class id, visibility.
MOT_GT = CsvLayout(
    coords=("x", "y", "w", "h"),
    names=("frame", "track_id", "x", "y", "w", "h", "flag", "class", "visibility"),
    image="frame",
    label_id="class",
    ignore_if=("flag", "0"),
    attrs=("track_id", "visibility"),
)


def _num(row: Mapping[str, str], col: str, no: int) -> float:
    try:
        return float(row[col])
    except (KeyError, TypeError, ValueError):
        raise BoxFormatError(f"row {no}: column {col!r} is not a number") from None


def _opt_int(row: Mapping[str, str], col: str | None) -> int | None:
    if not col or not (row.get(col) or "").strip():
        return None
    try:
        return int(float(row[col]))
    except ValueError:
        return None


def _scalar(v: str) -> object:
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            continue
    return v


def read_csv_boxes(
    text: str,
    layout: CsvLayout,
    *,
    img_w: int | None = None,
    img_h: int | None = None,
) -> dict[str, tuple[tuple[RawBox, ...], BoxCounts]]:
    """``{image key: (boxes, counts)}``. ``img_w``/``img_h`` apply to rows with no size column."""
    if layout.fmt not in _FORMATS:
        raise BoxFormatError(f"unknown box format {layout.fmt!r}")
    rows = csv.reader(io.StringIO(text), delimiter=layout.delimiter)
    header = list(layout.names) if layout.names else next(rows, None)
    if header is None:
        return {}
    header = [h.strip() for h in header]
    missing = [c for c in layout.coords if c not in header]
    if missing:
        raise BoxFormatError(f"CSV has no column {missing[0]!r}")
    need = max(header.index(c) for c in layout.coords) + 1
    tallies: dict[str, Tally] = {}
    boxes: dict[str, list[RawBox]] = {}
    for no, cells in enumerate(rows, 2 if layout.names is None else 1):
        while cells and not cells[-1].strip():
            cells = cells[:-1]
        if not cells:
            continue
        if len(cells) < need:  # trailing empty cells (stripped above) may be absent, coords may not
            raise BoxFormatError(f"row {no}: {len(cells)} cells, the box needs {need}")
        cells = [*cells, *[""] * (len(header) - len(cells))]
        row = {h: c.strip() for h, c in zip(header, cells, strict=False)}
        a, b, c, d = (_num(row, col, no) for col in layout.coords)
        coords = {
            "xywh": (a, b, a + c, b + d),
            "xyxy": (a, b, c, d),
            "cxcywh": (a - c / 2, b - d / 2, a + c / 2, b + d / 2),
        }[layout.fmt]
        key = row.get(layout.image, "") if layout.image else ""
        ignore = (
            layout.ignore_if is not None and row.get(layout.ignore_if[0]) == layout.ignore_if[1]
        )
        conf = (
            _num(row, layout.confidence, no)
            if layout.confidence and row.get(layout.confidence)
            else None
        )
        attrs = {k: _scalar(row[k]) for k in layout.attrs if row.get(k)}
        tally = tallies.setdefault(key, Tally())
        box = make_box(
            tally,
            coords,
            img_w=_opt_int(row, layout.img_w) or img_w,
            img_h=_opt_int(row, layout.img_h) or img_h,
            unit=layout.unit,
            is_crowd=bool(layout.crowd) and row.get(layout.crowd, "") in ("1", "true", "True"),
            ignore=ignore,
            attrs=attrs,
            native=row.get(layout.label) or None if layout.label else None,
            native_id=row.get(layout.label_id) or None if layout.label_id else None,
            confidence=conf,
            licence=row.get(layout.licence) or None if layout.licence else None,
        )
        if box is not None:
            boxes.setdefault(key, []).append(box)
    return {k: (tuple(boxes.get(k, ())), t.freeze()) for k, t in tallies.items()}


def parse_seqinfo(text: str) -> tuple[int | None, int | None]:
    """``(imWidth, imHeight)`` of a MOT ``seqinfo.ini``."""
    import configparser

    cp = configparser.ConfigParser()
    try:
        cp.read_string(text)
        sec = cp["Sequence"]
        return int(sec["imWidth"]), int(sec["imHeight"])
    except (configparser.Error, KeyError, ValueError):
        return None, None
