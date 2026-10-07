"""MOT-format track reader (WP-U9): not tied to any source.

Reads a MOT Challenge ``gt.txt`` (``frame, id, left, top, width, height, flag, class, visibility``)
or a single-object ``groundtruth.txt`` (``x, y, w, h`` per line, the line is the frame) into
:class:`TrackBox` records, plus the ``seqinfo.ini`` of a sequence. The :class:`MotLayout` names the
columns; boxes are pixel ``xywh`` and come out as normalised ``xyxy`` (:mod:`boxes_common`).

* ``frame_idx`` is **0-based** whatever the source numbers its frames from (MOT starts at 1, so
  ``img1_000001`` is ``frame_idx`` 0); the source's own number is kept as ``native_frame``.
* a row whose ``flag`` is 0 ("do not consider") is kept with ``attrs.ignore``; a row whose
  ``visibility`` is 0 (fully occluded) is kept with ``attrs.fully_occluded``; both are counted.
* boxes are clipped onto the image (counted); a box with no area left, or a non-finite one (an
  absent target is written ``0,0,0,0`` or ``NaN`` in single-object files), is dropped (counted
  ``degenerate``): the line number still advances, so no frame shifts.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass

from .boxes_common import BoxCounts, BoxFormatError, RawBox, Tally, make_box
from .boxes_csv import parse_seqinfo

__all__ = ["MOT_CHALLENGE", "SOT_GROUNDTRUTH", "MotLayout", "TrackBox", "parse_seqinfo", "read_mot"]


@dataclass(frozen=True)
class MotLayout:
    names: tuple[str, ...]  # header of the headerless file; x, y, w, h are always pixel columns
    frame: str | None = "frame"  # None: the physical line number is the frame
    track: str | None = "track_id"  # None: one track for the whole file
    first_frame: int = 1  # number of the first frame (with ``frame=None``: of the first line)
    flag: str | None = "flag"  # "0" marks a row to ignore
    label_id: str | None = "class"
    visibility: str | None = "visibility"
    delimiter: str = ","


MOT_CHALLENGE = MotLayout(
    names=("frame", "track_id", "x", "y", "w", "h", "flag", "class", "visibility")
)
SOT_GROUNDTRUTH = MotLayout(
    names=("x", "y", "w", "h"), frame=None, track=None, first_frame=0, flag=None,
    label_id=None, visibility=None,
)  # fmt: skip


@dataclass(frozen=True)
class TrackBox:
    frame_idx: int  # 0-based
    native_frame: int
    track_id: str
    box: RawBox


def _int(cells: dict[str, str], col: str, no: int) -> int:
    try:
        return int(float(cells[col]))
    except (KeyError, ValueError):
        raise BoxFormatError(f"line {no}: column {col!r} is not a number") from None


def read_mot(
    text: str,
    layout: MotLayout,
    *,
    img_w: int | None,
    img_h: int | None,
    default_track: str = "1",
) -> tuple[tuple[TrackBox, ...], BoxCounts]:
    """Every box of the file in file order, and what was seen (:class:`BoxCounts`)."""
    tally, out = Tally(), []
    for no, raw in enumerate(csv.reader(io.StringIO(text), delimiter=layout.delimiter), 1):
        cells = [c.strip() for c in raw]
        while cells and not cells[-1]:
            cells.pop()
        if not cells:
            continue
        row = dict(zip(layout.names, cells, strict=False))
        try:
            x, y, w, h = (float(row[k]) for k in "xywh")
        except (KeyError, ValueError):
            raise BoxFormatError(f"line {no}: x, y, w, h are not four numbers") from None
        native = _int(row, layout.frame, no) if layout.frame else no - 1 + layout.first_frame
        if native < layout.first_frame:
            raise BoxFormatError(f"line {no}: frame {native} < first frame {layout.first_frame}")
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            tally.bump("read")
            tally.bump("degenerate")
            continue
        attrs: dict[str, object] = {"native_frame": native}
        if layout.visibility and row.get(layout.visibility):
            vis = float(row[layout.visibility])
            attrs["visibility"] = vis
            if vis == 0:
                attrs["fully_occluded"] = True
        box = make_box(
            tally, (x, y, x + w, y + h), img_w=img_w, img_h=img_h, unit=False,
            ignore=bool(layout.flag) and row.get(layout.flag) == "0", attrs=attrs,
            native_id=(row.get(layout.label_id) or None) if layout.label_id else None,
        )  # fmt: skip
        if box is not None:
            track = row.get(layout.track, "") if layout.track else default_track
            out.append(TrackBox(native - layout.first_frame, native, track or default_track, box))
    return tuple(out), tally.freeze()
