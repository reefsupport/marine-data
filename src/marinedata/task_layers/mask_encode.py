"""MV-1: lossless normalisation of every staged mask encoding into the two HF forms.

* semantic: a single-channel ``uint8`` index PNG, pixel value = class id, **255 = ignore /
  unlabelled** (a source's own ignore value, e.g. Coralscapes' 0, is remapped to 255);
* instance: a single-channel ``uint16`` PNG, pixel value = instance id (1..N), **0 = background**.

Source encodings handled (``masks_table.MaskSource.encoding`` plus the instance sources):
``png-indexed`` (P / L index PNG), ``png-rgb-red`` (CoralSeg: the red channel is the class),
``rgb-palette`` (SUIM: the authors' RGB table, ``4R + 2G + B`` of the thresholded channels),
per-class binary PNGs (:func:`compose_parts`), COCO polygons and COCO RLE
(:func:`instance_png`). numpy / Pillow are imported lazily, like the rest of the export path.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence

IGNORE = 255
"""The ignore / unlabelled pixel value of every semantic HF mask."""
ENCODING_OVERRIDE = {"suim": "rgb-palette"}
"""Sources whose staged PNG encoding differs from the one their ``MaskSource`` declares."""


class MaskEncodingError(ValueError):
    """A staged mask that cannot be normalised without changing or inventing a label."""


def encoding_for(source_id: str, declared: str | None) -> str:
    return ENCODING_OVERRIDE.get(source_id, declared or "png-indexed")


def _array(data: bytes):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im)


def decode_index(data: bytes, encoding: str):
    """The class-index array (2-D) of one staged semantic PNG, before the ignore remap."""
    import numpy as np

    arr = _array(data)
    if arr.ndim == 2:
        return arr
    if encoding == "png-rgb-red":
        return arr[..., 0]
    colour = arr[..., :3]
    if bool((colour == colour[..., :1]).all()):  # grey replicated over the channels: an index
        return colour[..., 0]
    if encoding == "rgb-palette":
        bits = (colour >= 128).astype(np.uint8)
        return 4 * bits[..., 0] + 2 * bits[..., 1] + bits[..., 2]
    raise MaskEncodingError(f"colour mask under encoding {encoding!r}: no class index")


def remap_ignore(arr, ignore_value: int | None):
    """``uint8`` copy of ``arr`` with ``ignore_value`` (and a native 255) turned into
    :data:`IGNORE`. Raises if the values do not fit ``uint8`` or 255 is also a real class."""
    import numpy as np

    if arr.size and int(arr.max()) > IGNORE:
        raise MaskEncodingError(f"class index {int(arr.max())} does not fit a uint8 mask")
    if ignore_value not in (None, IGNORE) and bool((arr == IGNORE).any()):
        raise MaskEncodingError("pixel value 255 present but 255 is the reserved ignore value")
    out = arr.astype(np.uint8, copy=True)
    if ignore_value is not None:
        out[arr == ignore_value] = IGNORE
    return out


def png_bytes(arr) -> bytes:
    """Lossless PNG of a 2-D ``uint8`` (mode L) or ``uint16`` (mode I;16) array."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _fit(arr, size: tuple[int, int] | None):
    """Nearest-neighbour resize of a label array to ``size = (width, height)`` when it differs
    (labels are never interpolated); ``None`` keeps the array as is."""
    if size is None or (arr.shape[1], arr.shape[0]) == tuple(size):
        return arr
    import numpy as np
    from PIL import Image

    return np.asarray(Image.fromarray(arr).resize(tuple(size), Image.NEAREST))


def semantic_png(
    data: bytes, *, encoding: str, ignore_value: int | None, size: tuple[int, int] | None = None
) -> bytes:
    """One staged semantic mask -> the HF index PNG (255 = ignore)."""
    arr = remap_ignore(decode_index(data, encoding), ignore_value)
    return png_bytes(_fit(arr, size))


def compose_parts(
    parts: Sequence[tuple[bytes, int]], size: tuple[int, int], *, ignore_value: int | None = None
) -> bytes:
    """Per-class binary PNGs -> one index PNG: later parts paint over earlier ones, the rest is
    :data:`IGNORE`. ``parts`` is ``[(png bytes, class id)]``; any non-zero pixel is foreground."""
    import numpy as np
    from PIL import Image

    out = np.full((size[1], size[0]), IGNORE, dtype=np.uint8)
    for data, class_id in parts:
        if not 0 <= class_id < IGNORE or class_id == ignore_value:
            raise MaskEncodingError(f"class id {class_id} cannot be an index-mask value")
        with Image.open(io.BytesIO(data)) as im:
            fg = np.asarray(im.convert("L")) > 0
        out[_fit(fg.astype(np.uint8), size).astype(bool)] = class_id
    return png_bytes(out)


# ---- instances: COCO polygon / RLE / per-instance PNG -> uint16 id map -------------------------


def rle_counts(counts: object) -> list[int]:
    """COCO run lengths: a list is used as is, a (compressed) string is decoded the way
    ``pycocotools`` does (5-bit groups, delta against the run two back)."""
    if isinstance(counts, (list, tuple)):
        return [int(c) for c in counts]
    text = counts.decode("ascii") if isinstance(counts, bytes) else str(counts)
    out: list[int] = []
    p = 0
    while p < len(text):
        x, k, more = 0, 0, True
        while more:
            c = ord(text[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and c & 0x10:
                x |= -1 << (5 * k)
        if len(out) > 2:
            x += out[-2]
        out.append(x)
    return out


def decode_rle(rle: Mapping) -> object:
    """Boolean ``(h, w)`` array of a COCO RLE (column-major runs, 0 first)."""
    import numpy as np

    h, w = (int(v) for v in rle["size"])
    runs = rle_counts(rle["counts"])
    flat = np.repeat(np.arange(len(runs)) % 2 == 1, runs)
    if flat.size != h * w:
        raise MaskEncodingError(f"RLE covers {flat.size} pixels, size says {h * w}")
    return flat.reshape((h, w), order="F")


def polygon_mask(rings: Sequence[Sequence[float]], size: tuple[int, int]):
    """Boolean ``(h, w)`` union of COCO polygon rings (flat ``[x0, y0, x1, y1, ...]``)."""
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.new("L", size, 0)
    draw = ImageDraw.Draw(canvas)
    for ring in rings:
        pts = list(zip(ring[0::2], ring[1::2], strict=False))
        if len(pts) >= 3:
            draw.polygon(pts, fill=1, outline=1)
    return np.asarray(canvas) > 0


def instance_mask(row: Mapping, size: tuple[int, int], fetch: Callable[[str], bytes]):
    """Boolean mask of one instance row (``polygon`` / ``rle`` JSON, else the ``mask_ref`` PNG)."""
    import numpy as np
    from PIL import Image

    if row.get("polygon"):
        return polygon_mask(json.loads(row["polygon"]), size)
    if row.get("rle"):
        return decode_rle(json.loads(row["rle"]))
    if row.get("mask_ref"):
        with Image.open(io.BytesIO(fetch(row["mask_ref"]))) as im:
            return np.asarray(im.convert("L")) > 0
    raise MaskEncodingError(f"instance {row.get('ann_id')}: no polygon, rle or mask_ref")


def instance_png(
    rows: Sequence[Mapping],
    ids: Sequence[int],
    *,
    canvas: tuple[int, int],
    size: tuple[int, int],
    fetch: Callable[[str], bytes],
) -> bytes:
    """The ``uint16`` instance-id PNG of one image: ``ids[i]`` is painted for ``rows[i]``.
    Larger instances are painted first, so a small instance inside a large one keeps its id; an
    overlap is otherwise resolved by paint order (a pixel holds one id, the HF mask is flat)."""
    import numpy as np

    if len(rows) > 65535:
        raise MaskEncodingError(f"{len(rows)} instances do not fit uint16")
    out = np.zeros((canvas[1], canvas[0]), dtype=np.uint16)
    for i in sorted(range(len(rows)), key=lambda j: -_area(rows[j])):
        fg = _fit(instance_mask(rows[i], canvas, fetch).astype(np.uint8), canvas)
        out[fg.astype(bool)] = ids[i]
    return png_bytes(_fit(out, size))


def _area(row: Mapping) -> float:
    attrs = (
        json.loads(row["attrs"]) if isinstance(row.get("attrs"), str) else (row.get("attrs") or {})
    )
    return float(attrs.get("area") or 0)
