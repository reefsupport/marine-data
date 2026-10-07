"""Pixel-level normalisation for staging: byte-identical image copies, mask re-encoding.

Two operations only, matched to D1 §3's "images are copied byte-for-byte; never
re-encoded or resized at ingest. [...] Masks are the one unavoidable re-encode."

``pillow`` is an optional dependency (the ``ingest`` extra) and is imported lazily, so
importing this module never requires it — only calling into it does, with an error
naming the extra to install, matching the existing ``integrations/`` pattern.
"""

from __future__ import annotations

import io
from pathlib import Path

from .checksums import copy_digest, write_digest

CLASSES_PER_BYTE = 3
"""RGB bytes per palette entry — a PIL 'P'-mode palette is a flat 768-byte buffer
(256 entries × 3 channels), regardless of how many indices are actually in use."""


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pillow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return Image


def copy_image(src: Path, dest: Path) -> tuple[str, int, int]:
    """Copy ``src`` to ``dest`` byte-for-byte and read back its dimensions.

    Returns ``(sha256, width, height)``. The digest comes from :func:`copy_digest`
    (streamed, one pass); the size comes from a header-only open of the copy that is
    now on disk — ``Image.open`` does not decode pixel data until asked to, so this
    costs a `stat`-sized read, not a decode.
    """
    sha256 = copy_digest(src, dest)
    Image = _require_pillow()
    with Image.open(dest) as im:
        width, height = im.size
    return sha256, width, height


def _palette_bytes(classes: int) -> bytes:
    """A deterministic 768-byte palette, indices ``0..255``.

    Indices ``0..7`` reproduce SUIM's own 3-bit (R,G,B) packing (see
    ``registry/crosswalks/suim-8class.yaml``): bit 2 is red, bit 1 is green, bit 0 is
    blue, each 0 or 255. Indices beyond that repeat the same packing mod 8, which keeps
    the palette total, deterministic and distinct across ``classes <= 8`` without
    needing a second scheme for a hypothetical wider crosswalk.
    """
    entries = bytearray(256 * CLASSES_PER_BYTE)
    for index in range(256):
        r = 255 if (index >> 2) & 1 else 0
        g = 255 if (index >> 1) & 1 else 0
        b = 255 if index & 1 else 0
        entries[index * 3 : index * 3 + 3] = bytes((r, g, b))
    return bytes(entries)


def _rgb_bitpacked_indices(im) -> bytes:
    """Decode an RGB-mode mask into raw per-pixel indices via SUIM's own 3-bit packing.

    I2 override (2026-09-18 brief, upstream SUIM masks are very likely 24-bit RGB BMPs,
    not the ``'L'``-mode the design assumed): ``index = 4*(R>127) + 2*(G>127) +
    1*(B>127)``, confirmed against ``registry/crosswalks/suim-8class.yaml``'s own
    decoder comment (HD=(0,0,1)=1, RO=(1,0,0)=4, FV=(1,1,0)=6, WR=(0,1,1)=3,
    RI=(1,0,1)=5 — every one consistent with R contributing 4, G contributing 2, B
    contributing 1). Thresholded at 127, never exact-matched: upstream has off-pure
    values. Built from three ``'L'``-mode channel images combined with
    :class:`PIL.ImageChops`, not numpy — numpy is not part of the ``ingest`` extra, and
    the per-channel max (``4+2+1=7``) never overflows a single byte.
    """
    from PIL import ImageChops

    red, green, blue = im.split()
    red_bit = red.point(lambda v: 4 if v > 127 else 0)
    green_bit = green.point(lambda v: 2 if v > 127 else 0)
    blue_bit = blue.point(lambda v: 1 if v > 127 else 0)
    combined = ImageChops.add(ImageChops.add(red_bit, green_bit), blue_bit)
    return combined.tobytes()


def _encode_indexed_png(im, dest: Path, *, classes: int, label: str) -> tuple[str, frozenset[int]]:
    """Shared body of :func:`bmp_mask_to_indexed_png` and :func:`mask_bytes_to_indexed_png`
    (D2a): given an already-opened PIL image, validate its mode and pixel range and
    write ``dest`` as an indexed PNG with the deterministic palette. ``label`` is only
    used in error messages, so a path source and a bytes source can share this one copy
    of the index-validation logic rather than duplicating it.

    Accepts single-channel ``'L'``/already-indexed ``'P'`` masks unchanged, and 24-bit
    ``'RGB'`` masks via :func:`_rgb_bitpacked_indices` (I2 override — see its
    docstring). Raises for any other mode, and if any observed pixel index is ``>=
    classes``: an out-of-range index is a crosswalk error, not a warning, and must not
    be staged silently.

    Returns ``(sha256 of the written PNG, the set of pixel indices actually observed)``.
    """
    if im.mode == "RGB":
        raw = _rgb_bitpacked_indices(im)
    elif im.mode in {"L", "P"}:
        # Both 'L' and 'P' store one byte per pixel; ``tobytes()`` on either
        # returns those raw bytes verbatim (for 'P' it is the palette *index*,
        # never the resolved RGB). Building the target image from those bytes —
        # rather than ``Image.convert("P")``, which quantises/dithers by default —
        # is what keeps each pixel's integer value exactly what it was in the
        # source mask.
        raw = im.tobytes()
    else:
        raise ValueError(
            f"{label}: mode {im.mode!r} is not a supported mask mode (need 'L', 'P' or 'RGB')"
        )
    observed = frozenset(raw)
    if observed and max(observed) >= classes:
        raise ValueError(
            f"{label}: observed pixel index {max(observed)} is out of range for "
            f"{classes} declared classes"
        )
    Image = _require_pillow()
    paletted = Image.frombytes("P", im.size, raw)
    paletted.putpalette(_palette_bytes(classes))

    buffer = io.BytesIO()
    paletted.save(buffer, format="PNG", optimize=False, compress_level=6)

    return write_digest(dest, buffer.getvalue()), observed


def bmp_mask_to_indexed_png(src: Path, dest: Path, *, classes: int) -> tuple[str, frozenset[int]]:
    """Re-encode a SUIM ``.bmp`` mask as an indexed (mode ``'P'``) PNG.

    The one unavoidable re-encode in the pipeline (D1 §3) — ``.bmp`` is not a format we
    ship. See :func:`_encode_indexed_png` for the mode handling and validation this
    delegates to.

    Returns ``(sha256 of the written PNG, the set of pixel indices actually observed)``.
    """
    Image = _require_pillow()
    with Image.open(src) as im:
        return _encode_indexed_png(im, dest, classes=classes, label=str(src))


def mask_bytes_to_indexed_png(
    payload: bytes, dest: Path, *, classes: int
) -> tuple[str, frozenset[int]]:
    """Re-encode an already-decoded mask (D2a: an HF parquet mask cell's ``bytes``) as
    an indexed PNG. Bytes-input sibling of :func:`bmp_mask_to_indexed_png` — shares
    :func:`_encode_indexed_png` so the index-validation/palette logic is not duplicated.

    Returns ``(sha256 of the written PNG, the set of pixel indices actually observed)``.
    """
    Image = _require_pillow()
    with Image.open(io.BytesIO(payload)) as im:
        return _encode_indexed_png(im, dest, classes=classes, label="<bytes>")
