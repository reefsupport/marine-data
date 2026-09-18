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


def bmp_mask_to_indexed_png(src: Path, dest: Path, *, classes: int) -> tuple[str, frozenset[int]]:
    """Re-encode a single-channel SUIM ``.bmp`` mask as an indexed (mode ``'P'``) PNG.

    The one unavoidable re-encode in the pipeline (D1 §3) — ``.bmp`` is not a format we
    ship. Raises if the source mask is not single-channel (``'L'``) or already-indexed
    (``'P'``), or if any observed pixel index is ``>= classes``: an out-of-range index
    is a crosswalk error, not a warning, and must not be staged silently.

    Returns ``(sha256 of the written PNG, the set of pixel indices actually observed)``.
    """
    Image = _require_pillow()
    with Image.open(src) as im:
        if im.mode not in {"L", "P"}:
            raise ValueError(f"{src}: mode {im.mode!r} is not a single-channel index mask")
        # Both 'L' and 'P' store one byte per pixel; ``tobytes()`` on either returns
        # those raw bytes verbatim (for 'P' it is the palette *index*, never the
        # resolved RGB). Building the target image from those bytes — rather than
        # ``Image.convert("P")``, which quantises/dithers by default — is what keeps
        # each pixel's integer value exactly what it was in the source mask.
        raw = im.tobytes()
        observed = frozenset(raw)
        if observed and max(observed) >= classes:
            raise ValueError(
                f"{src}: observed pixel index {max(observed)} is out of range for "
                f"{classes} declared classes"
            )
        paletted = Image.frombytes("P", im.size, raw)
        paletted.putpalette(_palette_bytes(classes))

        buffer = io.BytesIO()
        paletted.save(buffer, format="PNG", optimize=False, compress_level=6)

    return write_digest(dest, buffer.getvalue()), observed
