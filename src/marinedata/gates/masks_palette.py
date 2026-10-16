"""G5 palette masks: a v1.1.0 ``semseg`` row embeds a mode-``P`` PNG whose values are class ids
and whose palette is the row's declared ``palette`` (JSON ``[{id, rgb}]``, ignore index 255 listed).

Per-row problems (each is a ``failed_<kind>`` counter of the ``masks`` gate):

* ``palette_missing``   mask bytes without a parseable ``palette`` column;
* ``mask_mode``         the embedded mask is not mode ``P`` (palette);
* ``undeclared_colour`` a pixel value the declared palette does not list (nothing is implied: 255
  must be declared too);
* ``palette_mismatch``  the PNG's own palette gives a used value another colour than declared;
* ``mask_sha256``       ``mask_bytes_sha256`` is not the sha256 of the embedded bytes;
* ``mask_unresolved``   neither mask bytes nor a ``mask_config`` (the v1.0 config that holds it).

A row whose mask lives in a v1.0 config (``mask_config`` set, no bytes) is not decoded: v1.0 bytes
are frozen and were gated when they shipped."""

from __future__ import annotations

import hashlib
import io
import json

IGNORE = 255


def declared_palette(cell: object) -> dict[int, tuple[int, int, int]] | None:
    """``{id: (r, g, b)}`` of a ``palette`` cell, or ``None`` when absent or malformed."""
    try:
        entries = json.loads(cell) if isinstance(cell, str) else cell
        out = {int(e["id"]): tuple(int(c) for c in e["rgb"]) for e in entries}
    except (TypeError, ValueError, KeyError):
        return None
    return out if out and all(len(rgb) == 3 for rgb in out.values()) else None


def check_palette_row(row: dict) -> tuple[list[tuple[str, str]], bool]:
    """``([(failure kind, message)], mask_is_empty)`` for one ``semseg`` row."""
    import numpy as np
    from PIL import Image

    data = (row.get("mask") or {}).get("bytes")
    if not data:
        if row.get("mask_config"):
            return [], False
        return [("mask_unresolved", "no mask bytes and no mask_config")], False
    palette = declared_palette(row.get("palette"))
    if palette is None:
        return [("palette_missing", "the row embeds a mask but declares no palette")], False
    problems: list[tuple[str, str]] = []
    if row.get("mask_bytes_sha256") != hashlib.sha256(data).hexdigest():
        problems.append(("mask_sha256", "mask_bytes_sha256 is not the sha256 of the mask bytes"))
    try:
        mask = Image.open(io.BytesIO(data))
        mask.load()
    except (OSError, ValueError) as exc:
        return [*problems, ("undecodable", str(exc)[:120])], False
    if mask.mode != "P":
        return [*problems, ("mask_mode", f"semseg mask is {mask.mode}, want P (palette)")], False
    values = {int(v) for v in np.unique(np.asarray(mask))}
    undeclared = sorted(values - palette.keys())
    if undeclared:
        problems.append(
            (
                "undeclared_colour",
                f"mask values {undeclared[:6]} are not in the declared palette ids "
                f"{sorted(palette)[:6]}",
            )
        )
    flat = mask.getpalette() or []
    wrong = [
        v for v in sorted(values & palette.keys()) if tuple(flat[3 * v : 3 * v + 3]) != palette[v]
    ]
    if wrong:
        problems.append(
            ("palette_mismatch", f"PNG palette differs from the declared rgb at {wrong[:6]}")
        )
    return problems, not (values & (palette.keys() - {IGNORE}))
