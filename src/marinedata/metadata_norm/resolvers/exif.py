"""EXIF date (and GPS, when present) from the first 64 KB of a JPEG; no full decode."""

from __future__ import annotations

import datetime as dt
from typing import Any

JPEG_SOI = b"\xff\xd8"
EXIF_IFD, GPS_IFD = 0x8769, 0x8825
DATETIME_TAGS = ((EXIF_IFD, 36867), (EXIF_IFD, 36868), (None, 306))  # Original, Digitized, DateTime
EARLIEST = dt.date(2000, 1, 1)


def exif_segment(head: bytes) -> bytes | None:
    """The APP1 ``Exif`` payload of a (possibly truncated) JPEG, or ``None``."""
    if head[:2] != JPEG_SOI:
        return None
    i = 2
    while i + 4 <= len(head):
        if head[i] != 0xFF:
            return None
        marker = head[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker == 0x01 or 0xD0 <= marker <= 0xD8:
            i += 2
            continue
        if marker in (0xD9, 0xDA):
            return None
        size = int.from_bytes(head[i + 2 : i + 4], "big")
        body = head[i + 4 : i + 2 + size]
        if marker == 0xE1 and body.startswith(b"Exif\x00\x00"):
            return body if len(body) == size - 2 else None
        i += 2 + size
    return None


def _parse_date(text: Any) -> dt.date | None:
    try:
        d = dt.datetime.strptime(str(text).strip()[:19], "%Y:%m:%d %H:%M:%S").date()
    except ValueError:
        return None
    return d if EARLIEST <= d <= dt.date.today() else None


def _dms(values: Any, ref: Any) -> float | None:
    try:
        deg, minutes, seconds = (float(v) for v in values)
    except (TypeError, ValueError):
        return None
    out = deg + minutes / 60 + seconds / 3600
    return -out if str(ref).strip().upper() in ("S", "W") else out


def exif_fields(head: bytes) -> tuple[dt.date | None, tuple[float, float] | None, str | None]:
    """``(date, (lat, lon) | None, tag name)`` read from the first bytes of a JPEG."""
    from PIL import Image

    payload = exif_segment(head)
    if payload is None:
        return None, None, None
    ex = Image.Exif()
    try:
        ex.load(payload)
        found, tag_name = None, None
        for ifd, tag in DATETIME_TAGS:
            raw = (ex.get_ifd(ifd) if ifd else ex).get(tag)
            found = _parse_date(raw) if raw else None
            if found:
                tag_name = {36867: "DateTimeOriginal", 36868: "DateTimeDigitized", 306: "DateTime"}[
                    tag
                ]
                break
        gps = ex.get_ifd(GPS_IFD)
    except (ValueError, OSError, SyntaxError, KeyError, TypeError):
        return None, None, None
    lat, lon = _dms(gps.get(2), gps.get(1)), _dms(gps.get(4), gps.get(3))
    point = (lat, lon) if lat is not None and lon is not None and (lat or lon) else None
    return found, point, tag_name
