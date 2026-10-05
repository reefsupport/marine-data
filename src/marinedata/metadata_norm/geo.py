"""Position / depth / time / platform extractors (WP-U14b). Pure: values in, override dict out.

Every extractor returns ``{field: value, ..., "_prov": {field: origin}}`` using only fields it
could read; an unreadable or out-of-range value is dropped (null), never guessed.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from collections.abc import Mapping
from typing import Any

from ..sample_schema import PLATFORMS

_PLATFORM_RULES: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(p, re.I), v)
    for p, v in (
        (r"\b(rov|remotely operated|rov-mounted)\b", "rov"),
        (r"\b(auv|autonomous underwater)\b", "auv"),
        (r"\b(scuba|diver|diving)\b", "diver"),
        (r"\bsnorkel", "snorkel"),
        (r"\b(towed|tow[- ]?(?:cam|body|sled)|sled)\b", "towed"),
        (r"\b(drop[- ]?cam|camera drop|drop[- ]?frame)", "drop-camera"),
        (r"\b(bruv|baited)", "bruv"),
        (r"\b(lander|stationary|time[- ]?lapse|fixed camera|observatory)\b", "lander"),
        (r"\b(uav|drone)\b", "uav"),
        (r"\b(vessel|ship|boat)\b", "vessel"),
        (r"\b(lab|laboratory|aquarium)\b", "lab"),
    )
)
INAT_GENERALISED_PRECISION_M = 22000.0
"""iNat obscures a coordinate within a 0.2 degree cell (~22 km): the stated precision."""


def to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def valid_position(lat: Any, lon: Any) -> tuple[float, float] | None:
    """``(lat, lon)`` when both parse and lie in WGS84 range; ``(0, 0)`` (null island) is rejected."""
    la, lo = to_float(lat), to_float(lon)
    if la is None or lo is None or not (-90 <= la <= 90 and -180 <= lo <= 180):
        return None
    return None if la == 0.0 and lo == 0.0 else (la, lo)


def utc_datetime(value: Any) -> dt.datetime | None:
    """An ISO string / date / datetime as aware UTC; a naive *datetime string* is rejected
    (no local-time guessing), a bare date becomes 00:00Z (the caller says so in provenance)."""
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.UTC) if value.tzinfo else None
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day, tzinfo=dt.UTC)
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return dt.datetime.fromisoformat(text).replace(tzinfo=dt.UTC)
        parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(dt.UTC) if parsed.tzinfo else None


def platform_from(*texts: Any) -> str | None:
    """First controlled-vocabulary platform any ``text`` names; ``None`` when none does."""
    for text in texts:
        for pattern, platform in _PLATFORM_RULES:
            if text and pattern.search(str(text)):
                assert platform in PLATFORMS
                return platform
    return None


def _pos(out: dict[str, Any], prov: dict[str, str], lat, lon, origin: str) -> None:
    pos = valid_position(lat, lon)
    if pos:
        out.update(lat=pos[0], lon=pos[1])
        prov.update(lat=origin, lon=origin)


def fathomnet_fields(rec: Mapping[str, Any]) -> dict[str, Any]:
    """``labels/files/<uuid>.json``: latitude, longitude, depthMeters, timestamp,
    imagingType, tags[key=platform]."""
    out: dict[str, Any] = {}
    prov: dict[str, str] = {}
    _pos(out, prov, rec.get("latitude"), rec.get("longitude"), "fathomnet-json:latitude,longitude")
    depth = to_float(rec.get("depthMeters"))
    if depth is not None and depth >= 0:
        out.update(depth_m=depth, depth_source="sensor")
        prov.update(depth_m="fathomnet-json:depthMeters", depth_source="fathomnet-json:depthMeters")
    when = utc_datetime(rec.get("timestamp"))
    if when:
        out["capture_datetime"] = when
        prov["capture_datetime"] = "fathomnet-json:timestamp"
    tag = next(
        (t.get("value") for t in rec.get("tags") or [] if isinstance(t, dict) and t.get("key") == "platform"),
        None,
    )  # fmt: skip
    platform = platform_from(rec.get("imagingType")) or platform_from(tag)
    if platform:
        out["platform"] = platform
        prov["platform"] = (
            "fathomnet-json:imagingType" if platform_from(rec.get("imagingType")) else "fathomnet-json:tags.platform"
        )  # fmt: skip
    return {**out, "_prov": prov}


_OBSCURED_FLAGS = ("geoprivacy", "coordinates_obscured", "private_latitude")


def inat_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Manifest ``latitude``, ``longitude``, ``observed_on`` honouring iNat geoprivacy.

    iNat publishes obscured coordinates as a random point inside a 0.2 degree cell and the
    manifest carries no flag column: unless a flag says the point is exact, the position is
    generalised to 0.1 degrees (``location_generalized``, ~22 km precision), never kept exact.
    Explicit ``obscured``/``private`` coordinates are dropped.
    """
    out: dict[str, Any] = {}
    prov: dict[str, str] = {}
    when = utc_datetime(row.get("observed_on"))
    if when:
        out["capture_datetime"] = when
        prov["capture_datetime"] = "inat-manifest:observed_on (date only, 00:00Z)"
    geoprivacy = str(row.get("geoprivacy") or "").strip().lower()
    obscured = row.get("coordinates_obscured")
    flagged = any(k in row for k in _OBSCURED_FLAGS)
    pos = valid_position(row.get("latitude"), row.get("longitude"))
    if (
        pos is None
        or geoprivacy in {"obscured", "private"}
        or obscured in (True, 1, "true", "True")
    ):
        return {**out, "_prov": prov}
    acc = to_float(row.get("positional_accuracy"))
    if flagged:  # an explicit "open" flag: the point is exact
        out.update(lat=pos[0], lon=pos[1], location_generalized=False)
        if acc is not None and acc >= 0:
            out["gps_precision_m"] = acc
            prov["gps_precision_m"] = "inat-manifest:positional_accuracy"
        origin = "inat-manifest:latitude,longitude (geoprivacy open)"
    else:
        out.update(
            lat=round(pos[0], 1),
            lon=round(pos[1], 1),
            location_generalized=True,
            gps_precision_m=INAT_GENERALISED_PRECISION_M,
        )
        prov["gps_precision_m"] = "inat-manifest:no geoprivacy column, 0.2 degree cell"
        origin = "inat-manifest:latitude,longitude (no geoprivacy flag: rounded to 0.1)"
    prov.update(lat=origin, lon=origin, location_generalized=origin)
    return {**out, "_prov": prov}


def mermaid_fields(image_id: str, events: Mapping[str, Any]) -> dict[str, Any]:
    """Join one image onto a cached MERMAID sample-event table (``events`` =
    ``{"images": {image_id: sample_event_id}, "events": {sample_event_id: record}}``).
    Only ``latitude``/``longitude`` (site position), ``sample_date`` and ``depth`` are read."""
    event_id = (events.get("images") or {}).get(image_id)
    rec = (events.get("events") or {}).get(event_id) if event_id else None
    if not isinstance(rec, Mapping):
        return {}
    out: dict[str, Any] = {}
    prov: dict[str, str] = {}
    origin = "mermaid-sample-event:latitude,longitude (site)"
    _pos(
        out, prov, rec.get("latitude", rec.get("lat")), rec.get("longitude", rec.get("lon")), origin
    )
    when = utc_datetime(rec.get("sample_date"))
    if when:
        out["capture_datetime"] = when
        prov["capture_datetime"] = "mermaid-sample-event:sample_date (date only, 00:00Z)"
    depth = to_float(rec.get("depth"))
    if depth is not None and depth >= 0:
        out.update(depth_m=depth, depth_source="site-nominal")
        prov.update(depth_m="mermaid-sample-event:depth", depth_source="mermaid-sample-event:depth")
    return {**out, "_prov": prov}
