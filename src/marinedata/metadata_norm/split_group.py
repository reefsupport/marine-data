"""Deterministic ``split_group`` (WP-U14b): near-identical frames never straddle splits.

Order (first hit wins; provenance names the rule):
1. ``registry-rule``: the source's registry ``SplitGroupRule`` has a pattern (the same call the
   release builder uses, ``Source.split_group_for``);
2. ``row:sequence``: a sequence / video / track / observation id on the row;
3. ``row:site``: a site / deployment / dive id on the row;
4. ``upstream_split+folder``: the upstream split plus the image's own upstream folder (skipped
   when that folder is only the split directory: one group per split is no grouping);
5. ``spatiotemporal`` (WP-U14c): a row with lat/lon and a capture date is grouped as
   ``<source_id>:<lat>,<lon>:<date>`` (lat/lon rounded to 0.01 deg, UTC date): frames of one
   dive / station / location-day share a group (fathomnet: a dive is a location-day);
6. ``image_sha256``;
7. ``registry-fallback``: ``<source_id>/<partition>``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

SEQUENCE_KEYS = ("sequence_id", "video_id", "track_id", "sequence", "video", "observation_uuid")
SITE_KEYS = ("site_id", "deployment_id", "dive_id", "site", "dive", "deployment")


def _text(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _first(row: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, str] | None:
    for k in keys:
        if _text(row.get(k)):
            return k, _text(row[k])
    return None


def _cell(x: Any) -> str | None:
    """``x`` rounded to 0.01 deg as text; ``None`` when it is not a finite number."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    out = f"{f:.2f}"
    return "0.00" if out == "-0.00" else out


def spatiotemporal_group(source_id: str, values: Mapping[str, Any]) -> str | None:
    """``<source_id>:<lat,lon at 0.01 deg>:<UTC date>`` when lat, lon and a capture time are set."""
    when = values.get("capture_datetime")
    lat, lon = _cell(values.get("lat")), _cell(values.get("lon"))
    if when is None or lat is None or lon is None or not hasattr(when, "date"):
        return None
    return f"{source_id}:{lat},{lon}:{when.date().isoformat()}"


def derive_split_group(
    source_id: str, row: Mapping[str, Any], values: Mapping[str, Any], rule: Any = None
) -> tuple[str, str]:
    """``(group, rule name)``. ``row`` = the staged row, ``values`` = the normalised fields."""
    stem = _text(values.get("stem"))
    partition = _text(row.get("partition")) or "default"
    path = _text(values.get("upstream_path"))
    if rule is not None and getattr(rule, "pattern", None):
        try:
            return rule.resolve(
                source_id=source_id, stem=stem, upstream_path=path, partition=partition
            ), "registry-rule"
        except ValueError:
            pass  # a stem the declared pattern misses: fall through, never raise per row
    for keys, name in ((SEQUENCE_KEYS, "row:sequence"), (SITE_KEYS, "row:site")):
        hit = _first(row, keys)
        if hit:
            return f"{source_id}/{hit[0]}:{hit[1]}", name
    split = _text(values.get("upstream_split"))
    folder = PurePosixPath(path).parent.name if path else ""
    if split and folder and folder.lower() != split.lower():
        return f"{source_id}/{split}/{folder}", "upstream_split+folder"
    spatiotemporal = spatiotemporal_group(source_id, values)
    if spatiotemporal:
        return spatiotemporal, "spatiotemporal:0.01deg+utc_date"
    sha = _text(values.get("image_sha256"))
    if sha:
        return f"{source_id}/sha:{sha}", "image_sha256"
    return f"{source_id}/{partition}", "registry-fallback"
