"""Access classes: what a source (or one row) may be released as (WP-L1a).

Five classes. ``open`` = commercial use and redistribution are fine; ``restricted-nc`` =
non-commercial only; ``restricted-nd`` = no derivatives; ``internal-only`` = research use /
no redistribution grant; ``unknown`` = nothing established (handled exactly like ``internal-only``).

A *flavour* is a release repo: ``open`` ships class ``open`` only; ``nc`` ships class
``restricted-nc`` only (the NC DELTA, nothing duplicated from ``open``). ``restricted-nd``,
``internal-only`` and ``unknown`` never ship, in any flavour.

* :func:`licence_class_of` - one licence string -> class.
* :func:`resolve_row_class` - source class + row licence string(s) + per-row flag -> class.
* :func:`source_class` / :func:`per_row_source` - the registry's class per source id.
* :func:`flavour_filter` - keep the rows a flavour may ship.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TypeVar

import yaml

OPEN, NC, ND = "open", "restricted-nc", "restricted-nd"
INTERNAL_ONLY, UNKNOWN = "internal-only", "unknown"
ACCESS_CLASSES = (OPEN, NC, ND, INTERNAL_ONLY, UNKNOWN)
FLAVOURS = {"open": OPEN, "nc": NC}
_RANK = {OPEN: 0, NC: 1, ND: 2, INTERNAL_ONLY: 3, UNKNOWN: 3}
T = TypeVar("T")

_ARR = re.compile(r"all[\s_-]*rights[\s_-]*reserved", re.I)
_CC = re.compile(
    r"(?<![A-Za-z0-9])CC[\s_-]?BY(?P<mods>(?:[\s_-]+(?:NC|ND|SA))*)(?![A-Za-z])"
    r"|(?<![A-Za-z0-9])CC[\s_-]?(?:0|zero)(?![A-Za-z0-9])"
    r"|Creative[\s_-]+Commons[\s_-]+Attribution(?P<long>(?:[\s_-]+(?:Non[\s_-]?Commercial|No[\s_-]?Deriv\w*|Share[\s_-]?Alike))*)",
    re.I,
)
_PLAIN = re.compile(
    r"(?<![A-Za-z0-9])(?:PDM|PD|US[\s_-]?GOV[\s_-]?PD|public[\s_-]+domain|Apache(?:[\s_-]?2(?:\.0)?)?|MIT|ODbL)(?![A-Za-z])",
    re.I,
)


def _strictest(classes: Iterable[str]) -> str:
    return max(classes, key=_RANK.__getitem__)


def coerce_class(value: object) -> str:
    """A valid class, else ``unknown`` (never raises: a typo must not release data)."""
    text = str(getattr(value, "value", value) or "").strip().lower()
    return text if text in ACCESS_CLASSES else UNKNOWN


def licence_class_of(licence: str | None) -> str:
    """``CC0 / PDM / PD / CC-BY(-SA) / Apache / MIT / ODbL`` -> open; ``CC-BY-NC(-SA)`` ->
    restricted-nc; any ``-ND`` -> restricted-nd; empty, unparseable, "all rights reserved" ->
    unknown. A string naming several licences gets the strictest class."""
    text = str(licence or "").strip()
    if not text or _ARR.search(text):
        return UNKNOWN
    found: list[str] = []
    for m in _CC.finditer(text):
        mods = (m.group("mods") or m.group("long") or "").upper()
        flat = re.sub(r"[\s_-]+", "", mods)
        found.append(
            ND
            if ("ND" in flat or "NODERIV" in flat)
            else NC
            if ("NC" in flat or "NONCOMMERCIAL" in flat)
            else OPEN
        )
    found.extend(OPEN for _ in _PLAIN.finditer(text))
    return _strictest(found) if found else UNKNOWN


def resolve_row_class(
    source_class_: object, *row_licences: str | None, per_row: bool = False
) -> str:
    """One row's class from its source's class and its own licence string(s).

    ``per_row`` source (every photo carries its own licence): the row class wins outright;
    no licence string, or none parseable, means ``unknown``. Any other source: the row class
    wins only when stricter than the source's; a missing or unparseable string is no
    information and the source class stands."""
    src = coerce_class(source_class_)
    strings = [s for s in row_licences if s and str(s).strip()]
    classes = [licence_class_of(s) for s in strings]
    if per_row:
        return _strictest(classes) if classes else UNKNOWN
    known = [c for c in classes if c != UNKNOWN]
    return _strictest([src, *known]) if known else src


# -- registry lookups ------------------------------------------------------------


def _registry_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root)
    from .registry import _default_root

    return _default_root()


def _scan(base: Path) -> dict[str, dict[str, Any]]:
    """``{id: {"access_class", "licence_per_row"}}``: registry sources, then ingest specs."""
    out: dict[str, dict[str, Any]] = {}
    paths = [
        *sorted((base / "sources").glob("*.yaml")),
        *sorted((base / "ingest-specs").glob("*.yaml")),
    ]
    for path in paths:
        data = yaml.safe_load(path.read_text()) or {}
        entries = data.get("sources", []) if path.parent.name == "sources" else [data]
        for e in entries if isinstance(entries, list) else []:
            if isinstance(e, dict) and isinstance(e.get("id"), str) and e["id"] not in out:
                out[e["id"]] = {
                    "access_class": coerce_class(e.get("access_class")),
                    "licence_per_row": bool(e.get("licence_per_row", False)),
                }
    return out


@functools.lru_cache(maxsize=4)
def _cached(base: str) -> dict[str, dict[str, Any]]:
    return _scan(Path(base))


def source_classes(root: str | Path | None = None) -> Mapping[str, str]:
    return {k: v["access_class"] for k, v in _cached(str(_registry_root(root))).items()}


def source_class(source_id: str, default: str = UNKNOWN, root: str | Path | None = None) -> str:
    """The registry's class for ``source_id``; ``default`` when the id is not in the registry."""
    entry = _cached(str(_registry_root(root))).get(source_id)
    return entry["access_class"] if entry else coerce_class(default)


def per_row_source(source_id: str, root: str | Path | None = None) -> bool:
    entry = _cached(str(_registry_root(root))).get(source_id)
    return bool(entry and entry["licence_per_row"])


# -- the flavour filter ----------------------------------------------------------


def _field(row: object, name: str) -> Any:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name, None)


def row_class(row: object, classes: Mapping[str, str] | None = None) -> str:
    """A row's own class (``licence_class`` field, else ``attrs.licence_class``), else its
    source's registry class, else ``unknown``. A per-row source never falls back to its source
    class (that class is only a bound): no class on the row means ``unknown``."""
    own = _field(row, "licence_class")
    if not own:
        attrs = _field(row, "attrs")
        if isinstance(attrs, str) and attrs:
            try:
                attrs = json.loads(attrs)
            except ValueError:
                attrs = None
        own = attrs.get("licence_class") if isinstance(attrs, Mapping) else None
    if own:
        return coerce_class(own)
    sid = _field(row, "source_id")
    if not isinstance(sid, str) or per_row_source(sid):
        return UNKNOWN
    return (
        coerce_class(classes[sid]) if classes is not None and sid in classes else source_class(sid)
    )


def flavour_filter(
    rows: Iterable[T], flavour: str, classes: Mapping[str, str] | None = None
) -> list[T]:
    """The rows ``flavour`` may ship: ``open`` -> class open only; ``nc`` -> class
    restricted-nc only. Nothing nd, internal-only or unknown ever passes."""
    if flavour not in FLAVOURS:
        raise ValueError(f"unknown flavour {flavour!r}; expected one of {sorted(FLAVOURS)}")
    want = FLAVOURS[flavour]
    return [r for r in rows if row_class(r, classes) == want]
