"""Per-row licence and attribution columns (PA-P2.2, WP-L1).

Every resolved row carries ``license license_url attribution rights_holder license_verified_by
license_verified_on provenance_url upstream_id``. The default is the registry entry. A source whose
licence is per item declares ``per_row_license:`` in its entry, a mapping from the output column to
the field of the staged ``metadata.parquet`` row that holds it::

    per_row_license:
      license: license
      license_url: license_url
      attribution: attribution
      rights_holder: photographer

An override REPLACES the default even when the staged field is empty: a per-row source whose row
states no licence gets ``license=null`` and never ships. Nothing is guessed; a licence string is
only re-spelled the way SPDX writes it (``CC BY-NC 4.0`` -> ``CC-BY-NC-4.0``).
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LICENCE_COLUMNS: tuple[str, ...] = (
    "license",
    "license_url",
    "attribution",
    "rights_holder",
    "license_verified_by",
    "license_verified_on",
    "provenance_url",
    "upstream_id",
)
PER_ROW_KEYS = frozenset(
    {"license", "license_url", "attribution", "rights_holder", "provenance_url", "upstream_id"}
)
_CC = re.compile(
    r"^\s*(?:cc|creative\s+commons)[\s_-]+(by(?:[\s_-]+(?:nc|sa|nd))*)(?:[\s_-]+(\d(?:\.\d)?))?\s*$",
    re.I,
)


@dataclass(frozen=True)
class LicenceDefaults:
    """The registry-entry values every row of one source starts from."""

    license: str | None = None
    license_url: str | None = None
    attribution: str | None = None
    rights_holder: str | None = None
    license_verified_by: str | None = None
    license_verified_on: dt.date | None = None
    provenance_url: str | None = None
    per_row: Mapping[str, str] = field(default_factory=dict)


def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def normalise_license(value: Any) -> str | None:
    """``CC BY-NC 4.0`` -> ``CC-BY-NC-4.0``; any other non-empty string is only stripped."""
    text = _text(value)
    if text is None:
        return None
    m = _CC.match(text)
    if not m:
        return text
    mods = re.sub(r"[\s_]+", "-", m.group(1).upper())
    return f"CC-{mods}" + (f"-{m.group(2)}" if m.group(2) else "")


def licence_defaults_for(src: Any, raw: Mapping[str, Any] | None = None) -> LicenceDefaults:
    """Defaults from one registry ``Source`` (``raw`` = its merged ingest entry, whose
    ``attribution`` wins over the one derived from the citation). ``ValueError`` for a
    ``per_row_license`` key that is not overridable."""
    mapping = dict(src.per_row_license or {})
    bad = sorted(set(mapping) - PER_ROW_KEYS)
    if bad:
        raise ValueError(f"per_row_license keys {bad} not in {sorted(PER_ROW_KEYS)}")
    from ...metadata_release import attribution_for

    return LicenceDefaults(
        license=normalise_license(src.licence.id),
        license_url=_text(src.licence.url),
        attribution=_text((raw or {}).get("attribution")) or _text(attribution_for(src)),
        rights_holder=_text(src.rights_holder),
        license_verified_by=_text(src.verification.verified_by),
        license_verified_on=src.verification.verified_on,
        provenance_url=_text(src.homepage) or _text(src.access.uri),
        per_row=mapping,
    )


def licence_defaults(source_id: str, registry_root: str | Path | None = None) -> LicenceDefaults:
    from ...registry import Registry, _default_root
    from ..defaults import registry_entry

    base = Path(registry_root) if registry_root else _default_root()
    return licence_defaults_for(
        Registry.load(base).source(source_id), registry_entry(source_id, base)
    )


def licence_columns(
    staged_row: Mapping[str, Any], defaults: LicenceDefaults | None
) -> dict[str, Any]:
    """The eight licence columns of one image: registry defaults, the staged row's own
    ``upstream_id``, then each declared per-row override (replacing the default, even with null)."""
    if defaults is None:
        return {}
    out: dict[str, Any] = {c: getattr(defaults, c) for c in LICENCE_COLUMNS if c != "upstream_id"}
    out["upstream_id"] = _text(staged_row.get("upstream_id"))
    for column, source_field in defaults.per_row.items():
        value = _text(staged_row.get(source_field))
        out[column] = normalise_license(value) if column == "license" else value
    return out
