"""Source-level constants and the registry date fallback (WP-U14b, U14c).

``default_platform`` / ``default_habitat`` / ``default_instrument`` are optional registry fields
(ingest spec or source entry), filled only where the evidence table marks the value known; they
are validated against the controlled vocabularies (``PLATFORMS``, :data:`HABITATS`,
:data:`INSTRUMENTS`) and used as a fallback with provenance ``registry_default``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..enums import Habitat
from ..sample_schema import PLATFORMS
from .base import parse_date

HABITATS = frozenset({h.value for h in Habitat} | {"brackish_water", "river"})
"""``Habitat`` plus two settings the enum lacks (WP-2b enum untouched): brackish water, river."""

INSTRUMENTS = frozenset({"imaging-sonar", "side-scan-sonar", "holographic-imager", "ifcb"})
"""Controlled vocabulary for the ``camera`` column (the imaging instrument), filled only from a
registry ``default_instrument`` that names evidence (WP-U14c); never inferred per row."""


def registry_entry(source_id: str, root: str | Path | None = None) -> dict[str, Any]:
    """Raw merge of the ingest spec (wins) and the ``registry/sources`` entry for ``source_id``."""
    import yaml

    from ..registry import _default_root

    base = Path(root) if root else _default_root()
    out: dict[str, Any] = {}
    spec = base / "ingest-specs" / f"{source_id}.yaml"
    if spec.is_file():
        out.update(yaml.safe_load(spec.read_text()) or {})
    for path in sorted((base / "sources").glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        for e in data.get("sources", []) if isinstance(data, dict) else []:
            if isinstance(e, dict) and e.get("id") == source_id:
                return {**e, **out}
    return out


def validated_defaults(entry: dict[str, Any]) -> dict[str, str]:
    """``{"platform": ..., "habitat": ..., "instrument": ...}`` of the entry's non-empty defaults;
    ``ValueError`` for a value outside the controlled vocabulary."""
    out: dict[str, str] = {}
    for name, vocab in (
        ("platform", PLATFORMS),
        ("habitat", HABITATS),
        ("instrument", INSTRUMENTS),
    ):
        value = str(entry.get(f"default_{name}") or "").strip()
        if value:
            if value not in vocab:
                raise ValueError(f"default_{name} {value!r} not in {sorted(vocab)}")
            out[name] = value
    return out


def registry_fetch_date(entry: dict[str, Any]) -> tuple[Any, str]:
    """``(date, "registry:<field>")`` from ``retrieved`` / ``fetched``; ``(None, "")`` if absent."""
    for key in ("retrieved", "fetched"):
        parsed = parse_date(entry.get(key)) if entry.get(key) else None
        if parsed:
            return parsed, f"registry:{key}"
    return None, ""
