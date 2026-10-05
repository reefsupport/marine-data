"""Run the per-row normalisers over a LOCAL staged tree (WP-R2b): the release path's per-row
licence is whatever :mod:`.per_row` resolves, not a separate ingest column.

``staged_row_licences`` returns only licences a row owns (a staged ``license`` column or the
source's per-image data, e.g. FathomNet ``annotationLicense``); the registry fallback string is
never returned, so a row without its own licence stays ``unknown`` (fail closed).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..checksums import parse_checksums
from ..licence_class import per_row_source, source_class
from .base import NormContext
from .stage import spec_fields

_NOT_OWN = ("registry", "INGEST.json")
_LABEL_DIR = Path("labels") / "files"


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def local_context(source_id: str, root: Path, registry_root: str | Path | None = None):
    """``(version, NormContext)`` for the tree at ``root`` (no network, no bucket)."""
    spec = spec_fields(source_id, registry_root)
    sums = root / "CHECKSUMS.sha256"
    ingest = _json(root / "INGEST.json") or {}
    labels: dict[str, Any] = {}
    if source_id == "fathomnet" and (root / _LABEL_DIR).is_dir():
        for path in sorted((root / _LABEL_DIR).glob("*.json")):
            rec = _json(path)
            if isinstance(rec, dict) and rec.get("uuid"):
                labels[str(rec["uuid"])] = rec
    version = spec["version"] or root.name
    ctx = NormContext(
        registry_licence=spec["license"],
        attribution=spec["attribution"],
        citation=spec["citation"],
        homepage=spec["homepage"],
        source_class=source_class(source_id, root=registry_root),
        per_row=per_row_source(source_id, root=registry_root),
        checksums=parse_checksums(sums.read_text()) if sums.is_file() else {},
        ingest=ingest,
        labels=labels,
        version=version,
    )
    return version, ctx


def staged_row_licences(
    source_id: str, root: Path, registry_root: str | Path | None = None
) -> dict[str, str]:
    """``stem -> the row's own licence string`` for a per-row source with a normaliser."""
    from . import NORMALISERS, normalise

    meta = Path(root) / "metadata.parquet"
    if source_id not in NORMALISERS or not meta.is_file():
        return {}
    import pyarrow.parquet as pq

    rows = pq.read_table(meta).to_pylist()
    if source_id == "inat-marine":  # staged rows are keyed by stem = photo id
        rows = [{"photo_id": r.get("photo_id") or r["stem"], **r} for r in rows]
    version, ctx = local_context(source_id, Path(root), registry_root)
    out: dict[str, str] = {}
    for row in normalise(source_id, version, rows, ctx).to_pylist():
        prov = dict(row.get("provenance") or [])
        lic = row.get("license")
        if lic and prov.get("license") not in _NOT_OWN:
            out[str(row["stem"])] = str(lic)
    return out
