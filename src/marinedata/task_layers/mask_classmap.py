"""MV-1: the HF ``class_map`` of a semantic mask row (pixel value -> label -> taxon -> coarse).

Split out of :mod:`marinedata.task_layers.configs` (which builds the ``semseg`` / ``instances``
rows these helpers decorate) so that module stays under the file-size ceiling.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

BENTHIC_COARSE_TASK = "benthic-coarse"


def json_or_none(value: object) -> str | None:
    return None if value is None else json.dumps(value, sort_keys=True)


def coarse_of(registry, task_id: str = BENTHIC_COARSE_TASK) -> Callable[[str | None], str | None]:
    """``taxon_node -> benthic-coarse class | None``. Never raises: a node coarser than the
    benthic-coarse vocabulary (or a registry without that task) has no coarse class."""
    try:
        projector = registry.projector_for(task_id)
    except Exception:
        return lambda node: None

    def coarse(node: str | None) -> str | None:
        if not node:
            return None
        try:
            return projector.project(node).target_class
        except ValueError:  # coarser=error: no honest coarse class
            return None

    return coarse


def hf_class_map(record: Mapping, coarse: Callable[[str | None], str | None]) -> str | None:
    """The HF ``class_map`` of one semantic mask row, as a JSON list of
    ``{id, label_native, taxon_node, coarse}`` (pixel value = ``id``); ``None`` for legacy rows
    that never carried the pixel -> label map. An unmapped class has no taxon and no coarse."""
    raw = record.get("class_map")
    if not raw:
        return None
    resolution = record.get("class_resolution") or {}
    out = []
    for pixel, label in sorted(json.loads(raw).items(), key=lambda kv: int(kv[0])):
        res = resolution.get(label) or {}
        taxon = res.get("taxon_node_id") if res.get("match_type") != "unmapped" else None
        out.append(
            {"id": int(pixel), "label_native": label, "taxon_node": taxon, "coarse": coarse(taxon)}
        )
    return json.dumps(out)
