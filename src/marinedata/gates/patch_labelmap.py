"""G9 labelmap of a patch (``release gate --patch-of``): the ``condition-masks`` class resolution.

The exported ``label-map`` is frozen in the prior release and the v1.1.0 crosswalk has no rows for
the patch's source, so the stock gate (which needs a ``label-map`` config) cannot run on a patch.
Each patch row declares its classes twice: ``class_map`` (pixel value -> class name) and
``attrs.class_resolution`` (class name -> ``pixel`` and the taxonomy node ids). Failures:

* ``class_resolution``  ``attrs`` carries no ``class_resolution``, or it is not valid JSON;
* ``class_map``         a class_map entry has no resolution, or its ``pixel`` differs from the key;
* ``no_node``           a class resolves to no taxonomy node at all;
* ``unknown_node``      a node id (taxon / form / condition) is not a node of the pinned taxonomy.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import GateResult, finish
from .labelmap import AXES, labelmap_config, load_release, node_namespaces, resolve_node
from .v11_common import Collector, shards, table_rows

NAME = "labelmap"
CONFIG = "condition-masks"


def run_patch_labelmap_gate(
    build_dir: Path, *, version: str | None = None, registry_root: Path | None = None
) -> GateResult:
    out, cfg = Collector(), labelmap_config(registry_root)
    if version is None:
        from ..label_schemes import current_taxonomy_version

        version = cfg["taxonomy_version"] or current_taxonomy_version()
    try:
        nodes = load_release(version, registry_root)["nodes"]
    except (OSError, ValueError, KeyError) as exc:
        return finish(NAME, {}, [f"taxonomy {version}: release file unusable ({exc})"])
    namespaces, seen = node_namespaces(nodes), set()
    files = shards(Path(build_dir), CONFIG)
    if not files:
        out.fail("no_config", f"no {CONFIG} config in the build: nothing to resolve")
    for row in table_rows(files, ("class_map", "attrs")):
        out.counts["rows"] += 1
        key = (row.get("class_map"), row.get("attrs"))
        if key in seen:  # rows repeat a handful of declarations: check each once
            continue
        seen.add(key)
        try:
            class_map = json.loads(row["class_map"])
            resolution = json.loads(row["attrs"])["class_resolution"]
        except (KeyError, TypeError, ValueError):
            out.fail("class_resolution", f"{row['_shard']}: attrs has no class_resolution")
            continue
        for pixel, name in sorted(class_map.items()):
            entry = resolution.get(name) if isinstance(resolution, dict) else None
            if not isinstance(entry, dict) or str(entry.get("pixel")) != str(pixel):
                out.fail("class_map", f"class {name!r} (pixel {pixel}) has no matching resolution")
                continue
            found = [entry.get(col) for col in AXES if entry.get(col) is not None]
            if not found:
                out.fail("no_node", f"class {name!r} resolves to no taxonomy node")
            for ident in found:
                if not resolve_node(str(ident), nodes, namespaces):
                    out.fail(
                        "unknown_node", f"class {name!r}: {ident!r} is not a node of {version}"
                    )
    return out.result(NAME, taxonomy_version=version, declarations=len(seen))
