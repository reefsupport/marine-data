"""G9 labelmap (v1.1.0): every class id the exported label tables carry resolves in the pinned
taxonomy (a bare id in any of its namespaces) and in the per-source crosswalk, and every fidelity
word is one of the four.

* ``label-map`` (the exported crosswalk, one row per ``(source, native label)``): each node id
  (``taxon`` / ``form`` / ``condition``) is a node of taxonomy ``<version>``. Node keys are
  namespaced (``rs-taxa-v1:A137039``) while edges and table rows carry bare ids, so a bare id
  resolves in every namespace the taxonomy's node keys have; one that resolves in more than one
  is counted as ``ambiguous_bare_ids``, not failed. A mapped row is that release's edge
  ``<crosswalk_id>:<native_label>`` with the same fidelity and targets; an unmapped row is
  ``unmappable`` and carries no node.
* label tables (``points``, ``bleaching``, ``boxes``, ``semseg``, ``instances``): node ids resolve,
  ``fidelity`` (when set; always, next to a node) is one of the four words, ``taxonomy_version`` is
  the gate's release, and a row with a node names a ``(source_id, label_native)`` that ``label-map``
  maps to the same nodes. Rows without ``label_native`` (semantic masks) skip the crosswalk step.
Rows are checked as distinct value combinations (one pyarrow group-by per shard); row totals stay
exact in ``counts``. Release, pin and fidelity words come from ``registry/gates.yaml``
``labelmap``."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

from ..task_layers.v11_tables import FIDELITY as EDGE_FIDELITY
from . import GateResult, finish
from .config import load_gates_config
from .v11_common import LABEL_TABLES, Collector, shards

NAME = "labelmap"
LABEL_MAP = "label-map"
FIDELITIES = ("exact", "coarsened", "approximate", "unmappable")
AXES = ("taxon_node_id", "form_node_id", "condition_node_id")
AXIS = {"taxon_node_id": "taxon", "form_node_id": "form", "condition_node_id": "condition"}
_MAP_COLUMNS = (
    "source_id",
    "crosswalk_id",
    "native_label",
    *AXES,
    "mapped",
    "fidelity",
    "taxonomy_version",
)
_TABLE_COLUMNS = ("source_id", "label_native", *AXES, "fidelity", "taxonomy_version")


def labelmap_config(root: Path | None = None) -> dict:
    """``labelmap`` of ``registry/gates.yaml``: ``taxonomy_version`` (or ``None``) and
    ``fidelities``."""
    cfg = load_gates_config(root).get(NAME) or {}
    words = cfg.get("fidelities", list(FIDELITIES)) if isinstance(cfg, dict) else None
    if not isinstance(words, list) or not words or not all(isinstance(w, str) for w in words):
        raise ValueError(
            "registry/gates.yaml labelmap.fidelities must be a non-empty list of words"
        )
    pin = cfg.get("taxonomy_version")
    return {"taxonomy_version": str(pin) if pin else None, "fidelities": tuple(words)}


def load_release(version: str, root: Path | None = None) -> dict:
    from ..licence_class import _registry_root

    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError(f"taxonomy version {version!r} is not MAJOR.MINOR.PATCH")
    data = json.loads(
        (_registry_root(root) / "taxonomy" / "releases" / f"{version}.json").read_text()
    )
    return {"nodes": data["nodes"], "edges": data["edges"]}


def node_namespaces(node_keys: Iterable[str]) -> tuple[str, ...]:
    """The namespaces of a taxonomy's node keys (``<namespace>:<id>``), read off the keys."""
    return tuple(sorted({key.split(":", 1)[0] for key in node_keys if ":" in key}))


def resolve_node(ident: str, nodes: Mapping[str, object], namespaces: Sequence[str]) -> list[str]:
    """The node keys a table id names: itself when it is a node key (a qualified id), else
    ``<namespace>:<ident>`` for each namespace of the taxonomy that has it."""
    if ident in nodes:
        return [ident]
    return [key for ns in namespaces if (key := f"{ns}:{ident}") in nodes]


def _distinct(
    files: Sequence[Path], columns: Sequence[str]
) -> Iterator[tuple[Path, int, list[dict]]]:
    """``(shard, row count, distinct value combinations of the columns it has)`` per shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    for path in files:
        wanted = [c for c in columns if c in pq.ParquetFile(path).schema_arrow.names]
        table = pq.read_table(path, columns=wanted)
        if not wanted:
            yield path, table.num_rows, [{}] if table.num_rows else []
            continue
        for i, name in enumerate(table.column_names):
            if pa.types.is_null(table.schema.field(i).type):
                table = table.set_column(i, name, table.column(i).cast(pa.string()))
        yield path, table.num_rows, table.group_by(wanted).aggregate([]).to_pylist()


def run_labelmap_gate(
    build_dir: Path, *, version: str | None = None, registry_root: Path | None = None
) -> GateResult:
    build, out = Path(build_dir), Collector()
    cfg = labelmap_config(registry_root)
    if version is None:
        from ..label_schemes import current_taxonomy_version

        version = cfg["taxonomy_version"] or current_taxonomy_version()
    words = frozenset(cfg["fidelities"])
    try:
        release = load_release(version, registry_root)
    except (OSError, ValueError, KeyError) as exc:
        return finish(NAME, {}, [f"taxonomy {version}: release file unusable ({exc})"])
    nodes, edges = release["nodes"], release["edges"]
    namespaces, ambiguous = node_namespaces(nodes), set()

    def check(where: str, row: dict, *, need_fidelity: bool) -> tuple:
        for col in AXES:
            ident = row.get(col)
            if ident is None:
                continue
            keys = resolve_node(ident, nodes, namespaces)
            if not keys:
                out.fail(
                    "unknown_node", f"{where} {col}={ident!r} is not a node of taxonomy {version}"
                )
            elif len(keys) > 1:
                ambiguous.add(ident)
        fid = row.get("fidelity")
        if (fid is not None or need_fidelity) and fid not in words:
            out.fail("bad_fidelity", f"{where} fidelity {fid!r} is not one of {sorted(words)}")
        stamp = row.get("taxonomy_version")
        if stamp is not None and stamp != version:
            out.fail("stale_taxonomy", f"{where} is stamped {stamp!r}, not {version}")
        return tuple(row.get(c) for c in AXES)

    lm_files = shards(build, LABEL_MAP)
    if not lm_files:
        out.fail(
            "no_label_map", f"no {LABEL_MAP} config: class ids cannot be resolved in the crosswalk"
        )
        return out.result(NAME, taxonomy_version=version)
    mapped: dict[tuple[str, str], tuple] = {}
    for path, total, rows in _distinct(lm_files, _MAP_COLUMNS):
        out.counts["label_map_rows"] += total
        for row in rows:
            sid, cid, native = (
                row.get("source_id"),
                row.get("crosswalk_id"),
                row.get("native_label"),
            )
            where = f"{LABEL_MAP}/{path.name} {sid}:{native}"
            found, fid = check(where, row, need_fidelity=True), row.get("fidelity")
            if not row.get("mapped"):
                if fid != "unmappable" or any(found):
                    out.fail(
                        "fidelity_mismatch", f"{where} is unmapped but {fid!r} with nodes {found}"
                    )
                continue
            if fid == "unmappable":
                out.fail("fidelity_mismatch", f"{where} is mapped but unmappable")
            edge = edges.get(f"{cid}:{native}")
            if edge is None:
                out.fail(
                    "unresolved_edge", f"{where}: no edge {cid}:{native} in taxonomy {version}"
                )
                continue
            want = tuple((edge.get("targets") or {}).get(AXIS[c]) for c in AXES)
            expected = EDGE_FIDELITY.get(str(edge.get("fidelity")), edge.get("fidelity"))
            if expected != fid or want != found:
                out.fail(
                    "edge_mismatch",
                    f"{where} differs from edge {cid}:{native} ({expected}, {want})",
                )
            mapped[(str(sid), str(native))] = found
    for table in LABEL_TABLES:
        for path, total, rows in _distinct(shards(build, table), _TABLE_COLUMNS):
            out.counts[f"{table}_rows"] += total
            for row in rows:
                sid, native = row.get("source_id"), row.get("label_native")
                where = f"{table}/{path.name} {sid}:{native}"
                found = check(
                    where,
                    row,
                    need_fidelity=any(found_ is not None for found_ in (row.get(c) for c in AXES)),
                )
                if any(found) and row.get("fidelity") == "unmappable":
                    out.fail("fidelity_mismatch", f"{where} carries nodes but is unmappable")
                if any(found) and native is not None:
                    known = mapped.get((str(sid), str(native)))
                    if known is None:
                        out.fail("unresolved_label", f"{where} is not a mapped {LABEL_MAP} entry")
                    elif known != found:
                        out.fail(
                            "label_mismatch",
                            f"{where} nodes {found} differ from {LABEL_MAP} {known}",
                        )
    return out.result(
        NAME,
        taxonomy_version=version,
        mapped_labels=len(mapped),
        ambiguous_bare_ids=len(ambiguous),
    )
