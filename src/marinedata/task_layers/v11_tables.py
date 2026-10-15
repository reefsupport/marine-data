"""v1.1.0 build stage (PR-B): the stamped task tables and the ``metadata`` spine.

``release build --tasks v2 --v11`` turns the task-layer configs into the v1.1.0 task tables:

* ``metadata`` (B7): one row per ``image_sha256`` of every admitted source's staged
  ``metadata.parquet`` (v1.0 images included): the ``metadata`` Hub-config columns plus
  ``bytes_config`` (the config holding the pixels: the v1.0 config that embeds the image, else
  ``images``), ``bytes_sha256`` (null here; the export fills it when it rewrites bytes),
  ``split`` and ``taxonomy_version``. A row never carries bytes.
* ``points`` (B1), ``bleaching`` (B2), ``boxes`` (B5, B4), ``semseg`` and ``instances`` (B3, B4):
  the existing builders' rows stamped with ``split``, ``taxonomy_version``, ``fidelity`` (the
  crosswalk fidelity of ``match_type``) and ``bytes_config`` (``mask_config`` / ``mask_ref`` on
  masks), typed by the WP-U1 table of the same name (:mod:`marinedata.annotation_schema`).

Taxonomy 3.0.0 validates a row against its stamped release, so every row of a new table carries
``taxonomy_version`` = the build's release; the v1.0 tables stay under 2.9.0, never rebuilt.
A row whose image has no split is written with ``split = null`` and counted (``unsplit``); an
image of a withheld near-dup group is dropped from every table; a row of a source outside
``V11Inputs.roots`` (the flavour's admitted set) is dropped, so an nc source cannot reach the
open tables through this layer.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..annotation_schema import TABLES
from ..label_schemes import current_taxonomy_version
from ..licence_class import coerce_class, resolve_row_class
from ..registry import Registry
from .configs.common import ConfigResult, _attrs
from .configs.constants import MASK_CONFIG_BY_SOURCE
from .mask_classmap import palette_json

IMAGES_CONFIG = "images"
V1_BYTES_CONFIG: dict[str, str] = {**MASK_CONFIG_BY_SOURCE, "roboflow-aquarium": "fish-boxes"}
"""Source -> the frozen v1.0 config that embeds its images. The UIIS family sits in both
``instance-masks`` and ``fish-boxes`` (grandfathered double embed): the mask config wins."""
V11_CONFIG_IDS = ("metadata", "points", "bleaching", "boxes", "semseg", "instances")
ANNOTATION_TABLE = {
    "points": "points", "bleaching": "image_labels", "boxes": "boxes", "semseg": "masks",
    "instances": "masks",
}  # fmt: skip
FIDELITY = {
    "exact": "exact", "broader": "coarsened", "narrower": "approximate", "related": "approximate",
    "unmapped": "unmappable",
}  # fmt: skip
"""``match_type`` (WP-U1) -> the crosswalk fidelity word. ``narrower`` is stored for audit and never
used as a positive, so it reads as approximate."""
_STAMP = (
    ("split", "str"),
    ("taxonomy_version", "str"),
    ("fidelity", "str"),
    ("bytes_config", "str"),
)
_EXTRA_CORE = {
    "bleaching": (("near_dup_group", "str"), ("near_dup_images", "int32")),
    "boxes": (("modality", "str"),),
    "semseg": (("mask_config", "str"), ("palette", "str")),
    "instances": (("modality", "str"), ("mask_config", "str")),
}  # fmt: skip


def bytes_config_of(source_id: str | None) -> str:
    """The config holding ``source_id``'s image bytes: its v1.0 config, else ``images``."""
    return V1_BYTES_CONFIG.get(source_id or "", IMAGES_CONFIG)


def fidelity_of(match_type: str | None) -> str | None:
    return FIDELITY.get(match_type) if match_type else None


@dataclass(frozen=True)
class V11Inputs:
    """What the v1.1.0 layer needs besides the config rows: the build's taxonomy release, the
    admitted source roots, the frozen split map and the near-dup groups."""

    taxonomy_version: str
    roots: Mapping[str, Path]
    split_of: Mapping[str, str] = field(default_factory=dict)
    """``image_sha256 -> split`` from the build's task manifests (wins over ``assignments``)."""
    assignments: Mapping[str, str] = field(default_factory=dict)
    """``split_group -> split``: the frozen (inherited) split map."""
    withheld_groups: frozenset[str] = frozenset()
    near_dup_group_of: Mapping[str, str] = field(default_factory=dict)
    """``image_sha256 -> dedup group`` (``groups.parquet``); else the image's ``split_group``."""


def inputs_from_build(
    *, taxonomy_version: str | None, roots: Mapping[str, str | Path], manifests: Iterable[Any],
    split_map: Any, dedup_groups: str | Path | None = None,
) -> V11Inputs:  # fmt: skip
    """:class:`V11Inputs` from what ``build_release`` already holds."""
    near: Mapping[str, str] = {}
    if dedup_groups is not None:
        from ..dedup.groups import load_groups

        near = load_groups(dedup_groups)[0]
    return V11Inputs(
        taxonomy_version=taxonomy_version or current_taxonomy_version(),
        roots={sid: Path(root) for sid, root in roots.items()},
        split_of={sha: split for m in manifests for sha, split in m.rows},
        assignments=dict(split_map.assignments),
        withheld_groups=frozenset(split_map.withheld),
        near_dup_group_of=dict(near),
    )


def metadata_columns() -> tuple[tuple[str, str], ...]:
    """The ``metadata`` Hub-config columns (SampleRow v3 names) plus the v1.1.0 additions."""
    from ..metadata_norm.resolvers.join import EXTRA_COLUMNS
    from ..metadata_release import METADATA_COLUMNS

    cols = dict(METADATA_COLUMNS)
    for name, kind in EXTRA_COLUMNS:
        cols.setdefault(name, kind)
    cols.setdefault("split", "string")
    cols.update(bytes_config="string", bytes_sha256="string", taxonomy_version="string")
    return tuple(cols.items())


def core_columns(config_id: str) -> tuple[tuple[str, str], ...]:
    """The typed columns of one v1.1.0 table (the rest are inferred from the rows)."""
    if config_id == "metadata":
        return metadata_columns()
    base = tuple((c.name, c.kind) for c in TABLES[ANNOTATION_TABLE[config_id]].columns)
    have = {n for n, _ in base}
    more = (*_EXTRA_CORE.get(config_id, ()), *_STAMP)
    return base + tuple(c for c in more if c[0] not in have)


_ARROW_KINDS = {
    "str": "string", "string": "string", "int16": "int16", "int32": "int32", "int64": "int64",
    "float32": "float32", "double": "float64", "float64": "float64", "bool": "bool_",
}  # fmt: skip


def _arrow_type(kind: str):
    """The pyarrow type of a declared column kind; ``None`` for a kind this layer does not know
    (the column is then typed from its rows)."""
    import pyarrow as pa

    name = _ARROW_KINDS.get(kind)
    return getattr(pa, name)() if name else None


def _inferred_type(rows: Sequence[Mapping[str, Any]], key: str):
    import pyarrow as pa

    kind = pa.array([r.get(key) for r in rows]).type
    return pa.string() if pa.types.is_null(kind) else kind


def table_schema(config_id: str, rows: Sequence[Mapping[str, Any]] = ()):
    """The pyarrow schema of one v1.1.0 table: :func:`core_columns` in declared order and type,
    then every other column of ``rows`` typed from its values (an all-null column is a string)."""
    import pyarrow as pa

    core = core_columns(config_id)
    fields = [pa.field(n, _arrow_type(k) or _inferred_type(rows, n)) for n, k in core]
    known = {n for n, _ in core}
    extra = list(dict.fromkeys(k for r in rows for k in r if k not in known))
    return pa.schema(fields + [pa.field(k, _inferred_type(rows, k)) for k in extra])


def v11_schemas(results: Mapping[str, ConfigResult]) -> dict[str, Any]:
    return {
        cid: table_schema(cid, results[cid].rows if cid in results else ())
        for cid in V11_CONFIG_IDS
    }


def _plain(value: Any) -> Any:
    return value.isoformat() if isinstance(value, (dt.date, dt.datetime)) else value


def _metadata_rows(registry: Registry, inputs: V11Inputs) -> dict[str, dict]:
    import pyarrow.parquet as pq

    names = [n for n, _ in metadata_columns()]
    seen: dict[str, dict] = {}
    order = sorted(inputs.roots, key=lambda s: (bytes_config_of(s) == IMAGES_CONFIG, s))
    for source_id in order:  # v1.0-embedded sources first: their config holds the bytes
        path = Path(inputs.roots[source_id]) / "metadata.parquet"
        if not path.is_file():
            continue
        source = registry.source(source_id)
        src_class = coerce_class(source.access_class)
        for rec in pq.read_table(path).to_pylist():
            sha = rec.get("image_sha256")
            if not sha or sha in seen:
                continue
            row = {n: _plain(rec.get(n)) for n in names}
            row.update(
                image_sha256=sha, source_id=source_id, bytes_config=bytes_config_of(source_id),
                taxonomy_version=inputs.taxonomy_version, bytes_sha256=None,
                location_generalized=bool(rec.get("location_generalized")),
                licence_class=resolve_row_class(
                    src_class, rec.get("license"), per_row=bool(source.licence_per_row)
                ),
            )  # fmt: skip
            seen[sha] = row
    return seen


def mark_near_dup_conflicts(
    rows: Iterable[Mapping[str, Any]], group_of: Mapping[str, str]
) -> tuple[dict, ...]:
    """B2: every row gets its ``near_dup_group`` (the image's own sha when it has none) and
    ``near_dup_images``; when the parties of a group — one per (source, image) — disagree on the
    canonical condition, every row of the group gets ``label_status = conflict``. All rows are kept.
    One image carrying several labels (a mask's bleached and healthy regions) is one party."""
    rows = list(rows)
    parties: dict[str, dict[tuple, set]] = defaultdict(lambda: defaultdict(set))
    for r in rows:
        sha = r.get("image_sha256") or r.get("sha256")
        party = parties[group_of.get(sha, sha)][(r.get("source_id"), sha)]
        if r.get("canonical_condition"):
            party.add(r["canonical_condition"])
    conflicted = {g for g, p in parties.items() if len({frozenset(v) for v in p.values() if v}) > 1}
    out = []
    for r in rows:
        sha = r.get("image_sha256") or r.get("sha256")
        group = group_of.get(sha, sha)
        out.append(
            {
                **r, "near_dup_group": group,
                "near_dup_images": len({s for _, s in parties[group]}),
                "label_status": "conflict" if group in conflicted else r.get("label_status", "ok"),
            }
        )  # fmt: skip
    return tuple(out)


def _stamp_row(config_id: str, row: Mapping[str, Any], inputs: V11Inputs, split_of) -> dict:
    sha = row.get("image_sha256") or row.get("sha256")
    source_id = row.get("source_id")
    out = {
        **row, "image_sha256": sha, "split": split_of.get(sha),
        "taxonomy_version": inputs.taxonomy_version, "fidelity": fidelity_of(row.get("match_type")),
        "bytes_config": bytes_config_of(source_id),
    }  # fmt: skip
    if config_id in ("boxes", "instances"):
        out["modality"] = _attrs(row).get("modality") or row.get("modality")
    if config_id in ("semseg", "instances"):
        # mask_config: the v1.0 config embedding this mask (null: the export embeds it)
        out["mask_config"] = MASK_CONFIG_BY_SOURCE.get(source_id or "")
        out["mask_ref"] = row.get("mask_ref") or row.get("mask_key")
    if config_id == "semseg":
        out["palette"] = palette_json(row.get("hf_class_map"))
    return out


def build_v11_tables(
    registry: Registry, results: Mapping[str, ConfigResult], inputs: V11Inputs
) -> dict[str, ConfigResult]:
    """The six v1.1.0 tables (``metadata`` plus the five stamped configs) from the built
    ``results``; the metadata result's ``notes`` carries the build report."""
    meta = _metadata_rows(registry, inputs)
    split_of = dict(inputs.split_of)
    withheld: set[str] = set()
    meta_rows: list[dict] = []
    for sha, row in meta.items():
        group = row.get("split_group")
        if group in inputs.withheld_groups:
            withheld.add(sha)
            continue
        split = split_of.get(sha) or inputs.assignments.get(group)
        if split:
            split_of[sha] = split
        meta_rows.append({**row, "split": split})
    group_of = {sha: r["split_group"] for sha, r in meta.items() if r.get("split_group")}
    group_of.update(inputs.near_dup_group_of)
    tables: dict[str, ConfigResult] = {}
    report: dict[str, Any] = {"taxonomy_version": inputs.taxonomy_version, "tables": {}}
    for cid in V11_CONFIG_IDS[1:]:
        base = results.get(cid)
        rows = base.rows if base else ()
        if cid == "bleaching":
            rows = mark_near_dup_conflicts(rows, group_of)
        rows = tuple(_stamp_row(cid, r, inputs, split_of) for r in rows)
        rows = tuple(
            r
            for r in rows
            if r["image_sha256"] not in withheld and r.get("source_id") in inputs.roots
        )
        tables[cid] = ConfigResult(cid, rows, dict(base.unmapped_by_source) if base else {})
    tables = {"metadata": ConfigResult("metadata", tuple(meta_rows), {}), **tables}
    for cid, res in tables.items():
        report["tables"][cid] = {
            "rows": len(res.rows), "images": res.n_images,
            "unsplit": sum(1 for r in res.rows if r.get("split") is None),
        }  # fmt: skip
    report["withheld_images"] = len(withheld)
    notes = {"report": json.loads(json.dumps(report))}
    return {**tables, "metadata": ConfigResult("metadata", tuple(meta_rows), {}, notes=notes)}


def write_v11_report(results: Mapping[str, ConfigResult], out_dir: str | Path) -> Path:
    """``V11_TABLES.json`` next to the tables: taxonomy version, rows / images / unsplit rows per
    table and the withheld-image count (the build's own accounting for release gate G6)."""
    meta = results.get("metadata")
    report = dict(meta.notes.get("report", {})) if meta is not None else {}
    path = Path(out_dir) / "V11_TABLES.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return path
