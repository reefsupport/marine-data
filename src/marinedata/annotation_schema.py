"""The unified annotation schema: one parquet table per task, one set of common columns (WP-U1).

Spec: ``UNIFIED-SCHEMA.md`` section 3 (open-dataset session, 2026-10-05). Modelled on
:mod:`marinedata.sample_schema`: a frozen column spec per table, :func:`arrow_schema`,
row/table validators and a ``SCHEMA_VERSION``. Every annotation of every task lives in
``data/_annotations/<task>/<source_id>/<source_version>.parquet`` (:func:`annotation_path`).

Join key: ``image_sha256`` + ``source_id`` + ``source_version`` + ``ann_id`` (:data:`JOIN_KEY`).
``image_sha256`` joins to :class:`~marinedata.sample_schema.SampleRow`; ``ann_id`` is
``<source_id>:<ordinal>``, stable across re-runs and unique per (source_id, source_version).

Null semantics (a null always means one specific thing):

* ``label_native`` is the source's own string, byte-exact. It is required except on a
  semantic mask row (the per-class strings live in ``class_map``) and on ``identities``.
  ``label_native_id`` null = the source has no id of its own.
* ``taxon_node_id`` / ``form_node_id`` / ``condition_node_id`` are nodes of taxonomy 2.6.0
  (the three ``rs-benthic-v1`` axes). ``match_type = unmapped`` means all three are null and so
  are the derived ``taxon_rank`` / ``worms_aphia_id`` / ``rs_benthic_code``. Any other match
  type needs at least one node (semantic masks excepted: their classes map per pixel value).
  The derived columns are only ever set next to a ``taxon_node_id``.
* ``annotator_type`` has five values. ``annotator_detail`` is an optional refinement, e.g.
  ``"rule"`` for a ``derived_rule`` origin that maps to ``pseudo`` (:func:`annotator_from_origin`).
* ``ann_license`` / ``ann_attribution`` are the *annotation's* own terms and may differ from the
  image's (CoralSCOP pseudo-masks). Null = the annotation carries no terms beyond the image's.
  Unlike ``SampleRow.license`` they are nullable: a null is "same as the image", never a guess.
* ``confidence`` is in [0, 1] when the source provides one, else null (never invented).
* ``upstream_split`` is already normalised with :func:`~marinedata.sample_schema.normalise_split`.
* Normalised coordinates are float32 in [0, 1]. Validators accept values up to
  :data:`COORD_TOLERANCE` outside that range; :func:`to_table` clips them onto it.
* The taxon-free tables (``captions``, ``vqa``, ``depth``, ``pairs``) carry the keys, the
  provenance columns and their own payload, and no label or taxon column at all.
* ``tracks.image_sha256`` is null when no frame image was staged (``video_id`` + ``frame_idx``
  then locate the frame).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import cast

from .sample_schema import SPLIT_HINTS, normalise_split

__all__ = [
    "ANNOTATOR_TYPES",
    "COORD_TOLERANCE",
    "JOIN_KEY",
    "MATCH_TYPES",
    "SCHEMA_VERSION",
    "TABLES",
    "AnnotationSchemaError",
    "AnnotatorType",
    "Col",
    "MatchType",
    "TableSpec",
    "annotation_path",
    "annotator_from_origin",
    "arrow_schema",
    "normalise_split",
    "read_annotations",
    "table_spec",
    "to_table",
    "validate_row",
    "validate_rows",
    "validate_table",
    "write_annotations",
]

SCHEMA_VERSION = "1"
COORD_TOLERANCE = 1e-6
JOIN_KEY = ("image_sha256", "source_id", "source_version", "ann_id")
_INT32_MAX = 2**31 - 1


class AnnotationSchemaError(ValueError):
    """A row or table violates the unified annotation schema."""


class MatchType(str, Enum):
    """How a native label relates to its taxonomy node (spec 3.3)."""

    EXACT = "exact"
    BROADER = "broader"  # the node is an ancestor of the native meaning (was `coarsened`)
    NARROWER = "narrower"  # the native label spans several children; stored, never positive
    RELATED = "related"  # was `approximate`
    UNMAPPED = "unmapped"  # was `unmappable`


class AnnotatorType(str, Enum):
    """Who produced the annotation (spec 3.4)."""

    EXPERT = "expert"
    HUMAN = "human"
    CROWD = "crowd"
    MODEL = "model"
    PSEUDO = "pseudo"


MATCH_TYPES = frozenset(m.value for m in MatchType)
ANNOTATOR_TYPES = frozenset(a.value for a in AnnotatorType)
LABEL_STATUSES = frozenset({"ok", "conflict", "ambiguous", "flagged_hard"})  # D-U values
MASK_KINDS = frozenset({"semantic", "instance"})
CAPTION_TYPES = frozenset({"caption", "summary", "qa-answer"})
DEPTH_UNITS = frozenset({"m", "disparity_px", "relative"})
GT_TYPES = frozenset({"sensor", "stereo", "sfm", "synthetic", "estimated"})
PAIR_ROLES = frozenset({"degraded", "enhanced", "stereo_right", "sonar"})
INDIVIDUAL_SCOPES = frozenset({"dataset", "site"})

# label-origin.yaml origin (and the producers' literals) -> (annotator_type, annotator_detail).
_ORIGIN_MAP: dict[str, tuple[str, str | None]] = {
    "human_expert": ("expert", None),
    "human_crowd": ("crowd", None),
    "human": ("human", None),
    "pseudo_model": ("pseudo", None),
    "pseudo": ("pseudo", None),
    "model": ("model", None),
    "derived_rule": ("pseudo", "rule"),
}

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ORDINAL = re.compile(r"^[0-9]+$")
_LANG = re.compile(r"^[A-Za-z]{2,3}(-[A-Za-z0-9]{2,8})*$")
_LICENSE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+:_()\- ]*$")  # SPDX id or expression
_PIXEL_KEY = re.compile(r"^-?[0-9]+$")


def annotator_from_origin(origin: str) -> tuple[str, str | None]:
    """``(annotator_type, annotator_detail)`` for a label-origin value (spec 3.4).

    ``derived_rule`` is ``("pseudo", "rule")``. ``unknown`` and anything else raises: an
    annotation never gets an invented annotator."""
    try:
        return _ORIGIN_MAP[origin]
    except KeyError:
        raise AnnotationSchemaError(f"no annotator_type for label origin {origin!r}") from None


@dataclass(frozen=True)
class Col:
    """One column: ``kind`` is str | int32 | int64 | float32 | bool."""

    name: str
    kind: str
    nullable: bool = True


@dataclass(frozen=True)
class TableSpec:
    """One annotation table: its columns in storage order."""

    name: str
    columns: tuple[Col, ...]
    has_taxon: bool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)


def _c(name: str, kind: str = "str", *, req: bool = False) -> Col:
    return Col(name, kind, nullable=not req)


def _keys(*, image_required: bool = True) -> tuple[Col, ...]:
    return (
        _c("image_sha256", req=image_required),
        _c("source_id", req=True),
        _c("source_version", req=True),
        _c("ann_id", req=True),
    )


_LABEL = (
    _c("label_native"),
    _c("label_native_id"),
    _c("label_set", req=True),
    _c("taxon_node_id"),
    _c("form_node_id"),
    _c("condition_node_id"),
    _c("taxon_rank"),
    _c("worms_aphia_id", "int64"),
    _c("rs_benthic_code"),
    _c("match_type", req=True),
)
_PROVENANCE = (
    _c("annotator_type", req=True),
    _c("annotator_detail"),
    _c("ann_license"),
    _c("ann_attribution"),
    _c("confidence", "float32"),
    _c("upstream_split"),
    _c("label_status", req=True),
)
_ATTRS = (_c("attrs"),)


def _box(*, required: bool, crowd: bool = False) -> tuple[Col, ...]:
    """The box columns. Normalised xyxy is required where a box is the annotation; the pixel
    columns and image size are conveniences and stay nullable; ``is_crowd`` is required on
    ``boxes`` only (null on a track or an identity)."""
    return (
        _c("x_min", "float32", req=required),
        _c("y_min", "float32", req=required),
        _c("x_max", "float32", req=required),
        _c("y_max", "float32", req=required),
        _c("x_min_px", "int32"),
        _c("y_min_px", "int32"),
        _c("x_max_px", "int32"),
        _c("y_max_px", "int32"),
        _c("is_crowd", "bool", req=crowd),
        _c("img_w", "int32"),
        _c("img_h", "int32"),
    )


def _taxon_table(name: str, extra: tuple[Col, ...], *, image_required: bool = True) -> TableSpec:
    cols = _keys(image_required=image_required) + _LABEL + _PROVENANCE + _ATTRS + extra
    return TableSpec(name, cols, has_taxon=True)


def _free_table(name: str, extra: tuple[Col, ...]) -> TableSpec:
    return TableSpec(name, _keys() + _PROVENANCE + extra, has_taxon=False)


TABLES: dict[str, TableSpec] = {
    t.name: t
    for t in (
        _taxon_table("boxes", _box(required=True, crowd=True)),
        _taxon_table(
            "masks",
            (
                _c("mask_kind", req=True),
                _c("mask_ref"),
                _c("rle"),
                _c("polygon"),
                _c("class_map"),
                _c("ignore_value", "int32"),
                _c("class_counts"),
                _c("canonical_class_counts"),
            ),
        ),
        _taxon_table(
            "points",
            (
                _c("x", "float32", req=True),
                _c("y", "float32", req=True),
                _c("x_px", "int32"),
                _c("y_px", "int32"),
            ),
        ),
        _taxon_table("image_labels", (_c("label_key", req=True),)),
        _taxon_table(
            "tracks",
            (
                _c("video_id", req=True),
                _c("frame_idx", "int32", req=True),
                _c("track_id", req=True),
                *_box(required=True),
            ),
            image_required=False,
        ),
        _free_table(
            "captions",
            (_c("text", req=True), _c("lang", req=True), _c("caption_type", req=True)),
        ),
        _free_table(
            "vqa",
            (
                _c("question", req=True),
                _c("answer", req=True),
                _c("qa_type", req=True),
                _c("lang", req=True),
            ),
        ),
        _free_table(
            "depth",
            (
                _c("depth_ref", req=True),
                _c("units", req=True),
                _c("gt_type", req=True),
                _c("valid_mask_ref"),
            ),
        ),
        _free_table("pairs", (_c("pair_role", req=True), _c("ref_image_sha256", req=True))),
        _taxon_table(
            "identities",
            (
                _c("individual_id", req=True),
                _c("individual_scope", req=True),
                *_box(required=False),
            ),
        ),
    )
}

_UNIT_COLUMNS = frozenset({"x_min", "y_min", "x_max", "y_max", "x", "y", "confidence"})
_SORT_PREFIX = re.compile(r"^(.*):([0-9]+)$")


def table_spec(table: str) -> TableSpec:
    try:
        return TABLES[table]
    except KeyError:
        raise AnnotationSchemaError(
            f"unknown annotation table {table!r}; expected one of {sorted(TABLES)}"
        ) from None


def annotation_path(data_dir: Path, table: str, source_id: str, source_version: str) -> Path:
    """``<data_dir>/_annotations/<table>/<source_id>/<source_version>.parquet``."""
    table_spec(table)
    return data_dir / "_annotations" / table / source_id / f"{source_version}.parquet"


def arrow_schema(table: str):
    """The pyarrow schema of ``table`` (imported lazily: pyarrow is the ``ingest`` extra)."""
    import pyarrow as pa

    types = {
        "str": pa.string(),
        "int32": pa.int32(),
        "int64": pa.int64(),
        "float32": pa.float32(),
        "bool": pa.bool_(),
    }
    spec = table_spec(table)
    return pa.schema(
        [pa.field(c.name, types[c.kind], nullable=c.nullable) for c in spec.columns],
        metadata={
            b"marinedata.annotation_schema": SCHEMA_VERSION.encode(),
            b"marinedata.annotation_table": table.encode(),
        },
    )


# --- row validation -------------------------------------------------------------------------


def _type_error(col: Col, v: object) -> str | None:
    """Why ``v`` (not None) does not fit ``col``'s kind, else None."""
    if col.kind == "str":
        if not isinstance(v, str):
            return f"{col.name} must be a string"
        return f"{col.name} is empty" if not v.strip() else None
    if col.kind == "bool":
        return None if isinstance(v, bool) else f"{col.name} must be a bool"
    if col.kind in ("int32", "int64"):
        if isinstance(v, bool) or not isinstance(v, int):
            return f"{col.name} must be an integer"
        hi = _INT32_MAX if col.kind == "int32" else 2**63 - 1
        return None if -hi - 1 <= v <= hi else f"{col.name} out of {col.kind} range"
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return f"{col.name} must be a number"
    return None if math.isfinite(v) else f"{col.name} is not finite"


def _in_unit(v: float) -> bool:
    return -COORD_TOLERANCE <= v <= 1.0 + COORD_TOLERANCE


def _f(row: Mapping[str, object], name: str) -> float:
    return cast(float, row[name])


def _i(row: Mapping[str, object], name: str) -> int:
    return cast(int, row[name])


def _json_error(name: str, v: object, want: type) -> str | None:
    try:
        parsed = json.loads(str(v))
    except ValueError:
        return f"{name} is not valid JSON"
    return None if isinstance(parsed, want) else f"{name} must be a JSON {want.__name__}"


def _common_checks(spec: TableSpec, row: Mapping[str, object], errs: list[str]) -> None:
    sha, sid = row.get("image_sha256"), row.get("source_id")
    if sha is not None and not _HEX64.match(str(sha)):
        errs.append("image_sha256 is not 64 lowercase hex")
    ann_id = str(row.get("ann_id"))
    head, _, ordinal = ann_id.rpartition(":")
    if head != sid or not _ORDINAL.match(ordinal):
        errs.append("ann_id must be '<source_id>:<ordinal>'")
    enums = (
        ("annotator_type", ANNOTATOR_TYPES),
        ("label_status", LABEL_STATUSES),
        ("match_type", MATCH_TYPES if spec.has_taxon else None),
    )
    for name, allowed in enums:
        if allowed is not None and row.get(name) not in allowed:
            errs.append(f"{name} {row.get(name)!r} not in {sorted(allowed)}")
    if row.get("confidence") is not None and not _in_unit(_f(row, "confidence")):
        errs.append("confidence outside [0, 1]")
    split = row.get("upstream_split")
    if split is not None and (normalise_split(str(split)) != split or split not in SPLIT_HINTS):
        errs.append("upstream_split is not normalised (train|val|test)")
    lic = row.get("ann_license")
    if lic is not None and not _LICENSE.match(str(lic)):
        errs.append("ann_license is not an SPDX-like id or expression")
    attrs = row.get("attrs")
    if attrs is not None and (msg := _json_error("attrs", attrs, dict)):
        errs.append(msg)


def _taxon_checks(table: str, row: Mapping[str, object], errs: list[str]) -> None:
    nodes = [row.get(n) for n in ("taxon_node_id", "form_node_id", "condition_node_id")]
    derived = [row.get(n) for n in ("taxon_rank", "worms_aphia_id", "rs_benthic_code")]
    semantic = table == "masks" and row.get("mask_kind") == "semantic"
    if row.get("match_type") == "unmapped":
        if any(v is not None for v in nodes + derived):
            errs.append("match_type unmapped needs null node and derived columns")
    elif not semantic and all(v is None for v in nodes):
        errs.append("a mapped annotation needs a taxon, form or condition node")
    if any(v is not None for v in derived) and row.get("taxon_node_id") is None:
        errs.append("taxon_rank / worms_aphia_id / rs_benthic_code need a taxon_node_id")
    if row.get("worms_aphia_id") is not None and _i(row, "worms_aphia_id") <= 0:
        errs.append("worms_aphia_id must be positive")
    if row.get("label_native") is None and not semantic and table != "identities":
        errs.append("label_native is required")


def _box_checks(row: Mapping[str, object], errs: list[str]) -> None:
    keys = ("x_min", "y_min", "x_max", "y_max")
    pkeys = ("x_min_px", "y_min_px", "x_max_px", "y_max_px")
    present = [row.get(k) is not None for k in keys]
    if not any(present):
        if any(row.get(k) is not None for k in pkeys):
            errs.append("pixel box set without a normalised box")
        return
    if not all(present):
        errs.append("box coordinates must be all set or all null")
        return
    norm = [_f(row, k) for k in keys]
    if not all(_in_unit(v) for v in norm):
        errs.append("box coordinate outside [0, 1]")
    if not (norm[0] < norm[2] and norm[1] < norm[3]):
        errs.append("box needs x_min < x_max and y_min < y_max")
    w, h = row.get("img_w"), row.get("img_h")
    if (w is None) != (h is None) or (
        w is not None and (_i(row, "img_w") <= 0 or _i(row, "img_h") <= 0)
    ):
        errs.append("img_w and img_h must both be set and positive")
    ppresent = [row.get(k) is not None for k in pkeys]
    if not any(ppresent):
        return
    if not all(ppresent) or w is None or h is None:
        errs.append("pixel box needs all four pixel columns and img_w/img_h")
        return
    px = [_i(row, k) for k in pkeys]
    dims = (_i(row, "img_w"), _i(row, "img_h"), _i(row, "img_w"), _i(row, "img_h"))
    for p, n, dim in zip(px, norm, dims, strict=True):
        if not 0 <= p <= dim:
            errs.append("pixel box outside the image")
            return
        if abs(p / dim - n) > 1.0 / dim + COORD_TOLERANCE:
            errs.append("pixel box disagrees with the normalised box")
            return
    if not (px[0] < px[2] and px[1] < px[3]):
        errs.append("pixel box needs x_min_px < x_max_px and y_min_px < y_max_px")


def _masks_checks(row: Mapping[str, object], errs: list[str]) -> None:
    kind = row.get("mask_kind")
    if kind not in MASK_KINDS:
        errs.append(f"mask_kind {kind!r} not in {sorted(MASK_KINDS)}")
        return
    if (p := row.get("polygon")) is not None and (msg := _json_error("polygon", p, list)):
        errs.append(msg)
    for name in ("class_counts", "canonical_class_counts"):
        if (v := row.get(name)) is not None and (msg := _json_error(name, v, dict)):
            errs.append(msg)
    cm = row.get("class_map")
    if kind == "semantic":
        if row.get("mask_ref") is None:
            errs.append("a semantic mask needs mask_ref")
        if row.get("rle") is not None or row.get("polygon") is not None:
            errs.append("a semantic mask carries no rle or polygon")
        if cm is None:
            errs.append("a semantic mask needs class_map")
        elif msg := _json_error("class_map", cm, dict):
            errs.append(msg)
        else:
            m = json.loads(str(cm))
            if not all(_PIXEL_KEY.match(k) and isinstance(v, str) for k, v in m.items()):
                errs.append("class_map must map pixel values to label strings")
    else:
        if all(row.get(n) is None for n in ("mask_ref", "rle", "polygon")):
            errs.append("an instance mask needs mask_ref, rle or polygon")
        if cm is not None:
            errs.append("class_map is for semantic masks only")


def _points_checks(row: Mapping[str, object], errs: list[str]) -> None:
    if not (_in_unit(_f(row, "x")) and _in_unit(_f(row, "y"))):
        errs.append("point coordinate outside [0, 1]")
    xp, yp = row.get("x_px"), row.get("y_px")
    if (xp is None) != (yp is None):
        errs.append("x_px and y_px must both be set or both null")
    elif xp is not None and (_i(row, "x_px") < 0 or _i(row, "y_px") < 0):
        errs.append("pixel point is negative")


def _tracks_checks(row: Mapping[str, object], errs: list[str]) -> None:
    if _i(row, "frame_idx") < 0:
        errs.append("frame_idx must be >= 0")
    _box_checks(row, errs)


def _enum_check(
    name: str, row: Mapping[str, object], allowed: frozenset[str], errs: list[str]
) -> None:
    if row.get(name) not in allowed:
        errs.append(f"{name} {row.get(name)!r} not in {sorted(allowed)}")


def _captions_checks(row: Mapping[str, object], errs: list[str]) -> None:
    _enum_check("caption_type", row, CAPTION_TYPES, errs)
    if not _LANG.match(str(row["lang"])):
        errs.append("lang is not a BCP-47 tag")


def _vqa_checks(row: Mapping[str, object], errs: list[str]) -> None:
    if not _LANG.match(str(row["lang"])):
        errs.append("lang is not a BCP-47 tag")


def _depth_checks(row: Mapping[str, object], errs: list[str]) -> None:
    _enum_check("units", row, DEPTH_UNITS, errs)
    _enum_check("gt_type", row, GT_TYPES, errs)


def _pairs_checks(row: Mapping[str, object], errs: list[str]) -> None:
    _enum_check("pair_role", row, PAIR_ROLES, errs)
    if not _HEX64.match(str(row["ref_image_sha256"])):
        errs.append("ref_image_sha256 is not 64 lowercase hex")


def _identities_checks(row: Mapping[str, object], errs: list[str]) -> None:
    _enum_check("individual_scope", row, INDIVIDUAL_SCOPES, errs)
    _box_checks(row, errs)


_TABLE_CHECKS = {
    "boxes": _box_checks,
    "masks": _masks_checks,
    "points": _points_checks,
    "tracks": _tracks_checks,
    "captions": _captions_checks,
    "vqa": _vqa_checks,
    "depth": _depth_checks,
    "pairs": _pairs_checks,
    "identities": _identities_checks,
}


def validate_row(table: str, row: Mapping[str, object]) -> list[str]:
    """Every violation in ``row`` (empty list = valid). Never raises for a bad row.

    A missing key counts as null. Type errors are reported alone: the semantic checks only run
    on a row whose columns all have the right type."""
    spec = table_spec(table)
    errs = [f"unknown column {n!r}" for n in sorted(set(row) - set(spec.names))]
    for col in spec.columns:
        v = row.get(col.name)
        if v is None:
            if not col.nullable:
                errs.append(f"{col.name} is required")
        elif msg := _type_error(col, v):
            errs.append(msg)
    if errs:
        return errs
    _common_checks(spec, row, errs)
    if spec.has_taxon:
        _taxon_checks(table, row, errs)
    if check := _TABLE_CHECKS.get(table):
        check(row, errs)
    return errs


def validate_rows(table: str, rows: Iterable[Mapping[str, object]]) -> None:
    """Raise :class:`AnnotationSchemaError` listing the first violations (and duplicate keys)."""
    problems: list[str] = []
    seen: set[tuple[object, object, object]] = set()
    for row in rows:
        key = (row.get("source_id"), row.get("source_version"), row.get("ann_id"))
        problems += [f"{key[2] or '?'}: {e}" for e in validate_row(table, row)]
        if key in seen:
            problems.append(f"{key[2]}: duplicate ann_id within (source_id, source_version)")
        seen.add(key)
        if len(problems) > 20:
            break
    if problems:
        raise AnnotationSchemaError("; ".join(problems[:20]))


# --- tables and parquet ---------------------------------------------------------------------


def _ordinal(row: Mapping[str, object]) -> tuple[str, int]:
    m = _SORT_PREFIX.match(str(row["ann_id"]))
    return (str(row["source_version"]), int(m.group(2)) if m else 0)


def _clip(name: str, v: object) -> object:
    if v is None or name not in _UNIT_COLUMNS:
        return v
    return min(max(cast(float, v), 0.0), 1.0)


def to_table(table: str, rows: Sequence[Mapping[str, object]]):
    """Validated pyarrow table in the canonical schema, rows sorted by ordinal.

    Unit-range columns that sit within :data:`COORD_TOLERANCE` of the range are clipped onto it."""
    import pyarrow as pa

    spec = table_spec(table)
    ordered = sorted(rows, key=_ordinal)
    validate_rows(table, ordered)
    cols = {c.name: [_clip(c.name, r.get(c.name)) for r in ordered] for c in spec.columns}
    return pa.table(cols, schema=arrow_schema(table))


def write_annotations(path: Path, table: str, rows: Sequence[Mapping[str, object]]) -> None:
    """Write one annotation parquet (zstd, sorted rows, WITH column statistics: spec U14)."""
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_table(table, rows), path, compression="zstd", write_statistics=True)


def validate_table(table: str, tbl) -> None:
    """Check a read-back table: exact column names and types, then every row."""
    expected = arrow_schema(table)
    if [f.name for f in tbl.schema] != list(table_spec(table).names):
        raise AnnotationSchemaError(f"columns differ from annotation schema v{SCHEMA_VERSION}")
    for got in tbl.schema:
        want = expected.field(got.name)
        if not got.type.equals(want.type):
            raise AnnotationSchemaError(f"{got.name}: type {got.type} != {want.type}")
    validate_rows(table, tbl.to_pylist())


def read_annotations(path: Path, table: str):
    """Read an annotation parquet back as a validated pyarrow table."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    validate_table(table, tbl)
    return tbl
