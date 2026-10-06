"""Coral and benthic masks into the unified ``masks`` table (WP-U4).

One *semantic* row per image/mask pair in :mod:`marinedata.annotation_schema`'s ``masks`` table:
``mask_ref`` is the bucket key of the PNG, ``class_map`` maps each pixel value present to the source's
own label (byte-exact), ``class_counts`` / ``canonical_class_counts`` keep the per-image pixel
histograms (native label / ``rs-benthic-v1`` taxon node). Per-class resolution
(:meth:`marinedata.schema.Crosswalk.resolve`) lives in ``attrs.class_resolution``; the row's own
taxon columns stay null (semantic masks map per pixel value, U1) and its ``match_type`` is the
weakest match among the *mapped* classes, ``unmapped`` when none maps. An index missing from the
source's table is labelled ``__unknown_index_<n>`` and is ``unmapped``: never guessed.

Five staged sources (:data:`MASK_SOURCES`), read through :class:`~.s3_keyed.StagedTree`. The
per-instance Labelbox masks of ``reef-support-seaview-labels`` (``labels/instance_masks/<partition>/``,
classes in ``labels/exports/*.ndjson``) get *pending* rows (``image_key`` + TODO) the way WP-U3
does.

``attrs.licence_class`` is ``internal-only`` for ``coralscop-masks-rs`` (CC-BY-NC-SA-4.0 masks over
unlicensed UCSD base images); :func:`drop_internal_only` is the release filter.
"""  # noqa: E501

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    arrow_schema,
    normalise_split,
    validate_row,
    write_annotations,
)
from ..licence_class import source_class
from ..registry import Registry
from .points_table import resolver_for
from .producers.coralscapes_semseg import _ID_TO_LABEL as _CORALSCAPES_IDS
from .s3_keyed import StagedTree, _pixel_counts

Resolver = Callable[[str], tuple[str | None, str, int | None, str | None]]
UNKNOWN_PREFIX = "__unknown_index_"
INTERNAL_ONLY = "internal-only"
PENDING_COLUMNS = ("image_key", "image_sha256_todo")
PENDING_LABEL = "TODO: class label lives in export-result.ndjson (not parsed yet)"
TODO_TEXT = "TODO: hash image_key once the image is staged under sources/ (not resolvable today)"
_PLACEHOLDER_SHA = "0" * 64
_MATCH_ORDER = ("exact", "narrower", "broader", "related")  # strongest first


@dataclass(frozen=True)
class MaskSource:
    """How one staged source's masks decode and resolve."""

    source_id: str
    version: str
    crosswalk_id: str
    id_to_label: Mapping[int, str]
    ignore_value: int | None
    annotator: str  # a label-origin value (annotator_from_origin)
    ann_license: str | None
    licence_class: str
    mask_dirs: tuple[str, ...] = ("masks",)
    encoding: str = "png-indexed"
    label_set: str = "dataset-native"

    @property
    def tree(self) -> str:
        return f"{self.source_id}/{self.version}"


MASK_SOURCES: dict[str, MaskSource] = {
    s.source_id: s
    for s in (
        MaskSource(
            "coralscapes",
            "1.0",
            "coralscapes-39",
            _CORALSCAPES_IDS,
            0,
            "human",
            "Apache-2.0",
            "open",
            label_set="coralscapes-39",
        ),
        # D-AI3 / coralseg.py: red channel 0 Other, 1 Hard Coral, 2 Soft Coral.
        MaskSource(
            "coralseg-ucsd-mosaics",
            "unversioned",
            "coralseg-ucsd-mosaics",
            {0: "Other", 1: "Hard Coral", 2: "Soft Coral"},
            None,
            "human",
            None,
            "tdm-only",
            mask_dirs=("files",),
            encoding="png-rgb-red",
        ),
        # class-agnostic union mask: 1 = coral, 0 = everything else (kept out of class_map).
        MaskSource(
            "coralscop-masks-rs",
            "2026-09-23-3e8612678469",
            "coralscop-masks-rs",
            {1: "coral"},
            0,
            "pseudo_model",
            "CC-BY-NC-SA-4.0",
            INTERNAL_ONLY,
        ),
        # same encoding as Coralseg's mask (0 unlabelled, 1 Hard Coral, 2 Soft Coral): verified against  # noqa: E501
        # the vocab counts by the WP-U4 scan (see the report), unknown indexes stay unmapped.
        MaskSource(
            "reef-support-benthic-own",
            "2026-10-06",
            "reef-support-labelbox",
            {1: "Hard Coral", 2: "Soft Coral"},
            0,
            "human_expert",
            "CC-BY-4.0",
            "open",
            label_set="reef-support-labelbox",
        ),
        # per the suim-8class crosswalk header: 4R+2G+B of the authors' RGB table.
        MaskSource(
            "suim",
            "2020",
            "suim-8class",
            {0: "BW", 1: "HD", 2: "PF", 3: "WR", 4: "RO", 5: "RI", 6: "FV", 7: "SR"},
            None,
            "human",
            "MIT",
            "open",
        ),
    )
}

SEAVIEW_LABELS_SOURCE = "reef-support-seaview-labels"
SEAVIEW_LABELS_VERSION = "2026-10-06"
SEAVIEW_LABELS_PREFIX = f"sources/{SEAVIEW_LABELS_SOURCE}/{SEAVIEW_LABELS_VERSION}/"
"""Canonical tree of our SEAVIEW instance masks (partition = SEAVIEW_ATL / IDN_PHL / PAC_AUS / PAC_USA)."""  # noqa: E501


def licence_class(source_id: str) -> str:
    return MASK_SOURCES[source_id].licence_class


def is_internal_only(row: Mapping[str, object]) -> bool:
    attrs = row.get("attrs")
    if not attrs:
        return False
    return (json.loads(attrs) if isinstance(attrs, str) else dict(attrs)).get(
        "licence_class"
    ) == INTERNAL_ONLY  # noqa: E501, RUF100


def drop_internal_only(rows: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    """The release filter: rows whose annotation licence class is not ``internal-only``."""
    return [r for r in rows if not is_internal_only(r)]


def _weakest(matches: Iterable[str]) -> str:
    ranks = [_MATCH_ORDER.index(m) for m in matches if m in _MATCH_ORDER]
    return _MATCH_ORDER[max(ranks)] if ranks else "unmapped"


def semantic_row(
    *,
    spec: MaskSource,
    resolve: Resolver,
    ordinal: int,
    image_sha256: str,
    mask_ref: str,
    counts: Mapping[int, int],
    split: str | None = None,
) -> dict:
    """One semantic ``masks`` row from a mask's pixel histogram ``{pixel value: n}``."""
    class_map: dict[str, str] = {}
    native: dict[str, int] = {}
    canonical: dict[str, int] = {}
    resolution: dict[str, dict] = {}
    for value, n in sorted(counts.items()):
        if value == spec.ignore_value or n <= 0:
            continue
        label = spec.id_to_label.get(value, f"{UNKNOWN_PREFIX}{value}")
        taxon, match, aphia, l2 = (
            resolve(label) if value in spec.id_to_label else (None, "unmapped", None, None)
        )
        class_map[str(value)] = label
        native[label] = native.get(label, 0) + n
        resolution[label] = {
            "pixel": value, "taxon_node_id": taxon, "match_type": match,
            "worms_aphia_id": aphia, "rs_benthic_code": l2,
        }  # fmt: skip
        if taxon is not None and match != "unmapped":
            canonical[taxon] = canonical.get(taxon, 0) + n
    typ, detail = annotator_from_origin(spec.annotator)
    mapped_px = sum(n for lbl, n in native.items() if resolution[lbl]["match_type"] != "unmapped")
    attrs = {
        "mask_encoding": spec.encoding,
        "class_resolution": resolution,
        "licence_class": source_class(spec.source_id, spec.licence_class),
        "mapped_pixels": mapped_px,
        "labelled_pixels": sum(native.values()),
    }
    return {
        "image_sha256": image_sha256,
        "source_id": spec.source_id,
        "source_version": spec.version,
        "ann_id": f"{spec.source_id}:{ordinal}",
        "label_native": None,  # per-class strings live in class_map
        "label_native_id": None,
        "label_set": spec.label_set,
        # semantic masks resolve per pixel value (attrs.class_resolution): no row-level node
        "taxon_node_id": None,
        "form_node_id": None,
        "condition_node_id": None,
        "taxon_rank": None,
        "worms_aphia_id": None,
        "rs_benthic_code": None,
        "match_type": _weakest(r["match_type"] for r in resolution.values()),
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": spec.ann_license,
        "confidence": None,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(attrs, sort_keys=True),
        "mask_kind": "semantic",
        "mask_ref": mask_ref,
        "class_map": json.dumps(class_map, sort_keys=True),
        "ignore_value": spec.ignore_value,
        "class_counts": json.dumps(native, sort_keys=True),
        "canonical_class_counts": json.dumps(canonical, sort_keys=True),
    }


def _mask_items(tree: StagedTree, spec: MaskSource) -> list[tuple[tuple[str, str], str]]:
    """``((partition, stem), rel)`` for every ``labels/<mask_dir>/[<partition>/]<stem>.png``."""
    out = []
    for rel in sorted(tree.checksums):
        parts = PurePosixPath(rel).parts
        if (
            len(parts) >= 3
            and parts[0] == "labels"
            and parts[1] in spec.mask_dirs
            and rel.endswith(".png")
        ):  # noqa: E501, RUF100
            out.append(
                (((parts[2] if len(parts) == 4 else "default"), PurePosixPath(rel).stem), rel)
            )  # noqa: E501, RUF100
    return out


def _splits(tree: StagedTree) -> dict[tuple[str, str], str | None]:
    try:
        rows = tree.table("metadata.parquet")
    except Exception:  # metadata is optional for the join: a missing split is a null split
        return {}
    return {(r.get("partition") or "default", r["stem"]): r.get("upstream_split") for r in rows}


def staged_mask_rows(
    tree: StagedTree,
    spec: MaskSource,
    registry: Registry,
    *,
    limit: int | None = None,
    workers: int = 16,
) -> Iterator[dict]:
    """Semantic rows for the staged masks of ``spec`` (first ``limit`` in key order)."""
    resolve = resolver_for(registry, spec.crosswalk_id)
    items = [(k, rel) for k, rel in _mask_items(tree, spec) if k in tree.shas]
    if limit is not None:
        items = items[:limit]
    splits = _splits(tree)

    def counts(item: tuple[tuple[str, str], str]) -> dict[int, int] | None:
        data = tree.get_or_skip(item[1])
        return None if data is None else _pixel_counts(data)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for ordinal, ((key, rel), got) in enumerate(
            zip(items, pool.map(counts, items), strict=True)
        ):  # noqa: E501, RUF100
            if got is None:
                continue
            yield semantic_row(
                spec=spec, resolve=resolve, ordinal=ordinal, image_sha256=tree.shas[key],
                mask_ref=tree.key(rel), counts=got, split=splits.get(key),
            )  # fmt: skip


def write_masks(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    """Write rows to ``<data_dir>/_annotations/masks/<source_id>/<version>.parquet``."""
    path = annotation_path(data_dir, "masks", source_id, source_version)
    write_annotations(path, "masks", rows)
    return path


# ---- labels/instance_masks pending rows ------------------------------------------------------

_MASK_KEY = re.compile(
    r"^labels/instance_masks/(?P<site>[^/]+)/masks/(?P<stem>.+)_mask_(?P<k>\d+)\.png$"
)


def image_index(keys: Iterable[str], prefix: str = SEAVIEW_LABELS_PREFIX) -> dict[str, str]:
    """``<partition>/<stem>`` -> image key for every ``images/<partition>/<file>`` key."""
    out: dict[str, str] = {}
    for key in keys:
        parts = key.removeprefix(prefix).split("/")
        if len(parts) == 3 and parts[0] == "images":
            out[f"{parts[1]}/{PurePosixPath(parts[2]).stem}"] = key
    return out


def pending_rows(
    keys: Iterable[str],
    *,
    images: Mapping[str, str] | None = None,
    source_id: str = SEAVIEW_LABELS_SOURCE,
    source_version: str = SEAVIEW_LABELS_VERSION,
    ann_license: str = "CC-BY-4.0",
    prefix: str = SEAVIEW_LABELS_PREFIX,
) -> list[dict]:
    """Pending instance rows (``mask_ref`` + ``image_key`` + TODO) for the per-instance Labelbox
    masks. ``label_native`` is a TODO string, ``match_type`` ``unmapped``: no class is guessed
    until the ``labels/exports/*.ndjson`` is parsed."""
    rows: list[dict] = []
    typ, detail = annotator_from_origin("human_expert")
    for key in sorted(keys):
        m = _MASK_KEY.match(key.removeprefix(prefix))
        if m is None:
            continue
        ref = f"{m['site']}/{m['stem']}"
        image_key = (images or {}).get(ref) or f"{prefix}images/{m['site']}/{m['stem']}"
        rows.append(
            {
                "source_id": source_id, "source_version": source_version,
                "ann_id": f"{source_id}:{len(rows)}", "label_native": PENDING_LABEL,
                "label_set": "reef-support-labelbox", "match_type": "unmapped",
                "annotator_type": typ, "annotator_detail": detail, "ann_license": ann_license,
                "label_status": "ok", "mask_kind": "instance", "mask_ref": key,
                "attrs": json.dumps({"licence_class": "open", "todo": "label from labels/exports/*.ndjson"}),  # noqa: E501
                "image_key": image_key, "image_sha256_todo": TODO_TEXT,
            }
        )  # fmt: skip
    return rows


def validate_pending(rows: Iterable[Mapping[str, object]]) -> list[str]:
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_COLUMNS}
        core["image_sha256"] = _PLACEHOLDER_SHA
        errs += [f"{row['ann_id']}: {e}" for e in validate_row("masks", core)]
        if not row.get("image_key"):
            errs.append(f"{row['ann_id']}: image_key missing")
    return errs


def write_pending_masks(path: Path, rows: list[dict]) -> int:
    """Pending rows: ``masks`` columns minus ``image_sha256`` plus :data:`PENDING_COLUMNS`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if errs := validate_pending(rows):
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema("masks")
    fields = [f for f in base if f.name != "image_sha256"] + [
        pa.field(c, pa.string()) for c in PENDING_COLUMNS
    ]  # noqa: E501, RUF100
    table = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in fields},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return len(rows)


__all__ = [  # noqa: RUF022
    "SEAVIEW_LABELS_PREFIX",
    "SEAVIEW_LABELS_SOURCE",
    "SEAVIEW_LABELS_VERSION",
    "MASK_SOURCES",
    "MaskSource",
    "image_index",
    "drop_internal_only",
    "is_internal_only",
    "pending_rows",
    "semantic_row",
    "staged_mask_rows",
    "validate_pending",
    "write_masks",
    "write_pending_masks",
]
