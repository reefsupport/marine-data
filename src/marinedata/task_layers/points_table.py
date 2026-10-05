"""Benthic points into the unified ``points`` table (WP-U3).

One row per labelled point in :mod:`marinedata.annotation_schema`'s ``points`` table:
``x``/``y`` normalised to [0, 1] by the image's own width/height (``x_px``/``y_px`` keep the
source pixels), the native label byte-exact, and the taxon columns from
:meth:`marinedata.schema.Crosswalk.resolve`. Producers:

* :func:`staged_points_rows`: the staged ``labels/points.parquet`` of mermaid-aws and reefolution
  (``image_sha256`` from the tree's own ``CHECKSUMS.sha256``);
* :func:`seaview_rows` / :func:`ibf_rows`: ``benthic_datasets/point_labels/`` (SEAVIEW region CSVs,
  IBF CPCe files). Those images are not under ``sources/`` so no sha256 exists: rows carry
  ``image_key`` and ``image_sha256_todo`` (:func:`write_pending_points`) until the images are staged.
"""  # noqa: E501

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

from ..annotation_schema import (
    annotation_path,
    annotator_from_origin,
    normalise_split,
    table_spec,
    validate_row,
    write_annotations,
)
from ..registry import Registry
from ..taxonomy import L2_TASK

Resolver = Callable[[str], tuple[str | None, str, int | None, str | None]]
LABEL_SET = "dataset-native"
PENDING_COLUMNS = ("image_key", "image_sha256_todo")
TODO_TEXT = "TODO: hash image_key once the image is staged under sources/ (not resolvable today)"
_PLACEHOLDER_SHA = "0" * 64


def resolver_for(registry: Registry, crosswalk_id: str) -> Resolver:
    """Memoised ``native -> (taxon_node_id, match_type, worms_aphia_id, rs_benthic_code)``."""
    walk = registry.crosswalk(crosswalk_id)
    target = registry.label_schema(walk.target_schema)
    l2 = (
        frozenset(registry.task(L2_TASK).classes)
        if any(t.id == L2_TASK for t in registry.tasks)
        else frozenset()
    )
    cache: dict[str, tuple[str | None, str, int | None, str | None]] = {}

    def resolve(native: str) -> tuple[str | None, str, int | None, str | None]:
        if native not in cache:
            cache[native] = walk.resolve(native, target=target, l2_codes=l2)
        return cache[native]

    return resolve


def source_licence(registry: Registry, source_id: str) -> str | None:
    lic = registry.source(source_id).licence
    return None if lic is None else str(getattr(lic, "id", lic))


def _taxon_rank(registry: Registry, taxon: str | None) -> str | None:
    if taxon is None:
        return None
    node = registry.label_schema("rs-benthic-v1").node(taxon)
    return node.worms_rank if node is not None else None


def point_row(
    *,
    source_id: str,
    source_version: str,
    ordinal: int,
    image_sha256: str | None,
    native: str,
    resolve: Resolver,
    registry: Registry,
    x_px: float,
    y_px: float,
    width: int,
    height: int,
    annotator: str,
    ann_license: str | None,
    split: str | None = None,
    confidence: float | None = None,
    attrs: Mapping[str, object] | None = None,
    label_native_id: str | None = None,
    label_set: str = LABEL_SET,
) -> dict:
    """One ``points`` row. ``x_px``/``y_px`` are clamped into the image before normalising."""
    taxon, match_type, aphia, l2 = resolve(native)
    xi, yi = min(max(round(x_px), 0), width), min(max(round(y_px), 0), height)
    typ, detail = annotator_from_origin(annotator)
    return {
        "image_sha256": image_sha256,
        "source_id": source_id,
        "source_version": source_version,
        "ann_id": f"{source_id}:{ordinal}",
        "label_native": native,
        "label_native_id": label_native_id,
        "label_set": label_set,
        "taxon_node_id": taxon,
        "taxon_rank": _taxon_rank(registry, taxon),
        "worms_aphia_id": aphia,
        "rs_benthic_code": l2,
        "match_type": match_type,
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": ann_license,
        "confidence": confidence,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(dict(attrs), sort_keys=True) if attrs else None,
        "x": xi / width,
        "y": yi / height,
        "x_px": xi,
        "y_px": yi,
    }


def staged_points_rows(
    tree,
    registry: Registry,
    source_id: str,
    source_version: str,
    crosswalk_id: str,
    *,
    limit_images: int | None = None,
) -> list[dict]:
    """mermaid-aws / reefolution: staged ``labels/points.parquet`` -> ``points`` rows.

    ``tree`` is an :class:`~marinedata.task_layers.s3_keyed.StagedTree` (or a stand-in with
    ``table(rel)`` and ``shas``). A point whose image is not in the tree's checksums raises.
    """
    resolve = resolver_for(registry, crosswalk_id)
    lic = source_licence(registry, source_id)
    meta = {(r["partition"], r["stem"]): r for r in tree.table("metadata.parquet")}
    keep: set[tuple[str, str]] | None = None
    if limit_images is not None:
        keys = sorted(meta)
        step = max(len(keys) // limit_images, 1)
        keep = set(keys[::step][:limit_images])
    rows: list[dict] = []
    for p in tree.table("labels/points.parquet"):
        key = (p["partition"], p["stem"])
        if keep is not None and key not in keep:
            continue
        image, sha = meta.get(key), tree.shas.get(key)
        if image is None or sha is None:
            raise ValueError(f"{source_id}: point references unstaged image {key!r}")
        extra = {k: p[k] for k in ("form", "region") if p.get(k) is not None}
        rows.append(
            point_row(
                source_id=source_id,
                source_version=source_version,
                ordinal=len(rows),
                image_sha256=sha,
                native=p["label"],
                resolve=resolve,
                registry=registry,
                x_px=p["col"],
                y_px=p["row"],
                width=image["width"],
                height=image["height"],
                annotator="human_expert",
                ann_license=lic,
                split=p.get("partition"),
                attrs=extra,
                label_native_id=None if p.get("label_id") is None else str(p["label_id"]),
                label_set=p.get("schema_id") or LABEL_SET,
            )
        )
    return rows


ImageInfo = Callable[[str], "tuple[str, int, int] | None"]
"""``quadrat id / image name -> (image_key, width, height)``; None when the image is not in the bucket."""  # noqa: E501


def seaview_rows(
    csv_text: str,
    region: str,
    registry: Registry,
    image_info: ImageInfo,
    *,
    source_version: str,
    start: int = 0,
    only: Iterable[str] | None = None,
) -> list[dict]:
    """SEAVIEW ``annotations_<REGION>.csv`` (``quadratid,y,x,label_name,label,func_group,method,data_set``).

    ``x``/``y`` are pixels in the quadrat image. Rows for images ``image_info`` cannot find are
    skipped (the caller counts them). The result is *pending* (``image_key``, no sha256)."""  # noqa: E501
    source_id = "seaview-survey-imagery"
    resolve = resolver_for(registry, "seaview-point-labels")
    lic = source_licence(registry, source_id)
    wanted = None if only is None else set(only)
    rows: list[dict] = []
    for r in csv.DictReader(io.StringIO(csv_text)):
        qid = r["quadratid"]
        if wanted is not None and qid not in wanted:
            continue
        info = image_info(qid)
        if info is None:
            continue
        key, w, h = info
        row = point_row(
            source_id=source_id,
            source_version=source_version,
            ordinal=start + len(rows),
            image_sha256=None,
            native=r["label"],
            resolve=resolve,
            registry=registry,
            x_px=float(r["x"]),
            y_px=float(r["y"]),
            width=w,
            height=h,
            annotator="human_expert",
            ann_license=lic,
            split=r.get("data_set"),
            attrs={
                "func_group": r["func_group"],
                "label_name": r["label_name"],
                "method": r["method"],
                "region": region,
            },
        )
        rows.append({**row, "image_key": key, "image_sha256_todo": TODO_TEXT})
    return rows


def parse_cpc(text: str) -> tuple[str, int, int, list[tuple[float, float, str]]]:
    """``(image basename, width_px, height_px, [(x_px, y_px, code)])`` of one CPCe ``.cpc`` file.

    Line 1 is ``"codefile","imagepath",width_twips,height_twips,...``; 4 frame lines; the point
    count; then N ``x,y`` twip lines and N ``"n","CODE","Notes",""`` label lines. 15 twips = 1 px."""  # noqa: E501
    lines = text.splitlines()
    head = next(csv.reader([lines[0]]))
    name = head[1].replace("\\", "/").rsplit("/", 1)[-1]
    w, h = int(head[2]) // 15, int(head[3]) // 15
    n = int(lines[5])
    pts = [tuple(float(v) / 15 for v in ln.split(",")) for ln in lines[6 : 6 + n]]
    codes = [next(csv.reader([ln]))[1] for ln in lines[6 + n : 6 + 2 * n]]
    return name, w, h, [(x, y, c) for (x, y), c in zip(pts, codes, strict=True)]


def ibf_rows(
    cpc_files: Mapping[str, str],
    registry: Registry,
    *,
    source_version: str,
    image_keys: Mapping[str, str],
) -> list[dict]:
    """IBF: ``{cpc key: text}`` -> pending ``points`` rows. ``image_keys`` maps
    ``<folder>/<image basename>`` to the bucket key (a missing image gives ``image_key`` = the
    expected sibling path, flagged in ``attrs``)."""
    source_id = "ibf"
    resolve = resolver_for(registry, "ibf-cpce-codes")
    lic = source_licence(registry, source_id)
    rows: list[dict] = []
    for cpc_key in sorted(cpc_files):
        name, w, h, pts = parse_cpc(cpc_files[cpc_key])
        folder = str(PurePosixPath(cpc_key).parent)
        key = image_keys.get(f"{folder}/{name}") or f"{folder}/{name}"
        for x, y, code in pts:
            row = point_row(
                source_id=source_id,
                source_version=source_version,
                ordinal=len(rows),
                image_sha256=None,
                native=code,
                resolve=resolve,
                registry=registry,
                x_px=x,
                y_px=y,
                width=w,
                height=h,
                annotator="human",
                ann_license=lic,
                attrs={"cpc": PurePosixPath(cpc_key).name},
            )
            rows.append({**row, "image_key": key, "image_sha256_todo": TODO_TEXT})
    return rows


def write_points(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    """Write resolved rows to ``<data_dir>/_annotations/points/<source_id>/<version>.parquet``."""
    path = annotation_path(data_dir, "points", source_id, source_version)
    write_annotations(path, "points", rows)
    return path


def validate_pending(rows: Iterable[Mapping[str, object]]) -> list[str]:
    """U1 validation of pending rows: everything but the missing sha256 (a placeholder stands in)."""  # noqa: E501
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_COLUMNS}
        core["image_sha256"] = _PLACEHOLDER_SHA
        errs += [f"{row['ann_id']}: {e}" for e in validate_row("points", core)]
        if not row.get("image_key"):
            errs.append(f"{row['ann_id']}: image_key missing")
    return errs


def write_pending_points(path: Path, rows: list[dict]) -> int:
    """Write pending rows (points columns minus ``image_sha256`` plus :data:`PENDING_COLUMNS`)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from ..annotation_schema import arrow_schema

    errs = validate_pending(rows)
    if errs:
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema("points")
    fields = [f for f in base if f.name != "image_sha256"] + [
        pa.field(c, pa.string()) for c in PENDING_COLUMNS
    ]
    names = [f.name for f in fields]
    table = pa.table(
        {n: [r.get(n) for r in rows] for n in names},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return len(rows)


__all__ = [  # noqa: RUF022
    "table_spec",
    "ibf_rows",
    "parse_cpc",
    "point_row",
    "seaview_rows",
    "staged_points_rows",
    "resolver_for",
    "validate_pending",
    "write_pending_points",
    "write_points",
]
