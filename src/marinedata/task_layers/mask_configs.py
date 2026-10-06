"""MV-1: the self-contained HF mask configs (image + mask + class map in ONE row).

``coral-masks`` · ``coral-masks-machine`` · ``scene-masks`` (semantic) and ``instance-masks``
(instance) are built from the ``semseg`` / ``instances`` task-layer rows
(:mod:`marinedata.task_layers.configs`) through :data:`MASK_CONFIG_BY_SOURCE`. Unlike the
label-only task configs they embed the pixels themselves, so ``load_dataset(repo, config)`` needs
no join with ``images``. The flavour filter has already been applied to the rows by
:func:`marinedata.hf_export.build_layout`, which decides the repo.

Mask bytes are normalised lazily, one row group at a time, by :mod:`.mask_encode` and fetched
through ``fetch(key)``: a local staged tree first (:func:`make_fetch`), else the anonymous GET of
:func:`marinedata.task_layers.s3_keyed.fetch_small`. A row whose image is not in the release (no
split, no file) is dropped and counted, exactly like :func:`marinedata.task_layers.hf_wiring.layout_entries`.
"""  # noqa: E501

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..hf_parquet import ConfigSpec, EmbeddedBlob, ExportRow
from .configs import MASK_CONFIG_BY_SOURCE, MASK_EXPORT_CONFIG_IDS
from .mask_encode import compose_parts, encoding_for, instance_png, semantic_png

INSTANCE_CONFIG = "instance-masks"
SEMSEG_LAYER, INSTANCE_LAYER = "semseg", "instances"
MASK_BYTES_ESTIMATE = 64 * 1024
"""Shard-planning allowance for one mask PNG (its real size is only known once normalised)."""
SPLIT_ORDER = ("train", "validation", "test")
HUMAN_TYPES = frozenset({"human", "expert", "crowd"})

_COMMON = (
    ("image_sha256", "string"),
    ("image", "image"),
    ("mask", "image"),
)
_TAIL = (
    ("source", "string"),
    ("licence", "string"),
    ("annotation_licence", "string"),
    ("attribution", "string"),
    ("split", "string"),
    ("split_group", "string"),
    ("annotator_type", "string"),
    ("width", "int64"),
    ("height", "int64"),
    ("lat", "double"),
    ("lon", "double"),
)
SEMANTIC_COLUMNS = (*_COMMON, ("class_map", "class_map"), *_TAIL)
INSTANCE_COLUMNS = (*_COMMON, ("instances", "instances"), *_TAIL)
DESCRIPTIONS = {
    "semantic": "image + uint8 index mask (255 = ignore) + class_map, one row per image.",
    "instance": "image + uint16 instance-id mask (0 = background) + instances, one row per image.",
}


@dataclass(frozen=True)
class ImageInfo:
    """What the release knows about one image: its file, HF split and group."""

    path: Path
    split: str
    split_group: str | None = None
    licence: str | None = None


Fetch = Callable[[str], bytes]


def make_fetch(mask_root: Path | None = None) -> Fetch:
    """``key -> bytes``: ``<mask_root>/<key>`` when that file exists, else the anonymous GET."""

    def fetch(key: str) -> bytes:
        if mask_root is not None and (local := Path(mask_root) / key).is_file():
            return local.read_bytes()
        from .s3_keyed import fetch_small

        return fetch_small(key)

    return fetch


def config_kind(config: str) -> str:
    return "instance" if config == INSTANCE_CONFIG else "semantic"


def annotator_kind(annotator_type: str | None) -> str | None:
    """``human`` for expert / crowd / human labels, ``machine`` for pseudo / model ones."""
    if annotator_type is None:
        return None
    return "human" if annotator_type in HUMAN_TYPES else "machine"


def _size(path: Path) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _attrs(row: Mapping) -> dict:
    raw = row.get("attrs")
    return json.loads(raw) if isinstance(raw, str) else dict(raw or {})


def _bbox_norm(row: Mapping, size: tuple[int, int]) -> list[float] | None:
    box = _attrs(row).get("bbox_xywh")
    if not box or not size[0] or not size[1]:
        return None
    x, y, w, h = (float(v) for v in box)
    clip = lambda v: min(max(v, 0.0), 1.0)  # noqa: E731
    return [clip(x / size[0]), clip(y / size[1]), clip((x + w) / size[0]), clip((y + h) / size[1])]


def _common_values(row, info, sha, source_meta, geo, size) -> dict:
    meta = source_meta.get(row["source_id"]) or {}
    lat, lon = (geo or {}).get(sha, (None, None))
    return {
        "image_sha256": sha,
        "source": row["source_id"],
        "licence": meta.get("licence") or info.licence,
        "annotation_licence": row.get("ann_license"),
        "attribution": meta.get("attribution"),
        "split": info.split,
        "split_group": info.split_group,
        "annotator_type": annotator_kind(row.get("annotator_type")),
        "width": size[0],
        "height": size[1],
        "lat": lat,
        "lon": lon,
    }


def _blobs(info: ImageInfo, sha: str, mask_name: str, make_mask: Callable[[], bytes]):
    return (
        EmbeddedBlob(
            "image", f"{sha}{info.path.suffix}", info.path.stat().st_size, info.path.read_bytes
        ),
        EmbeddedBlob("mask", mask_name, MASK_BYTES_ESTIMATE, make_mask),
    )


def semantic_row(
    rec: Mapping, info: ImageInfo, fetch: Fetch, source_meta: Mapping, geo: Mapping | None
) -> ExportRow | None:
    """One semantic export row, or ``None`` when the record has no class map / mask to read."""
    sha = rec["image_sha256"]
    class_map = [c for c in json.loads(rec["hf_class_map"] or "[]") if c["id"] != 255]
    parts = json.loads(rec["mask_parts"]) if rec.get("mask_parts") else None
    if not class_map or not (rec.get("mask_key") or parts):
        return None
    size = _size(info.path)
    ignore = rec.get("ignore_value")
    encoding = encoding_for(rec["source_id"], rec.get("mask_encoding"))

    def make() -> bytes:
        if parts:
            return compose_parts(
                [(fetch(p["key"]), int(p["class_id"])) for p in parts], size, ignore_value=ignore
            )
        return semantic_png(
            fetch(rec["mask_key"]), encoding=encoding, ignore_value=ignore, size=size
        )

    values = _common_values(rec, info, sha, source_meta, geo, size)
    values["class_map"] = class_map
    name = f"{rec['source_id']}/{sha}.png"
    return ExportRow(values=values, blobs=_blobs(info, sha, name, make))


def instance_row(
    recs: Sequence[Mapping],
    info: ImageInfo,
    fetch: Fetch,
    source_meta: Mapping,
    geo: Mapping | None,
) -> ExportRow:
    """One instance export row for all instance records of one image (ids 1..N by upstream id)."""
    recs = sorted(recs, key=lambda r: (_int(r.get("instance_id")), r["ann_id"]))
    sha = recs[0]["image_sha256"]
    size = _size(info.path)
    attrs = _attrs(recs[0])
    canvas = (int(attrs["img_w"]), int(attrs["img_h"])) if attrs.get("img_w") else size
    ids = list(range(1, len(recs) + 1))
    instances = [
        {
            "id": i,
            "label_native": r.get("label_native"),
            "taxon_node": r.get("taxon_node_id") if r.get("match_type") != "unmapped" else None,
            "coarse": r.get("coarse") if r.get("match_type") != "unmapped" else None,
            "bbox_xyxy_norm": _bbox_norm(r, canvas),
        }
        for i, r in zip(ids, recs, strict=True)
    ]

    def make() -> bytes:
        return instance_png(recs, ids, canvas=canvas, size=size, fetch=fetch)

    values = _common_values(recs[0], info, sha, source_meta, geo, size)
    values["instances"] = instances
    return ExportRow(
        values=values, blobs=_blobs(info, sha, f"{recs[0]['source_id']}/{sha}.png", make)
    )


def _int(value: object) -> int:
    return int(value) if value is not None else 2**31


def mask_layout_entries(
    task_layers: Mapping[str, Sequence[Mapping]],
    images: Mapping[str, ImageInfo],
    *,
    fetch: Fetch | None = None,
    source_meta: Mapping[str, Mapping[str, str | None]] | None = None,
    geo_by_sha: Mapping[str, tuple[float | None, float | None]] | None = None,
    stats: Counter | None = None,
) -> dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]]:
    """``{config: (spec, {hf_split: rows})}`` for the four mask configs that have rows.

    ``images`` is ``{image_sha256: ImageInfo}`` for every image of the release; ``stats`` (a
    Counter) tallies ``dropped_no_image`` / ``dropped_no_class_map`` per config."""
    fetch = fetch or make_fetch()
    meta = source_meta or {}
    tally = stats if stats is not None else Counter()
    rows: dict[str, list[tuple[str, ExportRow]]] = defaultdict(list)

    def place(config: str, sha: str, build: Callable[[ImageInfo], ExportRow | None]) -> None:
        info = images.get(sha)
        if info is None:
            tally[f"{config}:dropped_no_image"] += 1
            return
        row = build(info)
        if row is None:
            tally[f"{config}:dropped_no_class_map"] += 1
            return
        rows[config].append((info.split, row))

    for rec in task_layers.get(SEMSEG_LAYER, ()):
        config = MASK_CONFIG_BY_SOURCE.get(rec.get("source_id"))
        if config is None or config_kind(config) != "semantic":
            continue
        if rec.get("mask_kind") not in (None, "semantic"):
            continue
        sha = rec.get("image_sha256") or rec.get("sha256")
        place(config, sha, lambda i, r={**rec, "image_sha256": sha}: semantic_row(
            r, i, fetch, meta, geo_by_sha))  # fmt: skip
    per_image: dict[tuple[str, str], list[Mapping]] = defaultdict(list)
    for rec in task_layers.get(INSTANCE_LAYER, ()):
        if MASK_CONFIG_BY_SOURCE.get(rec.get("source_id")) == INSTANCE_CONFIG:
            per_image[(rec["image_sha256"], rec["source_id"])].append(rec)
    for (sha, _source), recs in sorted(per_image.items()):
        place(INSTANCE_CONFIG, sha, lambda i, rs=recs: instance_row(rs, i, fetch, meta, geo_by_sha))

    out: dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]] = {}
    for config in MASK_EXPORT_CONFIG_IDS:
        if not rows.get(config):
            continue
        kind = config_kind(config)
        columns = INSTANCE_COLUMNS if kind == "instance" else SEMANTIC_COLUMNS
        grouped: dict[str, list[ExportRow]] = defaultdict(list)
        for split, row in sorted(
            rows[config], key=lambda sr: (sr[1].values["image_sha256"], sr[1].values["source"])
        ):
            grouped[split].append(row)
        ordered = {s: grouped[s] for s in (*SPLIT_ORDER, *sorted(grouped)) if s in grouped}
        out[config] = (ConfigSpec(config, columns, DESCRIPTIONS[kind]), ordered)
    return out


def mask_source_meta(registry) -> dict[str, dict[str, str | None]]:
    """``{source_id: {licence, attribution}}`` for every source in the mask table the registry
    knows (a source MV-2 has not registered yet is skipped)."""
    from ..hf_card import _attribution

    out: dict[str, dict[str, str | None]] = {}
    for source_id in MASK_CONFIG_BY_SOURCE:
        try:
            source = registry.source(source_id)
        except Exception:
            continue
        out[source_id] = {"licence": source.licence.id, "attribution": _attribution(source)}
    return out
