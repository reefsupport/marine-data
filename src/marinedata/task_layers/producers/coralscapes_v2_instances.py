"""Coralscapes V2 instance producer: cityscapes ``_instanceIds`` PNGs into ``masks`` rows.

The ``instances`` config of josauder/coralscapesV2 adds one uint16 PNG per frame, staged as
``labels/instances/default/<stem>_instanceIds.png``. ``extended_metadata.json`` at the pinned
revision fixes the encoding: ``stuff_pixels`` hold the semantic class id (always below the divisor)
and ``thing_pixels`` hold ``class_id * label_divisor + instance_index``, with ``label_divisor`` 2000
and ``fish_class_id`` 25. A pixel ``v >= 2000`` is instance ``v % 2000`` of class ``v // 2000``.
The ids pass through unchanged as uint16 (D4): the staged PNG is the export.

One ``mask_kind = instance`` row per (frame, thing id), the shape of ``sources.masks_instance``:
``instance_id`` is the pixel value (a uint16, so it fits the int32 column), ``mask_ref`` is the key
of the PNG, ``label_native`` is the class name, resolved through the ``coralscapes-95`` crosswalk
from the registry at run time. ``attrs`` keeps the class, the index, the divisor, the pixel area and
the image size. The rows share ``_annotations/masks/coralscapes-v2/<version>.parquet`` with the
semantic rows of :func:`~.coralscapes_v2_semseg.mask_source`; ``mask_kind`` tells them apart.
"""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from ...annotation_schema import annotator_from_origin, normalise_split
from ...licence_class import source_class
from ...registry import Registry
from ..masks_table import _splits, staged_mask_rows
from ..points_table import _taxon_rank, resolver_for
from ..s3_keyed import StagedTree
from .coralscapes_v2_semseg import (
    CROSSWALK_ID,
    ID_TO_LABEL,
    LABEL_SET,
    SOURCE_ID,
    SOURCE_VERSION,
    mask_source,
)

Resolver = Callable[[str], tuple[str | None, str, int | None, str | None]]

LABEL_DIVISOR = 2000  # extended_metadata.json: instances.label_divisor
FISH_CLASS_ID = 25  # extended_metadata.json: instances.fish_class_id
UINT16_MAX = 0xFFFF
INSTANCE_DIR = "instances"  # labels/instances/[<partition>/]<stem>_instanceIds.png
INSTANCE_SUFFIX = "_instanceIds"
IMAGE_SUFFIX = "_leftImg8bit"
ENCODING = "png-uint16-cityscapes"
MODALITY = "optical"


def decode_instance_id(value: int) -> tuple[int, int]:
    """``(class id, instance index)`` of a pixel value: ``value = class * 2000 + index``."""
    class_id, index = divmod(int(value), LABEL_DIVISOR)
    return class_id, index


def encode_instance_id(class_id: int, index: int) -> int:
    """The uint16 pixel value of instance ``index`` of ``class_id`` (inverse of the decode)."""
    value = class_id * LABEL_DIVISOR + index
    if class_id < 1 or not 0 <= index < LABEL_DIVISOR or value > UINT16_MAX:
        raise ValueError(f"class {class_id} instance {index} does not fit a uint16 id")
    return value


def is_thing(value: int) -> bool:
    """Thing pixels start at the divisor; below it a pixel is a stuff class id, not an instance."""
    return int(value) >= LABEL_DIVISOR


def decode_instance_png(data: bytes) -> tuple[dict[int, int], int, int]:
    """``({thing pixel value: pixel count}, height, width)`` of one ``_instanceIds`` PNG."""
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:  # an RGB export keeps the id in the first channel
        arr = arr[..., 0]
    values, counts = np.unique(arr, return_counts=True)
    things = {int(v): int(c) for v, c in zip(values, counts, strict=True) if is_thing(int(v))}
    return things, int(arr.shape[0]), int(arr.shape[1])


def instance_items(tree: StagedTree) -> list[tuple[tuple[str, str], str]]:
    """``((partition, image stem), rel)`` for each staged ``_instanceIds`` PNG.

    The image stem (``<stem>_leftImg8bit``) is the key :attr:`StagedTree.shas` holds for the frame.
    """
    out = []
    for rel in sorted(tree.checksums):
        path = PurePosixPath(rel)
        parts = path.parts
        if parts[:2] != ("labels", INSTANCE_DIR) or path.suffix != ".png":
            continue
        if not path.stem.endswith(INSTANCE_SUFFIX):
            continue
        partition = parts[2] if len(parts) == 4 else "default"
        out.append(((partition, path.stem.removesuffix(INSTANCE_SUFFIX) + IMAGE_SUFFIX), rel))
    return out


def instance_row(
    *,
    resolve: Resolver,
    registry: Registry,
    ordinal: int,
    image_sha256: str,
    mask_ref: str,
    value: int,
    area: int,
    split: str | None,
    img_h: int,
    img_w: int,
) -> dict | None:
    """One instance ``masks`` row, or ``None`` for a stuff value or an unknown class id."""
    class_id, index = decode_instance_id(value)
    native = ID_TO_LABEL.get(class_id)
    if not is_thing(value) or native is None:
        return None
    taxon, match, aphia, l2 = resolve(native)
    typ, detail = annotator_from_origin("human")
    attrs = {
        "mask_encoding": ENCODING,
        "pixel_value": value,
        "class_id": class_id,
        "instance_index": index,
        "label_divisor": LABEL_DIVISOR,
        "area": area,
        "img_w": img_w,
        "img_h": img_h,
        "licence_class": source_class(SOURCE_ID, "open"),
        "modality": MODALITY,
    }
    return {
        "image_sha256": image_sha256,
        "source_id": SOURCE_ID,
        "source_version": SOURCE_VERSION,
        "ann_id": f"{SOURCE_ID}:{ordinal}",
        "instance_id": value,
        "label_native": native,
        "label_native_id": str(class_id),
        "label_set": LABEL_SET,
        "taxon_node_id": taxon,
        "form_node_id": None,
        "condition_node_id": None,
        "taxon_rank": _taxon_rank(registry, taxon),
        "worms_aphia_id": aphia,
        "rs_benthic_code": l2,
        "match_type": match,
        "annotator_type": typ,
        "annotator_detail": detail,
        "ann_license": "Apache-2.0",
        "confidence": None,
        "upstream_split": normalise_split(split),
        "label_status": "ok",
        "attrs": json.dumps(attrs, sort_keys=True),
        "mask_kind": "instance",
        "mask_ref": mask_ref,
        "polygon": None,
        "rle": None,
        "class_map": None,
        "ignore_value": None,
        "class_counts": None,
        "canonical_class_counts": None,
    }


def staged_instance_rows(
    tree: StagedTree,
    registry: Registry,
    *,
    limit: int | None = None,
    start: int = 0,
    workers: int = 16,
) -> Iterator[dict]:
    """Instance rows for the staged ``_instanceIds`` PNGs (first ``limit`` files in key order)."""
    resolve = resolver_for(registry, CROSSWALK_ID)
    items = [(key, rel) for key, rel in instance_items(tree) if key in tree.shas]
    if limit is not None:
        items = items[:limit]
    splits = _splits(tree)

    def decode(item: tuple[tuple[str, str], str]) -> tuple[dict[int, int], int, int] | None:
        data = tree.get_or_skip(item[1])
        return None if data is None else decode_instance_png(data)

    ordinal = start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (key, rel), got in zip(items, pool.map(decode, items), strict=True):
            if got is None:
                continue
            things, height, width = got
            for value in sorted(things):
                row = instance_row(
                    resolve=resolve, registry=registry, ordinal=ordinal,
                    image_sha256=tree.shas[key], mask_ref=tree.key(rel), value=value,
                    area=things[value], split=splits.get(key), img_h=height, img_w=width,
                )  # fmt: skip
                if row is not None:
                    ordinal += 1
                    yield row


def staged_masks_rows(
    tree: StagedTree, registry: Registry, *, limit: int | None = None
) -> Iterator[dict]:
    """The whole V2 ``masks`` layer: the semantic rows, then the instance rows.

    ``ann_id`` is ``<source_id>:<ordinal>``, unique per file, so the instance ordinals
    continue after the semantic ones."""
    semantic = list(staged_mask_rows(tree, mask_source(), registry, limit=limit))
    yield from semantic
    yield from staged_instance_rows(tree, registry, limit=limit, start=len(semantic))
