"""``boxes`` and ``instances`` from the staged trees of a ``--v11`` release build.

Part B of :mod:`marinedata.task_layers.staged_annotations`: each producer reads one admitted
staged tree through ``ctx.fetch`` (pinned key ``sources/<id>/<ver>/<rel>`` = ``<root>/<rel>``) and
writes its table with the table's own writer into the overlay. No image is read.

* ``boxes``: exactly the :data:`~marinedata.task_layers.boxes_table.BOX_SOURCES` that the v1.1.0
  locks pin (``releases/marine-data/v1.1.0/{open,nc}/release.lock.yaml``): ``roboflow-aquarium``
  (yolo), ``uiis`` / ``uiis10k`` / ``usis10k`` (coco-instances) and ``synthetic-seabed-debris``
  (synthetic-json; it is in the nc lock, so it runs in the nc build only). Nothing is registered
  for the other BOX_SOURCES (``fathomnet``, ``fathomnet-fgvc23``, ``fathomnet-fgvc25``,
  ``rf100-coral-lwptl``, ``ruod``, ``brackishmot``, ``obsea-fish``, ``sss-mine-detection``,
  ``swdd-sss-wall``): none is in either v1.1.0 lock, so no build admits their trees, and the
  FathomNet readers need a stream index and a licence join a local tree does not carry.
* ``instances``: ``INSTANCE_SOURCES`` (``usis10k``, ``uiis``, ``uiis10k``), written as ``masks``
  rows (``_annotations/masks/<src>``, which ``build_instance_config`` reads). ``mask_ref`` stays
  null; the v1.1.0 table build stamps ``mask_config = instance-masks``.

A source that yields no row raises, and the overlay turns that into ``StagedLayerError``: the
message carries the reader's counts, its first unparsable reasons and the keys it could not fetch.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict

from ..registry import Registry
from .boxes_table import BOX_SOURCES, licence_split, staged_boxes, write_boxes
from .masks_table import MASK_SOURCES, write_masks
from .producers.coralscapes_v2_instances import staged_masks_rows
from .s3_keyed import StagedTree
from .sources.masks_instance import INSTANCE_SOURCES, mapped_pct, staged_instance_rows
from .staged_annotations import LayerProducer, ProducedLayer, ProducerContext, register_producer

BOX_LAYER_SOURCES = (
    "roboflow-aquarium",
    "synthetic-seabed-debris",
    "uiis",
    "uiis10k",
    "usis10k",
)
INSTANCE_LAYER_SOURCES = tuple(sorted(INSTANCE_SOURCES))
_SHOWN = 3  # reasons / keys kept in a layer's detail


def _pinned(version: str) -> Callable[[Registry], str]:
    return lambda _registry: version


def _diagnosis(detail: dict) -> str:
    return json.dumps(detail, sort_keys=True, default=str)


def _run_boxes(ctx: ProducerContext) -> ProducedLayer:
    spec = BOX_SOURCES[ctx.source_id]
    tree = StagedTree(spec.tree, ctx.fetch)
    result = staged_boxes(spec, ctx.registry, tree=tree, fetch=ctx.fetch)
    detail = {
        "counts": asdict(result.counts),
        "unlabelled_images": result.unlabelled_images,
        "pending": len(result.pending),
        "unparsable": list(result.unparsable[:_SHOWN]),
        "missing": len(tree.missing),
        "missing_keys": [key for key, _status in tree.missing[:_SHOWN]],
        "licence_classes": licence_split(result.rows),
    }
    if not result.rows:
        raise ValueError(f"no box was read: {_diagnosis(detail)}")
    write_boxes(ctx.annotations_root, spec.source_id, spec.version, list(result.rows))
    return ProducedLayer(len(result.rows), result.images, detail)


def _run_instances(ctx: ProducerContext) -> ProducedLayer:
    spec = INSTANCE_SOURCES[ctx.source_id]
    tree = StagedTree(spec.tree, ctx.fetch)
    rows, tally = staged_instance_rows(tree, spec, ctx.registry)
    detail = {
        "config": "instances",
        "tally": dict(tally),
        "missing": len(tree.missing),
        "mapped_pct": mapped_pct(rows),
    }
    if not rows:
        raise ValueError(f"no instance was read: {_diagnosis(detail)}")
    write_masks(ctx.annotations_root, spec.source_id, spec.version, rows)
    return ProducedLayer(len(rows), len({r["image_sha256"] for r in rows}), detail)


CORALSCAPES_V2 = "coralscapes-v2"


def _run_coralscapes_v2(ctx: ProducerContext) -> ProducedLayer:
    """One masks parquet for V2: the semantic rows (read by ``semseg``) then the fish instance rows
    (read by ``instances``); ``mask_kind`` keeps each row in one config."""
    spec = MASK_SOURCES[CORALSCAPES_V2]
    tree = StagedTree(spec.tree, ctx.fetch)
    rows = list(staged_masks_rows(tree, ctx.registry))
    kinds: dict[str, int] = {}
    for row in rows:
        kinds[row["mask_kind"]] = kinds.get(row["mask_kind"], 0) + 1
    detail = {"config": "semseg+instances", "mask_kinds": kinds, "missing": len(tree.missing)}
    if not kinds.get("semantic"):
        raise ValueError(f"no coralscapes-v2 semantic mask was read: {_diagnosis(detail)}")
    write_masks(ctx.annotations_root, spec.source_id, spec.version, rows)
    return ProducedLayer(len(rows), len({r["image_sha256"] for r in rows}), detail)


register_producer(
    LayerProducer(
        "masks", CORALSCAPES_V2, _pinned(MASK_SOURCES[CORALSCAPES_V2].version), _run_coralscapes_v2
    )
)
for _source in BOX_LAYER_SOURCES:
    register_producer(
        LayerProducer("boxes", _source, _pinned(BOX_SOURCES[_source].version), _run_boxes)
    )
for _source in INSTANCE_LAYER_SOURCES:
    register_producer(
        LayerProducer("masks", _source, _pinned(INSTANCE_SOURCES[_source].version), _run_instances)
    )

__all__ = ["BOX_LAYER_SOURCES", "INSTANCE_LAYER_SOURCES"]
