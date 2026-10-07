"""Boxes from the Lian/Li COCO instance sets (``uiis``, ``uiis10k``, ``usis10k``; fish-boxes).

The sets ship instance masks with a COCO ``bbox`` on every annotation, so each instance is one box.
Documents, split names and the staged-stem rule are the ones of the ``masks`` producer
(:data:`sources.masks_instance.INSTANCE_SOURCES`); USIS10K's class-agnostic ``foreground``
document is not read (same instances, no class). :func:`staged_boxes` dispatches here for
``coco-instances``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from .boxes_table import BoxResult, BoxSource, _Emit
from .image_labels_table import _stride
from .s3_keyed import StagedTree
from .sources.boxes_coco import read_coco
from .sources.boxes_common import BoxCounts
from .sources.masks_instance import INSTANCE_SOURCES, staged_stem


def staged_instances(spec: BoxSource, registry, tree: StagedTree, limit: int | None) -> BoxResult:
    inst = INSTANCE_SOURCES[spec.source_id]
    out = _Emit(spec, registry)
    docs = [(split, json.loads(tree.get(rel))) for rel, split in inst.docs]
    pairs = sorted(
        ((sp, im) for sp, d in docs for im in d["images"]),
        key=lambda p: (p[0], str(p[1].get("file_name"))),
    )
    chosen = {(sp, im["id"]) for sp, im in _stride(pairs, limit)}
    for split, doc in docs:
        ids = {i for s, i in chosen if s == split}
        part: Mapping = {  # only the chosen images, so the counts are theirs
            **doc,
            "images": [im for im in doc["images"] if im["id"] in ids],
            "annotations": [a for a in doc["annotations"] if a.get("image_id") in ids],
        }
        images, counts = read_coco(part)
        out.counts = out.counts + counts
        for img in images:
            sha = next(
                (
                    tree.shas[("default", s)]
                    for s in staged_stem(inst, split, img.file_name or "")
                    if ("default", s) in tree.shas
                ),
                None,
            )
            if sha is None:
                out.counts = out.counts + BoxCounts(orphan=1)
                continue
            out.image(sha, img.boxes, BoxCounts(), split=split, attrs={"modality": "optical"})
    return out.result()
