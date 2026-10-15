"""Task-layer configs: constants (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

from ..boxes_table import BOX_SOURCES
from ..depth_table import DEPTH_PAIR_SOURCES
from ..masks_table import MASK_SOURCES
from ..sources.masks_instance import INSTANCE_SOURCES
from ..vqa_table import VQA_SOURCES

BENTHIC_COARSE_TASK = "benthic-coarse"


BLEACHING_TASK = "bleaching-condition"


HEALTH_TASK = "coral-health-binary"


CONFIG_IDS = (
    "points", "vqa", "semseg", "benthic-coarse", "benthic-cover", "bleaching", "boxes", "captions",
    "depth", "pairs", "instances",
)  # fmt: skip


# WP-8e-resume (manager decision 3): the staged point/mask sources each config reads.
POINT_SOURCES = ("reefolution", "mermaid-aws", "seaview-survey-imagery")


# The benthic rollups (cover, coarse) keep reading the two original point sources: Seaview points
# feed the ``points`` config only until a cover decision is made (PA-P3.1 report, Open).
ROLLUP_POINT_SOURCES = ("reefolution", "mermaid-aws")


# WP-U4: registry-driven. The semseg config reads every source with a unified ``masks`` producer
# (``masks_table.MASK_SOURCES``); the benthic rollups stay on the two benthic-cover sources (a
# pseudo-label, a 3-class and a scene-segmentation source must not feed cover).
SEMSEG_SOURCES = tuple(MASK_SOURCES)


ROLLUP_MASK_SOURCES = ("coralscapes", "reef-support-benthic-own")


# WP-U6a: registry-driven like masks: every source with a unified ``boxes`` producer.
BOX_CONFIG_SOURCES = tuple(BOX_SOURCES)


# WP-U10: registry-driven too: every source with a unified ``vqa`` producer; coralvqa keeps its
# D-Z2 ``_tasklabels`` payload as the fallback while no unified file has been written.
VQA_CONFIG_SOURCES = tuple(VQA_SOURCES)


LEGACY_VQA_SOURCES = ("coralvqa",)


# WP-U11: registry-driven: every source bound to a depth / pairs producer feeds its config.
DEPTH_CONFIG_SOURCES = tuple(s for s, v in DEPTH_PAIR_SOURCES.items() if "depth" in v.tables)


PAIRS_CONFIG_SOURCES = tuple(s for s, v in DEPTH_PAIR_SOURCES.items() if "pairs" in v.tables)


# WP-U10b: registry-driven like the U4 masks: every source with a unified *instance* ``masks``
# producer (``sources.masks_instance.INSTANCE_SOURCES``: usis10k, uiis, uiis10k). Disjoint from
# SEMSEG_SOURCES, so no mask row is counted by both configs.
INSTANCE_CONFIG_SOURCES = tuple(INSTANCE_SOURCES)


# MV-1 (masks v1): self-contained HF mask configs (image + mask + class_map in one row); the
# flavour still decides the repo; a source with no producer yet (MV-2) simply yields no rows.
MASK_EXPORT_CONFIG_IDS = ("coral-masks", "coral-masks-machine", "scene-masks", "instance-masks")


_CORAL = ("coralscapes", "coralseg-ucsd-mosaics", "reef-support-seaview-labels",
          "reef-support-benthic-own", "deolhonoscorais")  # fmt: skip


MASK_CONFIG_BY_SOURCE: dict[str, str] = {
    **dict.fromkeys(_CORAL, "coral-masks"), "coralscop-masks-rs": "coral-masks-machine",
    "suim": "scene-masks", **dict.fromkeys(("uiis", "uiis10k", "usis10k"), "instance-masks"),
}  # fmt: skip


# WP-8e-resume (manager decision 2): bleaching is a headline reef task with its own config.
BLEACHING_SOURCES = (
    "noaa-pifsc-bleaching",
    "roboflow-coral-bleaching-final-v6i",
    "roboflow-coral-bleaching-general-v1-yolov8s",
    "roboflow-coral-classification-copy-changed-v13i",
    "roboflow-coral-reef-bleach-detection-v2i",
    "roboflow-coral-reef-classification-v3i",
    "reef-support-bleaching",
)
