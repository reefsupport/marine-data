"""The 5 v2 task-layer configs (WP-8c) — ``points``, ``vqa``, ``semseg``,
``benthic-coarse`` and ``benthic-cover``.

Each builder consumes whatever ``data/_tasklabels/<source_id>/<task>.parquet`` files
exist (D-Z2 — written by :mod:`marinedata.task_layers.producers` for the sources
available today, or by WP-8d's ``marinedata.task_layers.sources`` for others), applies
the WP-7 crosswalk (:class:`marinedata.harmonize.Harmonizer`) native -> canonical, and
for the two benthic configs projects canonical -> the ``benthic-coarse`` task's 6-class
vocabulary (:class:`marinedata.task.TaskProjector`) before the D-Y rollup
(:mod:`marinedata.task_layers.rollup`).

An unmapped native label (no crosswalk edge, or a crosswalk edge marked unmappable) never
becomes a guess — it is excluded from the row's known count (``unmapped_report`` tallies
how much was lost, per source), exactly the behaviour :mod:`marinedata.harmonize` and
:mod:`marinedata.task_layers.rollup` already enforce independently; this module only
wires the two together, and joins ``label_status`` (D-U) when present.
"""

from __future__ import annotations

from .assemble import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _build_all_configs,
    apply_flavour,
    assert_no_mixed_origin_in_eval,
    build_all_configs,
    write_configs,
)
from .benthic import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _build_benthic_rollup,
    _coarse_counts_from_masks,
    _coarse_counts_from_points,
    build_benthic_coarse_config,
    build_benthic_cover_config,
)
from .bleaching import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _bleaching_records,
    build_bleaching_config,
)
from .boxes import (
    build_boxes_config,
)
from .common import (  # noqa: F401  (private helpers the tests and sibling modules import)
    ConfigResult,
    _annotation_boxes,
    _annotation_free,
    _annotation_image_labels,
    _annotation_masks,
    _annotation_points,
    _attrs,
    _canonical,
    _canonical_taxon,
    _free_config_rows,
    _harmonizer,
    _SourceUnmapped,
)
from .constants import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _CORAL,
    BENTHIC_COARSE_TASK,
    BLEACHING_SOURCES,
    BLEACHING_TASK,
    BOX_CONFIG_SOURCES,
    CONFIG_IDS,
    DEPTH_CONFIG_SOURCES,
    HEALTH_TASK,
    INSTANCE_CONFIG_SOURCES,
    LEGACY_VQA_SOURCES,
    MASK_CONFIG_BY_SOURCE,
    MASK_EXPORT_CONFIG_IDS,
    PAIRS_CONFIG_SOURCES,
    POINT_SOURCES,
    ROLLUP_MASK_SOURCES,
    ROLLUP_POINT_SOURCES,
    SEMSEG_SOURCES,
    VQA_CONFIG_SOURCES,
)
from .free_tables import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _build_free_table_config,
    build_captions_config,
    build_depth_config,
    build_pairs_config,
    build_vqa_config,
)
from .masks import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _mask_records,
    build_instance_config,
    build_semseg_config,
)
from .points import (  # noqa: F401  (private helpers the tests and sibling modules import)
    _point_records,
    build_points_config,
)
from .tasklabels import (  # noqa: F401  (private helpers the tests and sibling modules import)
    TASKLABELS_DIR,
    TASKLABELS_MANIFEST,
    _load_label_status,
    _read_parquet,
    _read_tasklabels,
    _row_sha,
    _tasklabels_path,
    resolve_tasklabels_root,
    tasklabels_status,
)

__all__ = [
    "BENTHIC_COARSE_TASK",
    "BLEACHING_SOURCES",
    "BLEACHING_TASK",
    "BOX_CONFIG_SOURCES",
    "CONFIG_IDS",
    "DEPTH_CONFIG_SOURCES",
    "HEALTH_TASK",
    "INSTANCE_CONFIG_SOURCES",
    "LEGACY_VQA_SOURCES",
    "MASK_CONFIG_BY_SOURCE",
    "MASK_EXPORT_CONFIG_IDS",
    "PAIRS_CONFIG_SOURCES",
    "POINT_SOURCES",
    "ROLLUP_MASK_SOURCES",
    "ROLLUP_POINT_SOURCES",
    "SEMSEG_SOURCES",
    "TASKLABELS_DIR",
    "TASKLABELS_MANIFEST",
    "VQA_CONFIG_SOURCES",
    "ConfigResult",
    "apply_flavour",
    "assert_no_mixed_origin_in_eval",
    "build_all_configs",
    "build_benthic_coarse_config",
    "build_benthic_cover_config",
    "build_bleaching_config",
    "build_boxes_config",
    "build_captions_config",
    "build_depth_config",
    "build_instance_config",
    "build_pairs_config",
    "build_points_config",
    "build_semseg_config",
    "build_vqa_config",
    "resolve_tasklabels_root",
    "tasklabels_status",
    "write_configs",
]
