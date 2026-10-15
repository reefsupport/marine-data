"""The task-layer configs module was split into a package (P1.13): nothing was lost on the way.

The 80 names below are every top-level definition of the former 824-line ``configs.py``. Each one
must still import from ``marinedata.task_layers.configs`` (so no caller changes), be defined in
exactly one submodule, and no submodule may pass 400 lines.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import marinedata.task_layers.configs as configs

PKG = Path(configs.__file__).parent
FORMER_TOP_LEVEL_NAMES = [
    "BENTHIC_COARSE_TASK",
    "BLEACHING_SOURCES",
    "BLEACHING_TASK",
    "BOX_CONFIG_SOURCES",
    "CONFIG_IDS",
    "ConfigResult",
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
    "_CORAL",
    "_SourceUnmapped",
    "_annotation_boxes",
    "_annotation_free",
    "_annotation_image_labels",
    "_annotation_masks",
    "_annotation_points",
    "_attrs",
    "_bleaching_records",
    "_build_all_configs",
    "_build_benthic_rollup",
    "_build_free_table_config",
    "_canonical",
    "_canonical_taxon",
    "_coarse_counts_from_masks",
    "_coarse_counts_from_points",
    "_free_config_rows",
    "_harmonizer",
    "_load_label_status",
    "_mask_records",
    "_point_records",
    "_read_parquet",
    "_read_tasklabels",
    "_row_sha",
    "_tasklabels_path",
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
MAX_LINES = 400


def _top_level_names(path: Path) -> list[str]:
    out: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            out.append(node.name)
        elif isinstance(node, ast.Assign):
            out.extend(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.append(node.target.id)
    return out


def _submodules() -> list[Path]:
    return sorted(p for p in PKG.glob("*.py") if p.name != "__init__.py")


def test_the_flat_module_is_gone_and_the_package_replaces_it():
    assert not (PKG.parent / "configs.py").exists()
    assert (PKG / "__init__.py").is_file()


def test_every_former_top_level_name_still_imports_from_the_package():
    missing = [n for n in FORMER_TOP_LEVEL_NAMES if not hasattr(configs, n)]
    assert missing == []


def test_every_former_name_is_defined_in_exactly_one_submodule():
    defined: dict[str, list[str]] = {}
    for path in _submodules():
        for name in _top_level_names(path):
            defined.setdefault(name, []).append(path.stem)
    assert sorted(defined) == sorted(FORMER_TOP_LEVEL_NAMES)
    assert {n: m for n, m in defined.items() if len(m) != 1} == {}


def test_reexports_are_the_submodule_objects_and_public_names_are_in_all():
    for name in FORMER_TOP_LEVEL_NAMES:
        obj = getattr(configs, name)
        owner = getattr(obj, "__module__", None)
        if callable(obj) and owner is not None:
            assert owner.startswith("marinedata.task_layers.configs."), (name, owner)
            assert getattr(importlib.import_module(owner), name) is obj
    public = [n for n in FORMER_TOP_LEVEL_NAMES if not n.startswith("_")]
    assert sorted(configs.__all__) == sorted(public)


def test_no_module_of_the_package_passes_the_line_limit():
    sizes = {p.name: len(p.read_text(encoding="utf-8").splitlines()) for p in PKG.glob("*.py")}
    assert {n: s for n, s in sizes.items() if s > MAX_LINES} == {}
