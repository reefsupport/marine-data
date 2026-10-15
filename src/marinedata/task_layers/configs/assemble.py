"""Task-layer configs: assemble (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

from ...licence_class import drop_release_excluded, flavour_filter, require_flavour
from ...registry import Registry
from ...tables import _require_pyarrow
from .benthic import build_benthic_coarse_config, build_benthic_cover_config
from .bleaching import build_bleaching_config
from .boxes import build_boxes_config
from .common import ConfigResult
from .free_tables import (
    build_captions_config,
    build_depth_config,
    build_pairs_config,
    build_vqa_config,
)
from .masks import build_instance_config, build_semseg_config
from .points import build_points_config
from .tasklabels import _row_sha


def assert_no_mixed_origin_in_eval(
    rows: list[dict], split_of: Mapping[str, str], *, train_split: str = "train"
) -> None:
    """D-Y/WP-8c invariant: a non-train split must be entirely one ``label_origin``.

    Raises ``ValueError`` naming the offending split — a val/test split half human, half
    model-generated silently changes what the eval number means depending on which rows
    a later filter happens to keep.
    """
    origins_by_split: dict[str, set[str]] = {}
    for row in rows:
        split = split_of.get(_row_sha(row))
        if split is None or split == train_split:
            continue
        origins_by_split.setdefault(split, set()).add(
            row.get("label_origin") or row["annotator_type"]
        )  # noqa: E501, RUF100
    mixed = {split: origins for split, origins in origins_by_split.items() if len(origins) > 1}
    if mixed:
        raise ValueError(f"human and model rows mixed in eval split(s): {mixed}")


def apply_flavour(result: ConfigResult, flavour: str | None) -> ConfigResult:
    """``result`` restricted to the rows ``flavour`` may ship; with ``flavour=None`` (test
    escape only) the release-excluded sources are still dropped."""
    rows = flavour_filter(result.rows, flavour) if flavour else drop_release_excluded(result.rows)
    return dataclasses.replace(result, rows=tuple(rows))


def build_all_configs(
    registry: Registry, base_dir: str | Path, flavour: str | None = None
) -> dict[str, ConfigResult]:
    """Every config the sources available today can build (D-Z2 producers wired).
    ``flavour`` (``open`` | ``nc``) keeps only the rows that flavour ships (WP-L1a)."""
    require_flavour(flavour, "build_all_configs")
    built = _build_all_configs(registry, Path(base_dir))
    return {k: apply_flavour(v, flavour) for k, v in built.items()}


def _build_all_configs(registry: Registry, base_dir: Path) -> dict[str, ConfigResult]:
    return {
        "points": build_points_config(registry, base_dir),
        "vqa": build_vqa_config(base_dir),
        "semseg": build_semseg_config(registry, base_dir),
        "benthic-coarse": build_benthic_coarse_config(registry, base_dir),
        "benthic-cover": build_benthic_cover_config(registry, base_dir),
        "bleaching": build_bleaching_config(registry, base_dir),
        "boxes": build_boxes_config(registry, base_dir),
        "captions": build_captions_config(base_dir),
        "depth": build_depth_config(base_dir),
        "pairs": build_pairs_config(base_dir),
        "instances": build_instance_config(registry, base_dir),
    }


def write_configs(results: Mapping[str, ConfigResult], out_dir: str | Path) -> dict[str, Path]:
    """Write each config's rows as parquet under ``out_dir/<config_id>.parquet``."""
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for config_id, result in results.items():
        path = out_dir / f"{config_id}.parquet"
        table = pa.Table.from_pylist(list(result.rows)) if result.rows else pa.table({})
        pq.write_table(table, path)
        written[config_id] = path
    return written
