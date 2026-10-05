"""WP-8e: the v2 task-layer configs on the Hub card and in the HF export layout.

Kept out of the restricted ``hf_card.py``/``hf_export.py`` pair (D-M update): each of
those gains one keyword and one call into this module.

* :func:`layout_entries` turns ``{config_id: rows}`` (the parquet rows
  :func:`marinedata.task_layers.configs.write_configs` wrote) into
  :func:`marinedata.hf_export.build_layout` entries — one HF config per id in
  :data:`marinedata.hf_export.TASK_LAYER_CONFIGS`, each row placed in its image's split.
  A row whose image is not in the release has no split and is dropped (counted).
* :func:`card_section` renders :func:`marinedata.hf_card.task_layer_config_table`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..hf_parquet import ConfigSpec, ExportRow
from .configs import CONFIG_IDS

SPLIT_ORDER = ("train", "validation", "test")


def _columns(rows: Sequence[Mapping]) -> tuple[tuple[str, str], ...]:
    names: dict[str, None] = {}
    for row in rows:
        names.update(dict.fromkeys(row))
    return tuple((name, "string") for name in names)


def layout_entries(
    task_layers: Mapping[str, Sequence[Mapping]],
    split_by_sha: Mapping[str, str],
    config_ids: Sequence[str] | None = None,
) -> dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]]:
    if config_ids is None:
        from ..hf_export import TASK_LAYER_CONFIGS as config_ids
    out: dict[str, tuple[ConfigSpec, dict[str, list[ExportRow]]]] = {}
    for config_id in config_ids:
        rows = [
            r for r in task_layers.get(config_id, ()) if (r.get("image_sha256") or r.get("sha256")) in split_by_sha  # noqa: E501
        ]
        if not rows:
            continue
        columns = _columns(rows)
        grouped: dict[str, list[ExportRow]] = {}
        for row in rows:
            values = {n: (None if row.get(n) is None else str(row[n])) for n, _ in columns}
            grouped.setdefault(
                split_by_sha[row.get("image_sha256") or row["sha256"]], []
            ).append(ExportRow(values=values))
        ordered = {s: grouped[s] for s in (*SPLIT_ORDER, *sorted(grouped)) if s in grouped}
        out[config_id] = (ConfigSpec(config_id, columns), ordered)
    return out


def card_section(results: Mapping | None) -> list[str]:
    if not results:
        return []
    from ..hf_card import task_layer_config_table

    return ["## Task layers (v2)", "", task_layer_config_table(dict(results)), ""]


def read_task_layers(
    release_dir: str | Path, config_ids: Sequence[str] = CONFIG_IDS
) -> dict[str, list[dict]]:
    """``{config_id: rows}`` from ``<release_dir>/task_layers/<config_id>.parquet`` (what
    :func:`marinedata.release.build_release` wrote) — the HF export CLI passes this to
    :func:`marinedata.hf_export.build_layout`. A missing or column-less file is skipped."""
    root = Path(release_dir) / "task_layers"
    out: dict[str, list[dict]] = {}
    for config_id in config_ids:
        path = root / f"{config_id}.parquet"
        if not path.is_file():
            continue
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        if table.num_columns:
            out[config_id] = table.to_pylist()
    return out
