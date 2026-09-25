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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

from ..registry import Registry
from ..schema import Axis
from ..tables import _require_pyarrow
from .rollup import MIXED, UNKNOWN, rollup_counts

BENTHIC_COARSE_TASK = "benthic-coarse"
BLEACHING_TASK = "bleaching-condition"
HEALTH_TASK = "coral-health-binary"

CONFIG_IDS = ("points", "vqa", "semseg", "benthic-coarse", "benthic-cover", "bleaching")

# WP-8e-resume (manager decision 3): the staged point/mask sources each config reads.
POINT_SOURCES = ("reefolution", "mermaid-aws")
SEMSEG_SOURCES = ("coralscapes", "reef-support-benthic-own")
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


@dataclass(frozen=True)
class ConfigResult:
    """One config's built rows plus the bookkeeping the card/report needs."""

    config_id: str
    rows: tuple[dict, ...]
    unmapped_by_source: dict[str, float]
    """Fraction of native labels/pixels that had no crosswalk edge, per source_id —
    reported so a lossy crosswalk (e.g. Coralscapes' missing soft-coral row) is visible
    on the card, not just silently dropped."""

    @property
    def n_images(self) -> int:
        return len({row["sha256"] for row in self.rows})


def _read_parquet(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(path).to_pylist()


def _load_label_status(base_dir: Path) -> dict[str, str]:
    """``{sha256: label_status}`` from the WP-9b quality parquet, if present (D-U).

    Defaults to ``"ok"`` for anything not in the file — this join is additive, never a
    filter: absence of the file (or of a row) means "not yet reviewed", not "excluded".
    """
    path = base_dir / "_labelquality" / "2026-09-25" / "label_status.parquet"
    rows = _read_parquet(path)
    return {row["sha256"]: row["label_status"] for row in rows}


def _tasklabels_path(base_dir: Path, source_id: str, task: str) -> Path:
    return base_dir / "_tasklabels" / source_id / f"{task}.parquet"


@dataclass
class _SourceUnmapped:
    known: int = 0
    unmapped: int = 0

    def fraction(self) -> float:
        total = self.known + self.unmapped
        return self.unmapped / total if total else 0.0


@cache
def _harmonizer(registry: Registry, source_id: str):
    return registry.harmonizer_for(source_id)


@lru_cache(maxsize=1 << 16)
def _canonical(registry: Registry, source_id: str, native_label: str, axis: Axis) -> str | None:
    """Native label -> canonical node id on ``axis``, or ``None`` if unmappable (memoised:
    a full-scale points file repeats a few hundred labels across ~0.5M rows)."""
    harmonizer = _harmonizer(registry, source_id)
    if harmonizer is None:
        return None
    value = harmonizer.map_label(native_label).labels.get(axis)
    return value.node_id if value is not None else None


def _canonical_taxon(registry: Registry, source_id: str, native_label: str) -> str | None:
    """Native label -> canonical ``taxon`` node id, or ``None`` if unmappable."""
    return _canonical(registry, source_id, native_label, Axis.TAXON)


def build_points_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``points``: every :data:`POINT_SOURCES` point row, native label + canonical taxon."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []

    for source_id in POINT_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for point in _read_parquet(_tasklabels_path(base_dir, source_id, "points")):
            canonical = _canonical_taxon(registry, source_id, point["native_label"])
            if canonical is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            rows.append(
                {
                    "sha256": point["sha256"],
                    "source_id": source_id,
                    "label_origin": point["label_origin"],
                    "native_label": point["native_label"],
                    "canonical_taxon": canonical,
                    "x": point["x"],
                    "y": point["y"],
                    "label_status": label_status.get(point["sha256"], "ok"),
                }
            )

    return ConfigResult("points", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()})


def build_vqa_config(base_dir: str | Path) -> ConfigResult:
    """``vqa``: CoralVQA (today: train split) Q/A rows, unchanged — no crosswalk applies
    to free-text VQA (D-Y note: "treat as a language/UX asset, not a measurement one")."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    rows: list[dict] = []
    for source_id in ("coralvqa",):
        for record in _read_parquet(_tasklabels_path(base_dir, source_id, "vqa")):
            rows.append({**record, "label_status": label_status.get(record["sha256"], "ok")})
    return ConfigResult("vqa", tuple(rows), {})


def build_semseg_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``semseg``: Coralscapes (today) per-image mask rows, native + canonical pixel
    counts (unmapped pixels — e.g. Coralscapes has no soft-coral class — kept out of the
    canonical histogram, tallied in ``unmapped_by_source``)."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []

    for source_id in SEMSEG_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for record in _read_parquet(_tasklabels_path(base_dir, source_id, "semseg")):
            native_counts: dict[str, int] = json.loads(record["class_counts"])
            canonical_counts: dict[str, int] = {}
            for native_label, n in native_counts.items():
                canonical = _canonical_taxon(registry, source_id, native_label)
                if canonical is None:
                    tally.unmapped += n
                    continue
                tally.known += n
                canonical_counts[canonical] = canonical_counts.get(canonical, 0) + n
            rows.append(
                {
                    "sha256": record["sha256"],
                    "source_id": source_id,
                    "label_origin": record["label_origin"],
                    "mask_key": record["mask_key"],
                    "class_counts": record["class_counts"],
                    "canonical_class_counts": json.dumps(canonical_counts, sort_keys=True),
                    "label_status": label_status.get(record["sha256"], "ok"),
                }
            )

    return ConfigResult("semseg", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()})


def _coarse_counts_from_points(
    registry: Registry, projector, source_id: str, points: list[dict]
) -> dict[str, dict[str, int]]:
    """``{sha256: {coarse_class_or_unknown: n}}`` for one source's point rows."""
    per_image: dict[str, dict[str, int]] = {}
    for point in points:
        counts = per_image.setdefault(point["sha256"], {})
        canonical = _canonical_taxon(registry, source_id, point["native_label"])
        target = projector.project(canonical).target_class if canonical is not None else None
        cls = target or UNKNOWN
        counts[cls] = counts.get(cls, 0) + 1
    return per_image


def _coarse_counts_from_masks(
    registry: Registry, projector, source_id: str, masks: list[dict]
) -> dict[str, dict[str, int]]:
    """``{sha256: {coarse_class_or_unknown: n}}`` for one source's mask rows."""
    per_image: dict[str, dict[str, int]] = {}
    for record in masks:
        native_counts: dict[str, int] = json.loads(record["class_counts"])
        counts: dict[str, int] = {}
        for native_label, n in native_counts.items():
            canonical = _canonical_taxon(registry, source_id, native_label)
            target = projector.project(canonical).target_class if canonical is not None else None
            cls = target or UNKNOWN
            counts[cls] = counts.get(cls, 0) + n
        per_image[record["sha256"]] = counts
    return per_image


def _build_benthic_rollup(
    registry: Registry, base_dir: Path
) -> tuple[dict[str, tuple[str, dict, object]], dict[str, float]]:
    """Shared D-Y rollup for both benthic configs: ``{sha256: (source_id, origin,
    rollup_result)}`` plus per-source unmapped fractions."""
    projector = registry.projector_for(BENTHIC_COARSE_TASK)
    origins: dict[str, str] = {}
    per_image_counts: dict[str, dict[str, int]] = {}
    unmapped: dict[str, _SourceUnmapped] = {}
    source_of: dict[str, str] = {}

    def add(source_id: str, per_source: dict[str, dict[str, int]]) -> None:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for sha256, counts in per_source.items():
            per_image_counts.setdefault(sha256, {})
            for cls, n in counts.items():
                per_image_counts[sha256][cls] = per_image_counts[sha256].get(cls, 0) + n
                if cls == UNKNOWN:
                    tally.unmapped += n
                else:
                    tally.known += n
            origins.setdefault(sha256, "human")
            source_of.setdefault(sha256, source_id)

    for source_id in POINT_SOURCES:
        points = _read_parquet(_tasklabels_path(base_dir, source_id, "points"))
        add(source_id, _coarse_counts_from_points(registry, projector, source_id, points))
    for source_id in SEMSEG_SOURCES:
        masks = _read_parquet(_tasklabels_path(base_dir, source_id, "semseg"))
        add(source_id, _coarse_counts_from_masks(registry, projector, source_id, masks))

    result = {
        sha256: (source_of[sha256], origins[sha256], rollup_counts(counts))
        for sha256, counts in per_image_counts.items()
    }
    return result, {sid: t.fraction() for sid, t in unmapped.items()}


def build_benthic_coarse_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``benthic-coarse``: single-label view (D-Z). ``mixed`` -> null."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    per_image, unmapped_by_source = _build_benthic_rollup(registry, base_dir)
    rows = []
    for sha256, (source_id, origin, result) in per_image.items():
        dominant = None if result.dominant in (None, MIXED) else result.dominant
        rows.append(
            {
                "sha256": sha256,
                "source_id": source_id,
                "label_origin": origin,
                "benthic_dominant": dominant,
                "label_status": label_status.get(sha256, "ok"),
            }
        )
    return ConfigResult("benthic-coarse", tuple(rows), unmapped_by_source)


def build_benthic_cover_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``benthic-cover``: the full D-Y multi-label rollup (cover/dominant/present)."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    per_image, unmapped_by_source = _build_benthic_rollup(registry, base_dir)
    rows = []
    for sha256, (source_id, origin, result) in per_image.items():
        rows.append(
            {
                "sha256": sha256,
                "source_id": source_id,
                "label_origin": origin,
                "benthic_cover": json.dumps(result.cover, sort_keys=True),
                "benthic_dominant": result.dominant,
                "benthic_present": json.dumps(sorted(result.present)),
                "label_status": label_status.get(sha256, "ok"),
            }
        )
    return ConfigResult("benthic-cover", tuple(rows), unmapped_by_source)


def build_bleaching_config(registry: Registry, base_dir: str | Path) -> ConfigResult:
    """``bleaching``: one row per (image, native condition label) from every
    :data:`BLEACHING_SOURCES` file, with the canonical ``condition`` node and its
    projection onto both condition tasks — ``bleaching-condition`` (6-class; abstains,
    i.e. null, on a label too coarse to split, e.g. Roboflow "Unhealthy") and
    ``coral-health-binary`` (HEALTHY/UNHEALTHY). A label with no condition edge is
    unmapped (tallied per source by row), never guessed."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    fine = registry.projector_for(BLEACHING_TASK)
    binary = registry.projector_for(HEALTH_TASK)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []
    for source_id in BLEACHING_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for record in _read_parquet(_tasklabels_path(base_dir, source_id, "bleaching")):
            node = _canonical(registry, source_id, record["native_label"], Axis.CONDITION)
            if node is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            rows.append(
                {
                    "sha256": record["sha256"],
                    "source_id": source_id,
                    "label_origin": record["label_origin"],
                    "native_label": record["native_label"],
                    "canonical_condition": node,
                    "bleaching_condition": fine.project(node).target_class if node else None,
                    "coral_health": binary.project(node).target_class if node else None,
                    "evidence": record.get("evidence"),
                    "pixel_count": record.get("pixel_count"),
                    "confidence": record.get("confidence"),
                    "label_status": label_status.get(record["sha256"], "ok"),
                }
            )
    return ConfigResult(
        "bleaching", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()}
    )


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
        split = split_of.get(row["sha256"])
        if split is None or split == train_split:
            continue
        origins_by_split.setdefault(split, set()).add(row["label_origin"])
    mixed = {split: origins for split, origins in origins_by_split.items() if len(origins) > 1}
    if mixed:
        raise ValueError(f"human and model rows mixed in eval split(s): {mixed}")


def build_all_configs(registry: Registry, base_dir: str | Path) -> dict[str, ConfigResult]:
    """Every config the sources available today can build (D-Z2 producers wired)."""
    base_dir = Path(base_dir)
    return {
        "points": build_points_config(registry, base_dir),
        "vqa": build_vqa_config(base_dir),
        "semseg": build_semseg_config(registry, base_dir),
        "benthic-coarse": build_benthic_coarse_config(registry, base_dir),
        "benthic-cover": build_benthic_cover_config(registry, base_dir),
        "bleaching": build_bleaching_config(registry, base_dir),
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
