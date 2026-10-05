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

from ..annotation_schema import annotator_from_origin
from ..registry import Registry
from ..schema import Axis
from ..tables import _require_pyarrow
from .masks_table import MASK_SOURCES
from .rollup import MIXED, UNKNOWN, rollup_counts

BENTHIC_COARSE_TASK = "benthic-coarse"
BLEACHING_TASK = "bleaching-condition"
HEALTH_TASK = "coral-health-binary"

CONFIG_IDS = ("points", "vqa", "semseg", "benthic-coarse", "benthic-cover", "bleaching")

# WP-8e-resume (manager decision 3): the staged point/mask sources each config reads.
POINT_SOURCES = ("reefolution", "mermaid-aws")
# WP-U4: registry-driven. The semseg config reads every source with a unified ``masks`` producer
# (``masks_table.MASK_SOURCES``); the benthic rollups stay on the two benthic-cover sources (a
# pseudo-label, a 3-class and a scene-segmentation source must not feed cover).
SEMSEG_SOURCES = tuple(MASK_SOURCES)
ROLLUP_MASK_SOURCES = ("coralscapes", "reef-support-benthic-own")
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
        return len({_row_sha(row) for row in self.rows})


def _row_sha(row: Mapping) -> str | None:
    """The image join key of a task-layer row: ``image_sha256`` (unified schema) else the legacy
    ``sha256`` (D-Z2 parquet written before WP-U3 keeps being read)."""
    return row.get("image_sha256") or row.get("sha256")


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


TASKLABELS_DIR = "_tasklabels"


def resolve_tasklabels_root(
    registry: Registry | None = None, explicit: str | Path | None = None
) -> Path:
    """The directory holding ``_tasklabels/`` (and ``_labelquality/``) — ``<repo>/data``.

    INT-core3c: the release used to read ``Path(".")``, so the configs depended on the
    caller's cwd and came out empty from the repo root (``./_tasklabels`` does not exist
    there). Now: an ``explicit`` path wins (a path ending in ``_tasklabels`` is taken to
    mean its parent); otherwise the root is derived from the registry's location
    (``<repo>/registry`` → ``<repo>/data``), falling back to the source-tree layout.
    Fails closed — a root with no ``_tasklabels/`` raises rather than yield empty configs.
    """
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if root.name == TASKLABELS_DIR:
            root = root.parent
        candidates = [root]
    else:
        candidates = []
        registry_root = getattr(registry, "root", None)
        if registry_root is not None:
            candidates.append(Path(registry_root).resolve().parent / "data")
        candidates.append(Path(__file__).resolve().parents[3] / "data")
    for root in candidates:
        if (root / TASKLABELS_DIR).is_dir():
            return root
    tried = ", ".join(str(c / TASKLABELS_DIR) for c in candidates)
    raise FileNotFoundError(
        f"no {TASKLABELS_DIR}/ directory found (tried {tried}); pass --tasklabels-root"
    )


TASKLABELS_MANIFEST = "MANIFEST.json"
"""``<base_dir>/_tasklabels/MANIFEST.json``: ``{"files": {"<source_id>/<task>.parquet":
{"status": "invalid", "reason": ...}}}``. A file marked ``invalid`` is tracked for the
record but never read into a config (INT-core3b: coralvqa's null-keyed ``vqa.parquet``)."""


def tasklabels_status(base_dir: Path, source_id: str, task: str) -> str:
    """``"invalid"`` if the tasklabels manifest marks this file so, else ``"ok"``."""
    path = Path(base_dir) / "_tasklabels" / TASKLABELS_MANIFEST
    if not path.is_file():
        return "ok"
    files = json.loads(path.read_text()).get("files", {})
    return str(files.get(f"{source_id}/{task}.parquet", {}).get("status", "ok"))


def _read_tasklabels(base_dir: Path, source_id: str, task: str) -> list[dict]:
    """The D-Z2 rows of one tasklabels file — none if the manifest marks it ``invalid``,
    and never a row without its ``sha256`` join key (a null key joins to no image)."""
    if tasklabels_status(base_dir, source_id, task) == "invalid":
        return []
    rows = _read_parquet(_tasklabels_path(base_dir, source_id, task))
    return [row for row in rows if _row_sha(row)]


def _annotation_points(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``points`` rows of ``source_id`` (``_annotations/points/<source>/<version>.parquet``,
    newest version), or ``[]`` when none were written. Rows without an ``image_sha256`` are skipped."""  # noqa: E501
    root = base_dir / "_annotations" / "points" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_masks(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``masks`` rows of ``source_id`` (newest ``_annotations/masks/<source>/<version>.parquet``)."""  # noqa: E501
    root = base_dir / "_annotations" / "masks" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _annotation_image_labels(base_dir: Path, source_id: str) -> list[dict]:
    """The unified ``image_labels`` rows of ``source_id`` (newest ``_annotations/image_labels/<source>/<version>.parquet``)."""  # noqa: E501
    root = base_dir / "_annotations" / "image_labels" / source_id
    files = sorted(root.glob("*.parquet")) if root.is_dir() else []
    return [r for r in _read_parquet(files[-1]) if r.get("image_sha256")] if files else []


def _bleaching_records(base_dir: Path, source_id: str) -> list[dict]:
    """Bleaching rows of one source in the legacy shape (``sha256``, ``label_origin``,
    ``native_label``, ``evidence``, ``pixel_count``, ``confidence``) plus the unified columns
    (``image_sha256, ann_id, condition_node_id, match_type, annotator_type, licence_class``).
    The unified ``image_labels`` table wins (WP-U5); else ``_tasklabels/<source>/bleaching.parquet``."""  # noqa: E501
    unified = _annotation_image_labels(base_dir, source_id)
    if not unified:
        return _read_tasklabels(base_dir, source_id, "bleaching")
    out = []
    for r in unified:
        extra = _attrs(r)
        out.append(
            {
                "sha256": r["image_sha256"], "label_origin": r.get("annotator_type"),
                "native_label": r["label_native"], "evidence": extra.get("evidence", "image"),
                "pixel_count": extra.get("pixel_count"), "confidence": r.get("confidence"),
                "image_sha256": r["image_sha256"], "ann_id": r.get("ann_id"),
                "condition_node_id": r.get("condition_node_id"), "unified": True,
                "match_type": r.get("match_type"), "annotator_type": r.get("annotator_type"),
                "licence_class": extra.get("licence_class"),
            }
        )  # fmt: skip
    return out


def _attrs(row: Mapping) -> dict:
    return json.loads(row["attrs"]) if row.get("attrs") else {}


def _mask_records(base_dir: Path, source_id: str) -> list[dict]:
    """Semantic mask rows of one source in the legacy semseg shape (``sha256``, ``label_origin``,
    ``mask_key``, ``class_counts``) plus the unified columns. The unified table wins; else the
    legacy ``_tasklabels/<source>/semseg.parquet`` is read as before."""
    unified = _annotation_masks(base_dir, source_id)
    if not unified:
        return _read_tasklabels(base_dir, source_id, "semseg")
    return [
        {
            "sha256": r["image_sha256"],
            "label_origin": r.get("annotator_type"),
            "mask_key": r["mask_ref"],
            "class_counts": r["class_counts"],
            "image_sha256": r["image_sha256"],
            "ann_id": r.get("ann_id"),
            "mask_kind": r.get("mask_kind"),
            "match_type": r.get("match_type"),
            "annotator_type": r.get("annotator_type"),
            "ann_license": r.get("ann_license"),
            "licence_class": _attrs(r).get("licence_class"),
            "taxon_by_label": {
                k: v["taxon_node_id"] for k, v in _attrs(r).get("class_resolution", {}).items()
            },  # noqa: E501, RUF100
        }
        for r in unified
    ]


def _point_records(base_dir: Path, source_id: str) -> list[dict]:
    """Point rows of one source in the unified column names. The unified table wins; else the
    legacy ``_tasklabels/<source>/points.parquet`` is renamed on read (``native_label`` ->
    ``label_native``, ``label_origin`` -> ``annotator_type``, ``sha256`` -> ``image_sha256``)
    with ``match_type`` left null (legacy rows were never resolved through ``Crosswalk.resolve``)."""  # noqa: E501
    unified = _annotation_points(base_dir, source_id)
    if unified:
        return unified
    out: list[dict] = []
    for i, point in enumerate(_read_tasklabels(base_dir, source_id, "points")):
        origin = point.get("label_origin")
        out.append(
            {
                "image_sha256": _row_sha(point),
                "source_id": source_id,
                "ann_id": f"{source_id}:{i}",
                "label_native": point.get("label_native") or point["native_label"],
                "annotator_type": point.get("annotator_type")
                or annotator_from_origin(origin or "human")[0],  # noqa: E501, RUF100
                "x": point["x"],
                "y": point["y"],
            }
        )
    return out


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
    """``points``: every :data:`POINT_SOURCES` point in the unified column names (WP-U3):
    ``image_sha256, source_id, ann_id, label_native, label_set, taxon_node_id, match_type,
    worms_aphia_id, rs_benthic_code, annotator_type, x, y, x_px, y_px, label_status``."""
    base_dir = Path(base_dir)
    label_status = _load_label_status(base_dir)
    unmapped: dict[str, _SourceUnmapped] = {}
    rows: list[dict] = []

    for source_id in POINT_SOURCES:
        tally = unmapped.setdefault(source_id, _SourceUnmapped())
        for point in _point_records(base_dir, source_id):
            native = point["label_native"]
            taxon = point.get("taxon_node_id") or _canonical_taxon(registry, source_id, native)
            if taxon is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            rows.append(
                {
                    "image_sha256": point["image_sha256"],
                    "source_id": source_id,
                    "ann_id": point["ann_id"],
                    "label_native": native,
                    "label_set": point.get("label_set"),
                    "taxon_node_id": taxon,
                    "match_type": point.get("match_type")
                    or ("unmapped" if taxon is None else None),  # noqa: E501, RUF100
                    "worms_aphia_id": point.get("worms_aphia_id"),
                    "rs_benthic_code": point.get("rs_benthic_code"),
                    "annotator_type": point["annotator_type"],
                    "x": point["x"],
                    "y": point["y"],
                    "x_px": point.get("x_px"),
                    "y_px": point.get("y_px"),
                    "label_status": label_status.get(point["image_sha256"], "ok"),
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
        for record in _read_tasklabels(base_dir, source_id, "vqa"):
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
        for record in _mask_records(base_dir, source_id):
            native_counts: dict[str, int] = json.loads(record["class_counts"])
            canonical_counts: dict[str, int] = {}
            resolved = record.get("taxon_by_label")  # unified rows carry U2's per-class resolve
            for native_label, n in native_counts.items():
                canonical = (
                    resolved.get(native_label)
                    if resolved is not None
                    else _canonical_taxon(registry, source_id, native_label)
                )
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
                    **{
                        k: record[k]  # unified columns, present only on unified rows
                        for k in (
                            "image_sha256",
                            "ann_id",
                            "mask_kind",
                            "match_type",
                            "annotator_type",
                            "ann_license",
                            "licence_class",
                        )  # noqa: E501, RUF100
                        if k in record
                    },
                }
            )

    return ConfigResult("semseg", tuple(rows), {sid: t.fraction() for sid, t in unmapped.items()})


def _coarse_counts_from_points(
    registry: Registry, projector, source_id: str, points: list[dict]
) -> dict[str, dict[str, int]]:
    """``{sha256: {coarse_class_or_unknown: n}}`` for one source's point rows."""
    per_image: dict[str, dict[str, int]] = {}
    for point in points:
        counts = per_image.setdefault(point["image_sha256"], {})
        canonical = point.get("taxon_node_id") or _canonical_taxon(
            registry, source_id, point["label_native"]
        )  # noqa: E501, RUF100
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
        points = _point_records(base_dir, source_id)
        add(source_id, _coarse_counts_from_points(registry, projector, source_id, points))
    for source_id in ROLLUP_MASK_SOURCES:
        masks = _mask_records(base_dir, source_id)
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
        for record in _bleaching_records(base_dir, source_id):
            node = (
                record.get("condition_node_id")
                if record.get("unified")
                else _canonical(registry, source_id, record["native_label"], Axis.CONDITION)
            )
            if node is None:
                tally.unmapped += 1
            else:
                tally.known += 1
            rows.append(
                {
                    "sha256": record["sha256"],
                    "image_sha256": record["sha256"],
                    "source_id": source_id,
                    "ann_id": record.get("ann_id"),
                    "label_origin": record["label_origin"],
                    "annotator_type": record.get("annotator_type"),
                    "licence_class": record.get("licence_class"),
                    "native_label": record["native_label"],
                    "label_native": record["native_label"],
                    "canonical_condition": node,
                    "match_type": record.get("match_type"),
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
        split = split_of.get(_row_sha(row))
        if split is None or split == train_split:
            continue
        origins_by_split.setdefault(split, set()).add(
            row.get("label_origin") or row["annotator_type"]
        )  # noqa: E501, RUF100
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
