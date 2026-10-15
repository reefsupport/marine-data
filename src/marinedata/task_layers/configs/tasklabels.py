"""Task-layer configs: tasklabels (split out of the former configs.py, no behaviour change)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ...registry import Registry
from ...tables import _require_pyarrow


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
