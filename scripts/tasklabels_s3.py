"""Build every S3-keyed ``data/_tasklabels/<source_id>/<task>.parquet`` (WP-8e).

    python scripts/tasklabels_s3.py --out data/_tasklabels [--only reefolution,...] [--force]

Prints one JSON line per (source, task) with its row and image counts. Resumable: a job
whose parquet already exists is skipped (``_write`` renames a finished file into place, so
an existing file is a complete one) unless ``--force``. A per-item GET that still fails
after the backoff is skipped and listed in ``<out>/missing_keys.tsv`` (task, key,
last_status); a failed whole-file GET (CHECKSUMS, metadata, labels parquet) fails only
that job, is listed there too, and every other job still runs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from marinedata.task_layers import s3_keyed
from marinedata.task_layers.producers.coralscapes_semseg import _ID_TO_LABEL

# source_id -> (staged tree, task, kind, extra)
JOBS: dict[str, tuple[str, str, str, dict]] = {
    "reefolution": ("reefolution/2026-09-23-2c84cb0c9cda", "points", "points", {}),
    "mermaid-aws": ("mermaid-aws/2026-09-19-bc53d5a2c0b6", "points", "points", {}),
    "coralscapes": ("coralscapes/1.0", "semseg", "semseg", {}),
    # registry/sources/reef-support-own.yaml loader.params.mask_values (WS-D S31)
    "reef-support-benthic-own": (
        "reef-support-benthic-own/2026-09-24",
        "semseg",
        "semseg",
        {"id_to_label": {1: "Hard Coral", 2: "Soft Coral"}},
    ),
    "noaa-pifsc-bleaching": ("noaa-pifsc-bleaching/1-image-labels", "bleaching", "image", {}),
    "roboflow-coral-bleaching-final-v6i": (
        "roboflow-coral-bleaching-final-v6i/v6i-image-labels-r2",
        "bleaching",
        "image",
        {},
    ),
    "roboflow-coral-bleaching-general-v1-yolov8s": (
        "roboflow-coral-bleaching-general-v1-yolov8s/v1-yolov8s-image-labels-r2",
        "bleaching",
        "image",
        {},
    ),
    "roboflow-coral-classification-copy-changed-v13i": (
        "roboflow-coral-classification-copy-changed-v13i/v13i-image-labels",
        "bleaching",
        "image",
        {},
    ),
    "roboflow-coral-reef-bleach-detection-v2i": (
        "roboflow-coral-reef-bleach-detection-v2i/v2i-image-labels",
        "bleaching",
        "image",
        {},
    ),
    "roboflow-coral-reef-classification-v3i": (
        "roboflow-coral-reef-classification-v3i/v3i-image-labels",
        "bleaching",
        "image",
        {},
    ),
    # registry/sources/reef-support-own.yaml loader.params.mask_values (WS-D S31)
    "reef-support-bleaching": (
        "reef-support-bleaching/2026-09-24",
        "bleaching",
        "mask",
        {"values": {1: "bleached", 2: "non_bleached"}},
    ),
}


MISSING_HEADER = ("task", "key", "last_status")


def _record_missing(out: Path, task: str, missing: list[tuple[str, str]]) -> None:
    """Rewrite ``<out>/missing_keys.tsv`` with this task's rows replaced by ``missing``."""
    path = out / "missing_keys.tsv"
    rows: list[tuple[str, ...]] = []
    if path.is_file():
        with path.open(newline="") as fh:
            rows = [tuple(r) for r in csv.reader(fh, delimiter="\t")][1:]
    rows = [r for r in rows if r and r[0] != task] + [(task, k, st) for k, st in missing]
    out.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t", lineterminator="\n")
        writer.writerow(MISSING_HEADER)
        writer.writerows(sorted(rows))


def run(source_id: str, out: Path, *, force: bool = False, fetch=s3_keyed.fetch_small) -> dict:
    tree_id, task, kind, extra = JOBS[source_id]
    path = out / source_id / f"{task}.parquet"
    label = f"{source_id}/{task}"
    if path.is_file() and not force:
        return {"source_id": source_id, "task": task, "skipped": "complete", "path": str(path)}
    tree = None
    try:
        tree = s3_keyed.StagedTree(tree_id, fetch)
        if kind == "points":
            n = s3_keyed.produce_points(tree, source_id, path)
        elif kind == "semseg":
            n = s3_keyed.produce_semseg(
                tree, source_id, extra.get("id_to_label", _ID_TO_LABEL), path
            )
        elif kind == "image":
            n = s3_keyed.produce_image_labels(tree, source_id, path)
        else:
            n = s3_keyed.produce_mask_presence(tree, source_id, extra["values"], path)
    except s3_keyed.FetchFailed as exc:
        missing = [*(tree.missing if tree else []), (exc.key, exc.status)]
        _record_missing(out, label, missing)
        return {"source_id": source_id, "task": task, "failed": str(exc), "missing": len(missing)}
    _record_missing(out, label, tree.missing)
    import pyarrow.parquet as pq

    images = len(set(pq.read_table(path, columns=["sha256"]).column(0).to_pylist())) if n else 0
    return {
        "source_id": source_id,
        "task": task,
        "rows": n,
        "images": images,
        "staged_images": len(tree.shas),
        "missing": len(tree.missing),
        "path": str(path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/_tasklabels"))
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true", help="rebuild jobs whose parquet exists")
    args = ap.parse_args()
    for source_id in [s for s in args.only.split(",") if s] or list(JOBS):
        print(json.dumps(run(source_id, args.out, force=args.force)), flush=True)


if __name__ == "__main__":
    main()
