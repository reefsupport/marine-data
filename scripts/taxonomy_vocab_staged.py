"""WP-7d: vocab TSVs for the staged sources that had a crosswalk but no measured vocabulary.

    python scripts/taxonomy_vocab_staged.py <cache_dir> <repo_root>

Reads only labels and indexes, by anonymous HTTPS GET of known keys on rs-storage-open
(D-O: no ListBucket). Whole-image sources: ``labels/image_labels.parquet`` (one row per
image label). Dense-mask sources: every ``labels/masks/**.png`` named in the tree's
``CHECKSUMS.sha256`` is streamed into memory, its distinct pixel values counted, and
dropped (nothing is written to disk but the per-source count JSON). Needs numpy + Pillow
+ pyarrow. The count unit is written into each TSV header.
"""

from __future__ import annotations

import io
import json
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from taxonomy_vocab import write

BASE = "https://rs-storage-open.hel1.your-objectstorage.com/sources/"

# source id -> (staged tree, crosswalk id, mask pixel values or None for image labels)
IMAGE_LABELS = {
    "noaa-pifsc-bleaching": (
        "noaa-pifsc-bleaching/1-image-labels",
        "noaa-pifsc-bleaching-condition",
    ),
    "roboflow-coral-bleaching-final-v6i": (
        "roboflow-coral-bleaching-final-v6i/v6i-image-labels-r2",
        "roboflow-bleaching-condition-hb",
    ),
    "roboflow-coral-bleaching-general-v1-yolov8s": (
        "roboflow-coral-bleaching-general-v1-yolov8s/v1-yolov8s-image-labels-r2",
        "roboflow-bleaching-condition-hb",
    ),
    "roboflow-coral-classification-copy-changed-v13i": (
        "roboflow-coral-classification-copy-changed-v13i/v13i-image-labels",
        "roboflow-bleaching-condition-hb",
    ),
    "roboflow-coral-reef-bleach-detection-v2i": (
        "roboflow-coral-reef-bleach-detection-v2i/v2i-image-labels",
        "roboflow-bleaching-condition-hb",
    ),
    "roboflow-coral-reef-classification-v3i": (
        "roboflow-coral-reef-classification-v3i/v3i-image-labels",
        "roboflow-bleaching-condition-hu",
    ),
}
# `loader.params.mask_values` in registry/sources/reef-support-own.yaml (WS-D S31)
MASKS = {
    "reef-support-benthic-own": (
        "reef-support-benthic-own/2026-09-24",
        "reef-support-labelbox",
        {1: "Hard Coral", 2: "Soft Coral"},
    ),
    "reef-support-bleaching": (
        "reef-support-bleaching/2026-09-24",
        "reef-support-bleaching-condition",
        {1: "bleached", 2: "non_bleached"},
    ),
}
# CoralSCOP masks are class-agnostic: every mask file is one "coral" annotation.
AGNOSTIC = {
    "coralscop-masks-rs": (
        "coralscop-masks-rs/2026-09-23-3e8612678469",
        "coralscop-masks-rs",
        "coral",
    )
}


def _get(url: str, tries: int = 4) -> bytes:
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return resp.read()
        except (TimeoutError, OSError):
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


def _mask_keys(tree: str, cache: Path) -> list[str]:
    path = cache / tree.split("/")[0] / "CHECKSUMS.sha256"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(BASE + tree + "/CHECKSUMS.sha256"))
    keys = [line.split()[-1] for line in path.read_text().splitlines() if line.strip()]
    return [k for k in keys if k.startswith("labels/masks/") and k.endswith(".png")]


def _mask_values(url: str) -> list[int]:
    import numpy as np
    from PIL import Image

    arr = np.asarray(Image.open(io.BytesIO(_get(url))))
    return [int(v) for v in np.unique(arr)]


def _resumable_mask_values(tree: str, keys: list[str], log: Path) -> dict[str, list[int]]:
    """``{key: distinct pixel values}``, one JSON line per mask so a timed-out run resumes."""
    have: dict[str, list[int]] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            row = json.loads(line)
            have[row["key"]] = row["values"]

    def one(key: str) -> tuple[str, list[int] | None]:
        try:
            return key, _mask_values(f"{BASE}{tree}/{key}")
        except OSError:
            return key, None

    todo = [k for k in keys if k not in have]
    with log.open("a") as fh, ThreadPoolExecutor(4) as pool:
        for key, values in pool.map(one, todo):
            if values is not None:
                have[key] = values
                fh.write(json.dumps({"key": key, "values": values}) + "\n")
                fh.flush()
    missing = [k for k in keys if k not in have]
    if missing:
        sys.exit(f"{tree}: {len(missing)} masks timed out; re-run to resume")
    return have


def main(cache: Path, repo: Path) -> None:
    import pyarrow.parquet as pq

    for sid, (tree, xw) in IMAGE_LABELS.items():
        local = cache / sid / "image_labels.parquet"
        if not local.exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(_get(BASE + tree + "/labels/image_labels.parquet"))
        counts = Counter(pq.read_table(local, columns=["label"])["label"].to_pylist())
        write(
            repo,
            sid,
            xw,
            counts,
            kind="annotations",
            origin=f"rs-storage-open sources/{tree}/labels/image_labels.parquet (anonymous GET)",
            note="one annotation = one image_labels.parquet row (whole-image label)",
        )
    for sid, (tree, xw, values) in MASKS.items():
        done = cache / sid / "mask_value_counts.json"
        if not done.exists():
            keys = _mask_keys(tree, cache)
            per_mask = _resumable_mask_values(tree, keys, cache / sid / "mask_values.jsonl")
            seen = Counter(v for vals in per_mask.values() for v in vals)
            done.write_text(json.dumps({"masks": len(keys), "value_masks": seen}))
        stats = json.loads(done.read_text())
        extra = sorted(int(v) for v in stats["value_masks"] if int(v) not in values and v != "0")
        if extra:
            sys.exit(f"{sid}: undeclared mask values {extra}")
        counts = {name: stats["value_masks"].get(str(v), 0) for v, name in values.items()}
        write(
            repo,
            sid,
            xw,
            counts,
            kind="annotations",
            origin=f"rs-storage-open sources/{tree}/labels/masks/*.png ({stats['masks']} masks, "
            "keys from CHECKSUMS.sha256, anonymous GET, streamed, not stored)",
            note="one annotation = one mask containing the class value; "
            "0=unlabelled is not a label",
        )
    for sid, (tree, xw, label) in AGNOSTIC.items():
        n = len(_mask_keys(tree, cache))
        write(
            repo,
            sid,
            xw,
            {label: n},
            kind="annotations",
            origin=f"rs-storage-open sources/{tree}/CHECKSUMS.sha256 (labels/masks/*.png keys)",
            note="class-agnostic instance masks: one annotation = one mask file",
        )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
