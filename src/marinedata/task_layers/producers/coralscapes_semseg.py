"""Coralscapes semseg producer (WP-8c) — D-Z2 ``semseg`` payload.

Reads mask files only (never the images — the brief's Do-Not, and D-Z's "never the
images" for every source but Coralseg) via the existing ``images``/``masks`` pairing
``ImageMaskPairLoader`` already uses, decodes each palette-indexed mask's pixel
histogram, and writes one row per image: ``sha256, source_id, label_origin, mask_key,
class_counts`` (JSON, D-Z2). ``sha256`` still has to key the row against the image the
mask pairs with — the brief keys ``semseg`` by the *image* it annotates, matching every
other config; this producer trusts the image file exists (it does, staged already for
Coralscapes) but only ever opens the mask's bytes for pixels.

Index-to-label mapping: Coralscapes' raw mask pixel values are the upstream numeric
class ids 0-39 (registry ``verification note``: "observed indices 0-39", ``raster_ignore_value:
0``). That table is not recorded anywhere in this repo (no ``ANNOTATIONS.json``, no
``external/deepreefmap`` submodule checked out here) — ``_ID_TO_LABEL`` below is
transcribed from the dataset's own published class table
(https://huggingface.co/datasets/EPFL-ECEO/coralscapes, "Class Index" column, read
2026-09-25) rather than invented, and every name matches an existing alias in
``registry/schemas/coralscapes-39.yaml``/``registry/crosswalks/coralscapes-39.yaml``
exactly. Index 0 (not in the published table) is the upstream ignore/unlabelled value
and is dropped before counting (D-Y: "ignore/void pixels excluded").
"""

from __future__ import annotations

from pathlib import Path

from marinedata.checksums import file_digest
from marinedata.tables import _require_pyarrow

SOURCE_ID = "coralscapes"
TASK_ID = "semseg"

IGNORE_INDEX = 0

_ID_TO_LABEL: dict[int, str] = {
    1: "seagrass",
    2: "trash",
    3: "other coral dead",
    4: "other coral bleached",
    5: "sand",
    6: "other coral alive",
    7: "human",
    8: "transect tools",
    9: "fish",
    10: "algae covered substrate",
    11: "other animal",
    12: "unknown hard substrate",
    13: "background",
    14: "dark",
    15: "transect line",
    16: "massive/meandering bleached",
    17: "massive/meandering alive",
    18: "rubble",
    19: "branching bleached",
    20: "branching dead",
    21: "millepora",
    22: "branching alive",
    23: "massive/meandering dead",
    24: "clam",
    25: "acropora alive",
    26: "sea cucumber",
    27: "turbinaria",
    28: "table acropora alive",
    29: "sponge",
    30: "anemone",
    31: "pocillopora alive",
    32: "table acropora dead",
    33: "meandering bleached",
    34: "stylophora alive",
    35: "sea urchin",
    36: "meandering alive",
    37: "meandering dead",
    38: "crown of thorn",
    39: "dead clam",
}


def _mask_class_counts(mask_path: Path) -> dict[str, int]:
    """Native-label -> pixel-count histogram for one mask, ignore index dropped."""
    from PIL import Image

    with Image.open(mask_path) as im:
        im = im.convert("P") if im.mode != "P" else im
        histogram = im.histogram()

    counts: dict[str, int] = {}
    for index, n in enumerate(histogram):
        if n == 0 or index == IGNORE_INDEX:
            continue
        label = _ID_TO_LABEL.get(index)
        if label is None:
            continue  # an index outside the published 1-39 table; not a known class
        counts[label] = counts.get(label, 0) + n
    return counts


def produce_semseg(
    root: str | Path, out_path: str | Path, *, images_dir: str = "images", masks_dir: str = "masks"
) -> int:
    """Write ``data/_tasklabels/coralscapes/semseg.parquet``. Returns the row count."""
    _require_pyarrow()
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    images = root / images_dir
    masks = root / masks_dir
    by_stem = {p.stem: p for p in masks.rglob("*") if p.is_file()}

    rows: list[dict] = []
    for image in sorted(p for p in images.rglob("*") if p.is_file()):
        mask = by_stem.get(image.stem)
        if mask is None:
            continue
        counts = _mask_class_counts(mask)
        if not counts:
            continue
        rows.append(
            {
                "sha256": file_digest(image),
                "source_id": SOURCE_ID,
                "label_origin": "human",
                "mask_key": str(mask.relative_to(root)),
                "class_counts": json.dumps(counts, sort_keys=True),
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return len(rows)
