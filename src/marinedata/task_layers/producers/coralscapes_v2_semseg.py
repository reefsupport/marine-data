"""Coralscapes V2 semseg producer: the 95-class sibling of ``coralscapes_semseg``.

Same contract as the V1 producer: read mask files only (never image pixels), write one row per
image ``sha256, source_id, label_origin, mask_key, class_counts`` (JSON, native label -> pixel
count, D-Z2) to ``data/_tasklabels/coralscapes-v2/semseg.parquet``. What differs from V1:

* the staged tree is ``sources/coralscapes-v2/2.0/``: ``images/default/<stem>_leftImg8bit.png`` and
  ``labels/masks/default/<stem>_leftImg8bit.png``. V2 masks are mode ``L`` uint8, not palette PNGs;
* pixel value ``0`` is the upstream "unlabelled" value (up to 19 % of a frame) and is dropped before
  counting, like V1's index 0; values 1-95 are the dataset's own ids. ``ID_TO_LABEL`` is the
  dataset's published ``id2label.json`` at the pinned revision (README table identical), not a
  derivation; tests check it against ``registry/schemas/coralscapes-95.yaml`` and, once the
  ``coralscapes-v2`` label-scheme entry exists, against ``registry/label-schemes/sources.yaml``;
* nothing here maps a label to a taxon. The ``coralscapes-95`` crosswalk is resolved at build time
  through the registry (``MaskSource.crosswalk_id`` -> ``masks_table.staged_mask_rows``), exactly
  like V1's ``coralscapes-39``; :func:`mask_source` is that ``MASK_SOURCES`` entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from marinedata.checksums import file_digest
from marinedata.tables import _require_pyarrow

if TYPE_CHECKING:
    from ..masks_table import MaskSource

SOURCE_ID = "coralscapes-v2"
SOURCE_VERSION = "2.0"
TASK_ID = "semseg"
CROSSWALK_ID = "coralscapes-95"
LABEL_SET = "coralscapes-95"
HF_REVISION = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"  # josauder/coralscapesV2 (card 2026-09-15)

IGNORE_INDEX = 0
IMAGES_DIR = "images"
MASKS_DIR = "labels/masks"

ID_TO_LABEL: dict[int, str] = {
    1: "acanthaster planci",
    2: "acropora alive",
    3: "acropora bleached",
    4: "acropora dead",
    5: "algae covered brain coral",
    6: "algae covered branching coral",
    7: "algae covered lobophylliidae",
    8: "algae covered massive coral",
    9: "algae covered pocillopora",
    10: "algae covered porites",
    11: "algae covered substrate",
    12: "algae covered table acropora",
    13: "anemone",
    14: "background",
    15: "brain coral alive",
    16: "brain coral bleached",
    17: "brain coral dead",
    18: "branching coral alive",
    19: "branching coral bleached",
    20: "branching coral dead",
    21: "cirrhipathes",
    22: "dark",
    23: "dead giant clam",
    24: "feather worm",
    25: "fish",
    26: "fungiidae alive",
    27: "fungiidae bleached",
    28: "fungiidae dead",
    29: "galaxea alive",
    30: "galaxea bleached",
    31: "galaxea dead",
    32: "giant clam",
    33: "goniopora alive",
    34: "goniopora dead",
    35: "hard coral alive",
    36: "hard coral bleached",
    37: "hard coral dead",
    38: "hard substrate",
    39: "human",
    40: "human-made structure",
    41: "leather coral alive",
    42: "lobophylliidae alive",
    43: "lobophylliidae bleached",
    44: "lobophylliidae dead",
    45: "massive alive",
    46: "massive bleached",
    47: "massive dead",
    48: "meandering coral alive",
    49: "meandering coral bleached",
    50: "meandering coral dead",
    51: "millepora",
    52: "millepora bleached",
    53: "millepora dead",
    54: "nephtheidae alive",
    55: "other animal",
    56: "other coral alive",
    57: "other coral bleached",
    58: "other coral dead",
    59: "pavona alive",
    60: "pavona bleached",
    61: "pavona dead",
    62: "pocillopora alive",
    63: "pocillopora bleached",
    64: "pocillopora dead",
    65: "porites alive",
    66: "porites bleached",
    67: "porites dead",
    68: "rubble",
    69: "sand",
    70: "sea cucumber",
    71: "sea urchin",
    72: "seagrass",
    73: "seriatopora alive",
    74: "seriatopora bleached",
    75: "seriatopora dead",
    76: "soft coral (malacalcyonacea) alive",
    77: "soft coral (malacalcyonacea) bleached",
    78: "sponge",
    79: "stylophora alive",
    80: "stylophora bleached",
    81: "stylophora dead",
    82: "table acropora alive",
    83: "table acropora bleached",
    84: "table acropora dead",
    85: "thin plate/encrusting coral alive",
    86: "thin plate/encrusting coral bleached",
    87: "thin plate/encrusting coral dead",
    88: "transect line",
    89: "transect tools",
    90: "trash",
    91: "turbinaria",
    92: "turbinaria dead",
    93: "water column",
    94: "water surface",
    95: "xeniidae alive",
}


def mask_class_counts(mask_path: Path) -> dict[str, int]:
    """Native-label -> pixel-count histogram for one mask, ignore index dropped."""
    from PIL import Image

    with Image.open(mask_path) as im:
        im = im if im.mode in ("L", "P") else im.convert("L")
        histogram = im.histogram()

    counts: dict[str, int] = {}
    for index, n in enumerate(histogram):
        if n == 0 or index == IGNORE_INDEX:
            continue
        label = ID_TO_LABEL.get(index)
        if label is None:
            continue  # an index outside the published 1-95 table; not a known class
        counts[label] = counts.get(label, 0) + n
    return counts


def produce_semseg(
    root: str | Path,
    out_path: str | Path,
    *,
    images_dir: str = IMAGES_DIR,
    masks_dir: str = MASKS_DIR,
) -> int:
    """Write ``data/_tasklabels/coralscapes-v2/semseg.parquet``. Returns the row count."""
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
        counts = mask_class_counts(mask)
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


def mask_source() -> MaskSource:
    """The ``masks_table.MASK_SOURCES`` entry of V2 (imported lazily: ``masks_table`` is the
    module that will import :data:`ID_TO_LABEL` from here)."""
    from ..masks_table import MaskSource

    return MaskSource(
        SOURCE_ID,
        SOURCE_VERSION,
        CROSSWALK_ID,
        ID_TO_LABEL,
        IGNORE_INDEX,
        "human",
        "Apache-2.0",
        "open",
        label_set=LABEL_SET,
    )
