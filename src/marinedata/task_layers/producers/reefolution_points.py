"""Reefolution points producer (WP-8c) — D-Z2 ``points`` payload.

Reads the staged tree's ``metadata.parquet`` (one row per image: stem, partition, width,
height) and ``labels/points.parquet`` (one row per CoralNet-style point: stem, partition,
row, col, native ``label``), and writes one row per point:
``sha256, source_id, label_origin, native_label, x, y`` with ``x``/``y`` normalised to
0-1 by the image's own width/height (D-Z2: "x, y (0-1)").

``label_origin`` is always ``"human"`` — Reefolution's export is a human CoralNet
annotation, never a model prediction.
"""

from __future__ import annotations

from pathlib import Path

from marinedata.checksums import file_digest
from marinedata.tables import _require_pyarrow

SOURCE_ID = "reefolution"
TASK_ID = "points"


def _image_path(root: Path, partition: str, stem: str) -> Path | None:
    """The staged image for one point row, or ``None`` if the local cache is a partial
    fetch missing it — skipped, not raised: a producer runs against whatever a working
    tree happens to have on disk today, unlike ``release.py``'s enumerator, which
    requires a complete, pinned staged tree."""
    images_dir = root / "images" / partition
    matches = sorted(p for p in images_dir.glob(f"{stem}.*") if p.is_file())
    return matches[0] if matches else None


def produce_points(root: str | Path, out_path: str | Path) -> int:
    """Write ``data/_tasklabels/reefolution/points.parquet``. Returns the row count.

    ``root`` is the staged Reefolution tree (``metadata.parquet`` + ``labels/points.parquet``
    + ``images/<partition>/``); ``out_path`` is the D-Z2 parquet to write.
    """
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    metadata = {
        (rec["partition"], rec["stem"]): rec
        for rec in pq.read_table(root / "metadata.parquet").to_pylist()
    }
    points = pq.read_table(root / "labels" / "points.parquet").to_pylist()

    sha_cache: dict[tuple[str, str], str | None] = {}
    rows: list[dict] = []
    for point in points:
        key = (point["partition"], point["stem"])
        image_meta = metadata.get(key)
        if image_meta is None:
            raise ValueError(f"{SOURCE_ID}: point references unknown image {key!r}")
        sha256 = sha_cache.get(key, "")
        if sha256 == "":
            image_path = _image_path(root, point["partition"], point["stem"])
            sha256 = file_digest(image_path) if image_path is not None else None
            sha_cache[key] = sha256
        if sha256 is None:
            continue  # image missing from a partial local cache
        width, height = image_meta["width"], image_meta["height"]
        rows.append(
            {
                "sha256": sha256,
                "source_id": SOURCE_ID,
                "label_origin": "human",
                "native_label": point["label"],
                "x": point["col"] / width,
                "y": point["row"] / height,
            }
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    return len(rows)
