"""64-bit DCT perceptual hash and the image scan the dedup and splits gates share.

``imagehash`` is not in ``uv.lock``, so the hash is a short PIL + numpy implementation of the
same recipe: grey, 32x32 LANCZOS, 2-D DCT-II, top-left 8x8 block, bit = coefficient above the
block median. The DCT is rounded to 6 decimals before the comparison so a last-ulp BLAS
difference cannot flip a bit on another machine. ``<build>/phash.parquet`` stores one row per
embedded image (``image_sha256, config, split, source_id, phash`` as 16 hex digits) for the
gates, P1.9 and P2.3."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .shards import config_shards, iter_rows, shard_columns, split_counts

ALGORITHM = "dct-phash-64"
PHASH_FILE = "phash.parquet"
_SIZE = 32


@lru_cache(maxsize=1)
def _dct() -> np.ndarray:
    k, n = np.arange(_SIZE)[:, None], np.arange(_SIZE)[None, :]
    matrix = np.cos(np.pi * (2 * n + 1) * k / (2 * _SIZE)) * np.sqrt(2.0 / _SIZE)
    matrix[0] /= np.sqrt(2.0)
    return matrix


def phash(image) -> int:
    from PIL import Image

    grey = image.convert("L").resize((_SIZE, _SIZE), Image.Resampling.LANCZOS)
    low = np.round((_dct() @ np.asarray(grey, np.float64) @ _dct().T)[:8, :8], 6)
    return int("".join("1" if bit else "0" for bit in (low > np.median(low)).ravel()), 2)


def _popcount(values: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    return np.unpackbits(values.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def near_pairs(hashes: list[int], max_bits: int) -> Iterator[tuple[int, int, int]]:
    """``(i, j, distance)``, ``i < j``, for every pair within ``max_bits``. Pigeonhole: cut the
    64 bits into ``max_bits + 1`` ranges; two hashes within ``max_bits`` agree on one range, so
    only hashes sharing a range value are compared."""
    if len(hashes) < 2:
        return
    values = np.asarray(hashes, dtype=np.uint64)
    edges = np.linspace(0, 64, min(max_bits, 63) + 2).astype(int)
    seen: set[tuple[int, int]] = set()
    for low, high in zip(edges[:-1], edges[1:], strict=True):  # noqa: RUF007
        keys = (values >> np.uint64(low)) & np.uint64((1 << int(high - low)) - 1)
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        for start, end in zip(starts, np.r_[starts[1:], len(order)], strict=True):
            members = order[start:end]
            for a in range(len(members) - 1):
                dist = _popcount(values[members[a]] ^ values[members[a + 1 :]])
                for b in np.flatnonzero(dist <= max_bits):
                    pair = sorted((int(members[a]), int(members[a + 1 + b])))
                    if (pair[0], pair[1]) not in seen:
                        seen.add((pair[0], pair[1]))
                        yield pair[0], pair[1], int(dist[b])


@dataclass(frozen=True)
class ImageRecord:
    config: str
    split: str
    source: str
    image_sha256: str
    digest: str  # sha256 of the embedded bytes
    phash: int | None  # None: the bytes do not decode
    where: str
    exif_transposed: bool = False  # the stored bytes were re-encoded after the EXIF rotation
    row_size: tuple[int, int] | None = None  # the row's width/height columns, when it has them
    decoded_size: tuple[int, int] | None = None  # width/height of the embedded bytes


def image_configs(build_dir: Path) -> dict[str, list[Path]]:
    """Configs that embed pixels: ``images`` alone when it exists (pixels are stored once)."""
    found = {c: f for c, f in config_shards(build_dir).items() if "image" in shard_columns(f[0])}
    return {"images": found["images"]} if "images" in found else found


def _row_size(row: dict) -> tuple[int, int] | None:
    width, height = row.get("width"), row.get("height")
    return (int(width), int(height)) if width is not None and height is not None else None


def scan_images(build_dir: Path) -> list[ImageRecord]:
    from PIL import Image

    records = []
    for config, files in image_configs(build_dir).items():
        for path, index, split, row in iter_rows(
            files,
            ("image_sha256", "image", "source_id", "source", "exif_transposed", "width", "height"),
        ):
            data = (row.get("image") or {}).get("bytes")
            if not data:
                continue
            decoded = None
            try:
                image = Image.open(io.BytesIO(data))
                image.load()
                value, decoded = phash(image), image.size
            except (OSError, ValueError):
                value = None
            records.append(
                ImageRecord(
                    config,
                    split,
                    str(row.get("source_id") or row.get("source") or "?"),
                    str(row.get("image_sha256")),
                    hashlib.sha256(data).hexdigest(),
                    value,
                    f"{config}/{path.name}[{index}]",
                    bool(row.get("exif_transposed")),
                    _row_size(row),
                    decoded,
                )
            )
    return records


def write_table(build_dir: Path, records: list[ImageRecord]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = [r for r in records if r.phash is not None]
    table = pa.table(
        {
            "image_sha256": [r.image_sha256 for r in rows],
            "config": [r.config for r in rows],
            "split": [r.split for r in rows],
            "source_id": [r.source for r in rows],
            "phash": [f"{r.phash:016x}" for r in rows],
        }
    ).replace_schema_metadata({"marinedata_phash": json.dumps({"algorithm": ALGORITHM})})
    path = Path(build_dir) / PHASH_FILE
    pq.write_table(table, path)
    return path


def load_or_scan(build_dir: Path) -> list[ImageRecord]:
    """Records from ``phash.parquet`` when it covers every image row of the build, else a scan
    (which rewrites it). ``digest`` is empty for rows read back from the table."""
    import pyarrow.parquet as pq

    path = Path(build_dir) / PHASH_FILE
    shards = image_configs(build_dir)
    expected = sum(sum(split_counts(files).values()) for files in shards.values())
    if path.is_file() and pq.ParquetFile(path).metadata.num_rows == expected:
        return [
            ImageRecord(
                r["config"],
                r["split"],
                r["source_id"],
                r["image_sha256"],
                "",
                int(r["phash"], 16),
                "phash.parquet",
            )
            for r in pq.read_table(path).to_pylist()
        ]
    records = scan_images(build_dir)
    if records:
        write_table(build_dir, records)
    return records
