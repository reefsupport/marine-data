"""Fetch images from an HF build by sha256 and tile them into captioned contact sheets.

Used for the model audit of confident-learning flags and for the expert-audit sheet: a
reviewer (human or model) reads one sheet of ``cols × rows`` tiles instead of N files.
The HF ``images`` shards use 100-row row groups, so one image costs one row-group read.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Sequence
from pathlib import Path


def build_index(images_dir: Path) -> dict[str, tuple[Path, int, int]]:
    """sha256 → (shard, row group, row within the group) over ``images/*.parquet``."""
    import pyarrow.parquet as pq

    index: dict[str, tuple[Path, int, int]] = {}
    for shard in sorted(Path(images_dir).glob("*.parquet")):
        pf = pq.ParquetFile(shard)
        for rg in range(pf.num_row_groups):
            shas = pf.read_row_group(rg, columns=["image_sha256"]).column(0).to_pylist()
            for i, sha in enumerate(shas):
                index[sha] = (shard, rg, i)
    return index


def fetch(index: dict[str, tuple[Path, int, int]], shas: Iterable[str]) -> dict[str, bytes]:
    """Image bytes for each sha, grouped so each row group is read once."""
    import pyarrow.parquet as pq

    want: dict[tuple[Path, int], list[tuple[int, str]]] = {}
    for sha in shas:
        shard, rg, row = index[sha]
        want.setdefault((shard, rg), []).append((row, sha))
    out: dict[str, bytes] = {}
    for (shard, rg), rows in want.items():
        col = pq.ParquetFile(shard).read_row_group(rg, columns=["image"]).column(0)
        blobs = col.combine_chunks().field("bytes").to_pylist()
        for row, sha in rows:
            out[sha] = blobs[row]
    return out


def contact_sheet(
    tiles: Sequence[tuple[bytes, str]], out: Path, *, cols: int = 5, tile: int = 256
) -> Path:
    """Tile ``(image bytes, caption)`` pairs into one JPEG, captions under each tile."""
    from PIL import Image, ImageDraw

    cap_h = 30
    rows = max(1, -(-len(tiles) // cols))
    sheet = Image.new("RGB", (cols * tile, rows * (tile + cap_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for k, (blob, caption) in enumerate(tiles):
        with Image.open(io.BytesIO(blob)) as im:
            thumb = im.convert("RGB")
        thumb.thumbnail((tile, tile))
        x, y = (k % cols) * tile, (k // cols) * (tile + cap_h)
        sheet.paste(thumb, (x + (tile - thumb.width) // 2, y + (tile - thumb.height) // 2))
        for line_no, line in enumerate(caption.split("\n")[:2]):
            draw.text((x + 3, y + tile + 2 + 13 * line_no), line[:42], fill="black")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=85)
    return out
