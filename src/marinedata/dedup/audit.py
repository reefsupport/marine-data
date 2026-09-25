"""Precision audit: a stratified sample of confirmed pairs, thumbnails, contact sheets, a TSV.

Strata = distance band x low-texture. Bands: ``exact`` (byte or pixel identical),
``d0-2`` / ``d3-5`` / ``d6-8`` (best-orientation dHash), ``embed-only`` (dHash > 8,
confirmed by the SSCD cosine). The auditor fills ``verdict`` (dup | not-dup) per row.
"""

from __future__ import annotations

import csv
import io
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from .corpus import Item, read_payloads

BANDS = ("exact", "d0-2", "d3-5", "d6-8", "embed-only")
COLUMNS = [
    "pair_id",
    "stratum",
    "sha_a",
    "sha_b",
    "source_a",
    "source_b",
    "kind",
    "d_dhash",
    "d_phash",
    "cos",
    "lowtex",
    "thumb_a",
    "thumb_b",
    "sheet",
    "verdict",
    "note",
]


def band_of(row: dict) -> str:  # type: ignore[type-arg]
    if row["exact"] or row["pixel"]:
        return "exact"
    d = row["d_dhash"]
    return "d0-2" if d <= 2 else "d3-5" if d <= 5 else "d6-8" if d <= 8 else "embed-only"


def sample_pairs(out_dir: Path, n_total: int = 120, seed: int = 0) -> list[dict]:  # type: ignore[type-arg]
    import pyarrow.parquet as pq

    rows = [r for r in pq.read_table(out_dir / "pairs.parquet").to_pylist() if r["confirmed"]]
    strata: dict[str, list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for r in rows:
        strata[f"{band_of(r)}|lt={int(r['lowtex'])}"].append(r)
    rng = random.Random(seed)
    for v in strata.values():
        rng.shuffle(v)
    picked: list[dict] = []  # type: ignore[type-arg]
    quota = max(1, n_total // max(1, len(strata)))
    for key in sorted(strata):
        picked += [dict(r, stratum=key) for r in strata[key][:quota]]
    rest = [dict(r, stratum=k) for k in sorted(strata) for r in strata[k][quota:]]
    rng.shuffle(rest)
    picked += rest[: max(0, n_total - len(picked))]
    return picked


def _thumb(data: bytes, side: int = 220):  # type: ignore[no-untyped-def]
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side))
        return img


def render(pairs: list[dict], items: list[Item], dest: Path, per_sheet: int = 20) -> Path:  # type: ignore[type-arg]
    """Write thumbnails, sheets and ``audit.tsv`` under ``dest``; returns the TSV path."""
    from PIL import Image, ImageDraw

    by_sha = {}
    for it in items:
        by_sha.setdefault(it.sha256 or "", it)
    need = {s for p in pairs for s in (p["sha_a"], p["sha_b"])}
    by_ref: dict[tuple, list[str]] = defaultdict(list)  # type: ignore[type-arg]
    for s in need:
        ref = by_sha[s].payload
        by_ref[ref[:2] if isinstance(ref, tuple) else ("file", s)].append(s)
    thumbs = {}
    for shas in by_ref.values():
        for s, data in zip(shas, read_payloads([by_sha[s].payload for s in shas]), strict=True):
            thumbs[s] = _thumb(data)
    (dest / "thumbs").mkdir(parents=True, exist_ok=True)
    for s, im in thumbs.items():
        im.save(dest / "thumbs" / f"{s[:16]}.jpg", quality=85)
    cell_w, cell_h, label_h = 230, 230, 34
    cols = 2
    for start in range(0, len(pairs), per_sheet):
        chunk = pairs[start : start + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * (2 * cell_w + 20), rows * (cell_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        name = f"sheet_{start // per_sheet:02d}.jpg"
        for k, p in enumerate(chunk):
            p["pair_id"] = f"P{start + k:03d}"
            p["sheet"] = name
            x0 = (k % cols) * (2 * cell_w + 20)
            y0 = (k // cols) * (cell_h + label_h)
            sheet.paste(thumbs[p["sha_a"]], (x0, y0 + label_h))
            sheet.paste(thumbs[p["sha_b"]], (x0 + cell_w, y0 + label_h))
            draw.text((x0 + 2, y0 + 2), f"{p['pair_id']} {p['stratum']} {p['kind']}", fill="black")
            draw.text(
                (x0 + 2, y0 + 16),
                f"dh={p['d_dhash']} ph={p['d_phash']} cos={p['cos']:.2f}",
                fill="black",
            )
        sheet.save(dest / name, quality=80)
    src = {}
    for it in items:
        src.setdefault(it.sha256, it.source)
    tsv = dest / "audit.tsv"
    with tsv.open("w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(COLUMNS)
        for p in pairs:
            w.writerow(
                [
                    p["pair_id"],
                    p["stratum"],
                    p["sha_a"],
                    p["sha_b"],
                    src.get(p["sha_a"]),
                    src.get(p["sha_b"]),
                    p["kind"],
                    p["d_dhash"],
                    p["d_phash"],
                    f"{p['cos']:.4f}",
                    int(p["lowtex"]),
                    str(dest / "thumbs" / f"{p['sha_a'][:16]}.jpg"),
                    str(dest / "thumbs" / f"{p['sha_b'][:16]}.jpg"),
                    p["sheet"],
                    "",
                    "",
                ]
            )
    return tsv


def score(tsv: Path) -> dict[str, object]:
    with tsv.open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    judged = [r for r in rows if r["verdict"] in ("dup", "not-dup")]
    ok = np.array([r["verdict"] == "dup" for r in judged])
    by: dict[str, list[bool]] = defaultdict(list)
    for r, v in zip(judged, ok.tolist(), strict=True):
        by[r["stratum"]].append(v)
    return {
        "n": len(judged),
        "precision": float(ok.mean()) if len(ok) else None,
        "by_stratum": {k: f"{sum(v)}/{len(v)}" for k, v in sorted(by.items())},
    }
