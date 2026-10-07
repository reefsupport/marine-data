"""MarineInst20M annotation overlay (WP-6e-B, D-AB).

The GitHub release ships per-image COCO JSONs (RLE masks, boxes, category names, instance
captions) in 28 zips; the images live upstream. :func:`overlay_rows` turns one zip into
overlay rows keyed by the upstream image path; :func:`match_rates` joins those rows to our
staged sources by upstream path, then by file basename."""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

REPO = "zhengziqiang/MarineInst20M"
RAW = f"https://raw.githubusercontent.com/{REPO}/HEAD/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/HEAD?recursive=1"

# MarineInst20M folder -> our source ids (registry/ingest-specs), where one exists.
OUR_SOURCES: Mapping[str, tuple[str, ...]] = {
    "DeepFish": ("deepfish",),
    "OZFish": ("ozfish",),
    "UIIS": ("uiis", "uiis10k"),
    "URPC": ("urpc",),
    "Wildfish++": ("wildfish",),
    "FathomNet": ("fathomnet",),
    "EOL": ("treeoflife-10m",),
}


def norm_basename(path: str) -> str:
    """Case-folded file stem: the join key when upstream paths differ in folders/suffix."""
    return PurePosixPath(str(path).split("#")[-1]).stem.lower()


def _captions(obj: Mapping[str, Any]) -> list[str]:
    return [str(v) for k, v in obj.items() if "caption" in k.lower() and v]


def overlay_rows(zip_path: Path, dataset: str, group: str) -> Iterator[dict[str, Any]]:
    with zipfile.ZipFile(zip_path) as zf:
        for member in sorted(n for n in zf.namelist() if n.endswith(".json")):
            try:
                doc = json.loads(zf.read(member))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(doc, dict):
                continue
            by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
            for ann in doc.get("annotations") or []:
                seg = ann.get("segmentation") or {}
                by_image[ann.get("image_id")].append(
                    {
                        "rle": seg if isinstance(seg, dict) else None,
                        "bbox": ann.get("bbox"),
                        "category": ann.get("category_name"),
                        "captions": _captions(ann),
                        "score": ann.get("score"),
                    }
                )
            for img in doc.get("images") or []:
                anns = by_image.get(img.get("id"), [])
                fname = str(img.get("file_name") or PurePosixPath(member).stem)
                yield {
                    "group": group,
                    "upstream_dataset": dataset,
                    "json_member": member,
                    "upstream_path": fname,
                    "basename": norm_basename(fname),
                    "width": img.get("width"),
                    "height": img.get("height"),
                    "n_instances": len(anns),
                    "categories": json.dumps(
                        sorted({a["category"] for a in anns if a["category"]})
                    ),
                    "captions": json.dumps(
                        _captions(img) + [c for a in anns for c in a["captions"]]
                    ),
                    "annotations": json.dumps(anns, separators=(",", ":")),
                }


def match_rates(
    overlay: Iterable[Mapping[str, Any]], ours: Mapping[str, Iterable[str]]
) -> dict[str, dict[str, Any]]:
    """Per our source id: overlay images of its MarineInst dataset matched by exact
    upstream path, else by basename, against that source's staged ``upstream_id``s."""
    by_ds: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in overlay:
        by_ds[str(row["upstream_dataset"])].append(row)
    out: dict[str, dict[str, Any]] = {}
    for ds, ids in OUR_SOURCES.items():
        rows = by_ds.get(ds, [])
        for sid in ids:
            up = [str(u) for u in ours.get(sid, ())]
            if not up:
                out[sid] = {
                    "marineinst": ds,
                    "overlay_images": len(rows),
                    "ours": 0,
                    "status": "not staged",
                }
                continue
            paths = {u.split("#")[-1] for u in up}
            bases = {norm_basename(u) for u in up}
            by_path = sum(1 for r in rows if r["upstream_path"] in paths)
            by_base = sum(
                1 for r in rows if r["upstream_path"] not in paths and r["basename"] in bases
            )
            n = len(rows)
            out[sid] = {
                "marineinst": ds,
                "overlay_images": n,
                "ours": len(up),
                "matched_path": by_path,
                "matched_basename": by_base,
                "match_pct": round(100 * (by_path + by_base) / n, 2) if n else 0.0,
            }
    return out
