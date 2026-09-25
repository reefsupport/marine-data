"""Bounded imos-auv manifest builder (WP-6e-B): anonymous s3://imos-data/IMOS/AUV/.

Per dive: the ``track_files/*_latlong.csv`` names every stereo pair with lat/lon/depth,
so no image-prefix listing is needed; the LEFT camera (``*_LC16``) still is taken from
the dive's ``*_gtif/`` folder. Stops at ``--deadline`` seconds (partial, resumable by
re-running); campaigns x depth bands are capped by an exact hash bottom-k."""

from __future__ import annotations

import argparse
import csv
import io
import json
import time
from pathlib import Path

import boto3
import pyarrow as pa
import pyarrow.parquet as pq
from botocore import UNSIGNED
from botocore.config import Config

from marinedata.subset_filter import SubsetFilter, select_bottom_k

BUCKET, ROOT = "imos-data", "IMOS/AUV/"
URL = f"https://{BUCKET}.s3.ap-southeast-2.amazonaws.com/"
SKIP = {"AUV_articles", "auv_viewer_data"}


def depth_band(depth: float | None, width: int = 20) -> str:
    if depth is None:
        return "unknown"
    lo = int(depth // width) * width
    return f"{lo}-{lo + width}m"


def parse_track(text: str) -> list[dict]:
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines) if ln.startswith("year,")), None)
    if start is None:
        return []
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))


def track_rows(campaign: str, dive_prefix: str, gtif: str, track: list[dict]) -> list[dict]:
    out = []
    for r in track:
        left = (r.get("leftimage") or "").strip()
        if not left:
            continue
        try:
            depth = float(r["depth"])
            lat, lon = float(r["latitude"]), float(r["longitude"])
            y, mo, d = int(r["year"]), int(r["month"]), int(r["day"])
            h, mi, sec = int(r["hour"]), int(r["minute"]), float(r["second"])
            ts = f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{sec:06.3f}Z"
        except (KeyError, ValueError):
            continue
        key = f"{gtif}{left.rsplit('.', 1)[0]}.tif"
        out.append(
            {
                "key": key,
                "url": URL + key,
                "lat": lat,
                "lon": lon,
                "depth_m": depth,
                "capture_datetime": ts,
                "campaign": campaign,
                "dive": dive_prefix.rstrip("/").rsplit("/", 1)[-1],
                "depth_band": depth_band(depth),
                "pair_right": (r.get("rightimage") or "").strip(),
                "altitude_m": r.get("altitude"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--deadline", type=float, default=840)
    ap.add_argument("--cap", type=int, default=5000)
    a = ap.parse_args()
    t0 = time.time()
    s3 = boto3.client("s3", region_name="ap-southeast-2", config=Config(signature_version=UNSIGNED))

    def dirs(prefix: str) -> tuple[list[str], list[str]]:
        r = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
        return (
            [c["Prefix"] for c in r.get("CommonPrefixes", [])],
            [o["Key"] for o in r.get("Contents", [])],
        )

    rows: list[dict] = []
    stats = {"campaigns": 0, "dives": 0, "dives_no_track": 0, "complete": False}
    camps = [c for c in dirs(ROOT)[0] if c[len(ROOT) : -1] not in SKIP]
    for camp in camps:
        stats["campaigns"] += 1
        for dive in dirs(camp)[0]:
            if time.time() - t0 > a.deadline:
                break
            sub, _ = dirs(dive)
            gtif = next((s for s in sub if s.endswith("_gtif/")), None)
            tf = next((s for s in sub if s.endswith("track_files/")), None)
            csvs = [k for k in dirs(tf)[1] if k.endswith("_latlong.csv")] if tf else []
            if not gtif or not csvs:
                stats["dives_no_track"] += 1
                continue
            body = (
                s3.get_object(Bucket=BUCKET, Key=csvs[0])["Body"].read().decode("utf-8", "replace")
            )
            rows += track_rows(camp[len(ROOT) : -1], dive, gtif, parse_track(body))
            stats["dives"] += 1
        else:
            continue
        break
    else:
        stats["complete"] = True
    flt = SubsetFilter(stratify=("campaign", "depth_band"), cap=a.cap)
    keep = set(select_bottom_k(((r["key"], r) for r in rows), flt))
    sel = sorted((r for r in rows if r["key"] in keep), key=lambda r: r["key"])
    sizes = [
        s3.head_object(Bucket=BUCKET, Key=r["key"])["ContentLength"]
        for r in sel[:: max(1, len(sel) // 25)][:25]
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sel), a.out, compression="zstd")
    stats.update(
        left_images_seen=len(rows),
        selected=len(sel),
        sampled_sizes=len(sizes),
        mean_bytes=int(sum(sizes) / len(sizes)) if sizes else None,
        est_selected_bytes=int(sum(sizes) / len(sizes) * len(sel)) if sizes else None,
        seconds=round(time.time() - t0),
    )
    a.out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
