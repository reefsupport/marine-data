"""Resumable imos-auv manifest builder (WP-6e-B, D-AC in WP-6f): anonymous s3://imos-data/IMOS/AUV/.

Listing (resumable, run under nohup across passes):
  * per dive, ``track_files/*_latlong.csv`` names every stereo pair with lat/lon/depth; the LEFT
    camera (``*_LC16``) still is taken from the dive's ``*_gtif/`` folder. Each dive becomes one
    small parquet in ``<cache>/dives/<campaign>/<dive>.parquet`` (or a ``.none`` marker when it
    has no track/gtif); cached dives are skipped on the next pass.
  * Squidle+ (anonymous ``/api/media``) is scanned by id keyset for every IMOS-platform frame with
    at least one labelled point; pages land in ``<cache>/squidle/`` and resume from the max id.
Build (only from a COMPLETE cache — a partial manifest is never written), D-AC order:
  1. every Squidle+-annotated frame, uncapped;
  2. along-track thinning of the rest: keep frame ``seq % thin == 0`` of each dive's track;
  3. cap per campaign x 20 m depth band x MEOW ecoregion, the cap solved so the total <= target;
  4. within a cell, sha256(frame key) order — never listing order."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import hashlib
import io
import json
import statistics
import threading
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

BUCKET, ROOT = "imos-data", "IMOS/AUV/"
URL = f"https://{BUCKET}.s3.ap-southeast-2.amazonaws.com/"
SKIP = {"AUV_articles", "auv_viewer_data"}
SQUIDLE = "https://squidle.org/api/media"
# squidle.org answers 403 to the default Python-urllib agent (WP-6f pass 1); curl-like UA is fine
_UA = {
    "User-Agent": "reef-support-marinedata/1.0 (+https://reef.support)",
    "Accept": "application/json",
}
_IMOS_PLATFORM = {"name": "name", "op": "ilike", "val": "%IMOS%"}
_LABELLED_POINT = {"name": "label_id", "op": "is_not_null"}
SQUIDLE_FILTERS = [
    {
        "name": "deployment",
        "op": "has",
        "val": {"name": "platform", "op": "has", "val": _IMOS_PLATFORM},
    },
    {
        "name": "annotations",
        "op": "any",
        "val": {"name": "annotations", "op": "any", "val": _LABELLED_POINT},
    },
]


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
                "seq": len(out),
            }
        )
    return out


def stem(key: str) -> str:
    return key.rsplit("/", 1)[-1].rsplit(".", 1)[0]


def sha_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# --- MEOW ecoregion (point in polygon, no shapely) ---------------------------------------
def load_meow(path: Path) -> list[tuple[str, tuple[float, float, float, float], list]]:
    out = []
    for f in json.loads(path.read_text())["features"]:
        g = f["geometry"]
        polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        xs = [p[0] for poly in polys for p in poly[0]]
        ys = [p[1] for poly in polys for p in poly[0]]
        out.append((f["properties"]["ecoregion"], (min(xs), min(ys), max(xs), max(ys)), polys))
    return out


def _in_ring(x: float, y: float, ring: list) -> bool:
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = ring[i][0], ring[i][1], ring[j][0], ring[j][1]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def meow_region(lat: float, lon: float, meow: list) -> str:
    for name, (x0, y0, x1, y1), polys in meow:
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            for poly in polys:
                if _in_ring(lon, lat, poly[0]) and not any(_in_ring(lon, lat, h) for h in poly[1:]):
                    return name
    return "unassigned"


# --- D-AC selection (pure; unit-tested on a synthetic listing) ----------------------------
def cell_of(r: Mapping) -> tuple[str, str, str]:
    return (r["campaign"], r["depth_band"], r["region"])


def solve_cap(cells: Mapping[tuple, tuple[int, int]], target: int) -> int:
    """Largest per-cell cap C with sum_c a_c + min(t_c, max(0, C - a_c)) <= target.

    ``cells`` maps cell -> (annotated, thinned candidates). Annotated frames are never dropped
    and count against their cell's cap; when they alone exceed ``target`` the cap is 0."""

    def total(c: int) -> int:
        return sum(a + min(t, max(0, c - a)) for a, t in cells.values())

    hi = max((a + t for a, t in cells.values()), default=0)
    if total(hi) <= target:
        return hi
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        lo, hi = (mid, hi) if total(mid) <= target else (lo, mid - 1)
    return lo


def select_frames(
    rows: Iterable[Mapping], annotated: set[str], target: int, thin: int = 10
) -> tuple[list[dict], dict]:
    """Rows need key, campaign, dive, seq, depth_band, region. Returns (selected, stats)."""
    ann: dict[tuple, list[dict]] = defaultdict(list)
    cand: dict[tuple, list[dict]] = defaultdict(list)
    seen = 0
    for r in rows:
        seen += 1
        if stem(r["key"]) in annotated:
            ann[cell_of(r)].append({**r, "squidle_annotated": True, "selection": "annotated"})
        elif r["seq"] % thin == 0:
            cand[cell_of(r)].append({**r, "squidle_annotated": False, "selection": "capped"})
    cells = {c: (len(ann.get(c, ())), len(cand.get(c, ()))) for c in set(ann) | set(cand)}
    cap = solve_cap(cells, target)
    out: list[dict] = []
    for c, (a, _t) in cells.items():
        out += ann.get(c, [])
        pool = sorted(cand.get(c, []), key=lambda r: sha_key(r["key"]))
        out += pool[: max(0, cap - a)]
    out.sort(key=lambda r: r["key"])
    n_ann = sum(a for a, _ in cells.values())
    stats = {
        "frames_seen": seen,
        "annotated": n_ann,
        "thinned_candidates": sum(t for _, t in cells.values()),
        "cells": len(cells),
        "cap": cap,
        "selected": len(out),
        "target": target,
        "over_target_by_annotated": max(0, n_ann - target),
    }
    return out, stats


# --- listing (network) --------------------------------------------------------------------
def _s3():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config

    cfg = Config(signature_version=UNSIGNED, retries={"max_attempts": 8, "mode": "adaptive"})
    return boto3.client("s3", region_name="ap-southeast-2", config=cfg)


def _dirs(s3, prefix: str) -> tuple[list[str], list[str]]:
    subs, keys = [], []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=prefix, Delimiter="/"
    ):
        subs += [c["Prefix"] for c in page.get("CommonPrefixes", [])]
        keys += [o["Key"] for o in page.get("Contents", [])]
    return subs, keys


def enumerate_dives(s3) -> list[tuple[str, str]]:
    camps = [c for c in _dirs(s3, ROOT)[0] if c[len(ROOT) : -1] not in SKIP]
    with cf.ThreadPoolExecutor(8) as ex:
        per = ex.map(lambda c: [(c, d) for d in _dirs(s3, c)[0]], camps)
    return [cd for lst in per for cd in lst]


def _dive_path(cache: Path, camp: str, dive: str) -> Path:
    return cache / "dives" / camp[len(ROOT) : -1] / dive.rstrip("/").rsplit("/", 1)[-1]


def cached(cache: Path, camp: str, dive: str) -> bool:
    p = _dive_path(cache, camp, dive)
    return p.with_suffix(".parquet").exists() or p.with_suffix(".none").exists()


def cache_dive(s3, cache: Path, camp: str, dive: str) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    p = _dive_path(cache, camp, dive)
    p.parent.mkdir(parents=True, exist_ok=True)
    sub, _ = _dirs(s3, dive)
    gtif = next((s for s in sub if s.endswith("_gtif/")), None)
    tf = next((s for s in sub if s.endswith("track_files/")), None)
    csvs = [k for k in _dirs(s3, tf)[1] if k.endswith("_latlong.csv")] if tf else []
    rows: list[dict] = []
    if gtif and csvs:
        body = s3.get_object(Bucket=BUCKET, Key=csvs[0])["Body"].read()
        rows = track_rows(
            camp[len(ROOT) : -1], dive, gtif, parse_track(body.decode("utf-8", "replace"))
        )
    if not rows:
        reason = "no-gtif" if not gtif else "no-track" if not csvs else "empty-track"
        p.with_suffix(".none").write_text(reason)
        return 0
    tmp = p.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    tmp.rename(p.with_suffix(".parquet"))
    return len(rows)


def scan_squidle(cache: Path, page: int = 2000, stop: threading.Event | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = cache / "squidle"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "_done").exists():
        return
    pages = sorted(d.glob("*.parquet"))
    last = pq.read_table(pages[-1])["id"].to_pylist()[-1] if pages else 0
    while not (stop and stop.is_set()):
        q = {"filters": [*SQUIDLE_FILTERS, {"name": "id", "op": "gt", "val": last}],
             "order_by": [{"field": "id", "direction": "asc"}]}  # fmt: skip
        url = SQUIDLE + "?" + urllib.parse.urlencode({"q": json.dumps(q), "results_per_page": page})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(
                    urllib.request.Request(url, headers=_UA), timeout=300
                ) as resp:
                    objs = json.load(resp)["objects"]
                break
            except Exception:  # retried, then the pass ends (resumable)
                if attempt == 5:
                    raise
                time.sleep(30 * (attempt + 1))
        if objs:
            rows = [
                {"id": o["id"], "key": o["key"], "deployment": o["deployment"]["key"]} for o in objs
            ]
            tmp = d / f"{rows[0]['id']:010d}.tmp"
            pq.write_table(pa.Table.from_pylist(rows), tmp)
            tmp.rename(tmp.with_suffix(".parquet"))
            last = rows[-1]["id"]
        if len(objs) < page:
            (d / "_done").write_text(str(last))
            return
        time.sleep(1)


def progress(cache: Path, dives: list[tuple[str, str]], extra: dict) -> dict:
    sq = cache / "squidle"
    st = {
        "dives_total": len(dives),
        "dives_cached": sum(cached(cache, c, d) for c, d in dives),
        "squidle_pages": len(list(sq.glob("*.parquet"))) if sq.exists() else 0,
        "squidle_done": (sq / "_done").exists(),
        "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    st["complete"] = st["dives_cached"] == st["dives_total"] > 0 and st["squidle_done"]
    (cache / "_progress.json").write_text(json.dumps(st, indent=1))
    return st


def list_all(cache: Path, workers: int) -> dict:
    s3 = _s3()
    dives = enumerate_dives(s3)
    (cache / "_dives.json").write_text(json.dumps(dives))
    todo = [(c, d) for c, d in dives if not cached(cache, c, d)]
    stop = threading.Event()
    sq = threading.Thread(target=scan_squidle, args=(cache,), kwargs={"stop": stop}, daemon=True)
    sq.start()
    failed = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(cache_dive, s3, cache, c, d): d for c, d in todo}
        for i, f in enumerate(cf.as_completed(futs), 1):
            if f.exception() is not None:
                failed += 1
            if i % 25 == 0:
                print(
                    json.dumps(progress(cache, dives, {"dives_failed_this_pass": failed})),
                    flush=True,
                )
    sq.join()
    return progress(cache, dives, {"dives_failed_this_pass": failed})


def build(cache: Path, out: Path, meow_path: Path, target: int, thin: int) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    st = json.loads((cache / "_progress.json").read_text())
    if not st.get("complete"):
        raise SystemExit(
            f"cache incomplete ({st['dives_cached']}/{st['dives_total']}); no manifest"
        )
    meow = load_meow(meow_path)
    rows: list[dict] = []
    for f in sorted((cache / "dives").rglob("*.parquet")):
        dive_rows = pq.read_table(f).to_pylist()
        region = meow_region(
            statistics.median(r["lat"] for r in dive_rows),
            statistics.median(r["lon"] for r in dive_rows),
            meow,
        )
        rows += [{**r, "region": region} for r in dive_rows]
    annotated = {
        k
        for f in sorted((cache / "squidle").glob("*.parquet"))
        for k in pq.read_table(f)["key"].to_pylist()
    }
    sel, stats = select_frames(rows, annotated, target, thin)
    stats["squidle_media"] = len(annotated)
    stats["squidle_unmatched"] = len(annotated - {stem(r["key"]) for r in rows})
    stats["regions"] = len({r["region"] for r in sel})
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(sel), out, compression="zstd")
    stats.update(
        complete=True,
        dives=st["dives_total"],
        bytes=out.stat().st_size,
        sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        meow_sha256=hashlib.sha256(meow_path.read_bytes()).hexdigest(),
        thin=thin,
    )
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    return stats


def upload(out: Path, prefix: str) -> dict:
    """PUT the manifest + stats to rs-storage-open ``prefix`` and HEAD both anonymously."""
    from marinedata.s3_upload import client_from_rclone

    s3 = client_from_rclone("rs-hel1")
    res = {}
    for f in (out, out.with_suffix(".json")):
        key = prefix.rstrip("/") + "/" + f.name
        s3.upload_file(str(f), "rs-storage-open", key)
        url = f"{s3.meta.endpoint_url.rstrip('/')}/rs-storage-open/{key}"
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            res[key] = {"url": url, "status": r.status, "length": int(r.headers["Content-Length"])}
        assert res[key]["length"] == f.stat().st_size, key
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--meow", type=Path, required=True, help="MEOW ecoregions GeoJSON")
    ap.add_argument("--target", type=int, default=300_000)
    ap.add_argument("--thin", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--phase", choices=("list", "build", "all"), default="all")
    ap.add_argument(
        "--upload-prefix", help="e.g. sources/imos-auv/_manifest (only after a full build)"
    )
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    if a.phase in ("list", "all"):
        st = list_all(a.cache, a.workers)
        print(json.dumps(st), flush=True)
        if not st["complete"]:
            raise SystemExit(2)
    if a.phase in ("build", "all"):
        print(json.dumps(build(a.cache, a.out, a.meow, a.target, a.thin)), flush=True)
        if a.upload_prefix:
            print(json.dumps(upload(a.out, a.upload_prefix)), flush=True)


if __name__ == "__main__":
    main()
