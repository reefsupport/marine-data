"""Build registry/overlays/marineinst20m.parquet from the MarineInst20M GitHub zips.

Streams one zip at a time into ``--cache`` and deletes it after parsing (peak local
disk = the largest zip, 82 MB). Usage: build_marineinst_overlay.py --cache DIR --out PATH"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.adapters._http import download
from marinedata.overlay_marineinst import RAW, TREE, overlay_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    a.cache.mkdir(parents=True, exist_ok=True)
    tree = json.loads(urllib.request.urlopen(TREE, timeout=60).read())["tree"]
    zips = sorted(e["path"] for e in tree if e["path"].endswith(".zip"))
    rows: list[dict] = []
    for path in zips:
        group, dataset = path.split("/")[:2]
        if a.only and dataset not in a.only:
            continue
        dest = a.cache / path.replace("/", "__")
        download(RAW + path, dest)
        n0 = len(rows)
        rows.extend(overlay_rows(dest, dataset, group))
        dest.unlink()
        print(f"{path}\t{len(rows) - n0}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), a.out, compression="zstd", compression_level=19)
    print("TOTAL", len(rows), a.out.stat().st_size)


if __name__ == "__main__":
    main()
