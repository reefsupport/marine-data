"""Build the iNat-marine D-AB manifest (WP-6e-A) for ``adapter: inat-open-data``.

Streams ``observations.csv.gz`` and ``photos.csv.gz`` from the anonymous
``inaturalist-open-data`` bucket (resumable HTTP range reads, nothing raw written to
disk), keeps research-grade observations of the committed marine taxon set, the
position-0 photo of each, then applies :func:`marinedata.adapters.inat.select_subset`
(<= cap per taxon by sha256(observation_uuid)). Writes one parquet with
``MANIFEST_COLUMNS`` + species/class, and prints a JSON summary line.

Usage: python scripts/inat_manifest.py OUT.parquet [--cap 100] [--taxa <csv.gz>]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.request

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from marinedata.adapters.inat import MANIFEST_COLUMNS, load_marine_taxa, select_subset

BASE = "https://inaturalist-open-data.s3.amazonaws.com/"
TAXA = "registry/ingest-specs/data/inat-marine-taxa-2026-09-25.csv.gz"


class Resumable(io.RawIOBase):
    """Range-resuming GET stream (the 20 GB photos CSV outlives single connections)."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.resp = url, 0, None

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        for attempt in range(8):
            try:
                if self.resp is None:
                    req = urllib.request.Request(
                        self.url,
                        headers={"Range": f"bytes={self.pos}-", "User-Agent": "marinedata-wp6e-a"},
                    )
                    self.resp = urllib.request.urlopen(req, timeout=120)
                n = self.resp.readinto(b)
                self.pos += n
                return n
            except Exception as exc:
                print(f"retry {self.url} @{self.pos}: {exc}", file=sys.stderr, flush=True)
                self.resp = None
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"stream failed: {self.url}")


def _reader(name: str, cols: list[str], types: dict[str, pa.DataType]):
    raw = io.BufferedReader(Resumable(BASE + name), 8 << 20)
    stream = pa.input_stream(pa.PythonFile(raw, mode="r"), compression="gzip")
    return pacsv.open_csv(
        stream,
        read_options=pacsv.ReadOptions(block_size=128 << 20),
        parse_options=pacsv.ParseOptions(
            delimiter="\t", quote_char=False, invalid_row_handler=lambda r: "skip"
        ),
        convert_options=pacsv.ConvertOptions(include_columns=cols, column_types=types),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--taxa", default=TAXA)
    a = ap.parse_args(argv)
    t0 = time.time()
    taxa = load_marine_taxa(a.taxa)
    value_set = pa.array(sorted(taxa), pa.int64())
    s = pa.string()
    obs_cols = [
        "observation_uuid",
        "observer_id",
        "latitude",
        "longitude",
        "taxon_id",
        "quality_grade",
        "observed_on",
    ]
    parts, n_obs = [], 0
    for b in _reader(
        "observations.csv.gz",
        obs_cols,
        {"taxon_id": pa.int64(), "observed_on": s, "observation_uuid": s},
    ):
        n_obs += b.num_rows
        m = pc.and_(pc.equal(b["quality_grade"], "research"), pc.is_in(b["taxon_id"], value_set))
        parts.append(pa.Table.from_batches([b]).filter(m).drop_columns(["quality_grade"]))
    obs = pa.concat_tables(parts)
    print(f"observations {n_obs} marine-RG {obs.num_rows} {time.time() - t0:.0f}s", flush=True)
    ph_cols = [
        "photo_id",
        "observation_uuid",
        "extension",
        "license",
        "width",
        "height",
        "position",
    ]
    joined, n_ph = [], 0
    for b in _reader(
        "photos.csv.gz",
        ph_cols,
        {"photo_id": pa.int64(), "position": pa.int64(), "observation_uuid": s},
    ):
        n_ph += b.num_rows
        t = pa.Table.from_batches([b])
        t = t.filter(pc.equal(pc.fill_null(t["position"], 0), 0))
        j = obs.join(t, "observation_uuid", join_type="inner")
        if j.num_rows:
            joined.append(j)
    print(
        f"photos {n_ph} joined {sum(j.num_rows for j in joined)} {time.time() - t0:.0f}s",
        flush=True,
    )
    df = pa.concat_tables(joined).to_pandas()
    picked = select_subset(df, cap=a.cap)
    picked["class"] = picked["taxon_id"].map(lambda t: taxa[int(t)][0])
    picked["species"] = picked["taxon_id"].map(lambda t: taxa[int(t)][1])
    cols = [*MANIFEST_COLUMNS, "position", "obs_sha", "class", "species"]
    pq.write_table(pa.Table.from_pandas(picked[cols], preserve_index=False), a.out)
    print(
        json.dumps(
            {
                "rows": len(picked),
                "taxa": int(picked["taxon_id"].nunique()),
                "by_license": picked["license"].value_counts().to_dict(),
                "with_latlon": int(picked["latitude"].notna().sum()),
                "seconds": round(time.time() - t0),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
