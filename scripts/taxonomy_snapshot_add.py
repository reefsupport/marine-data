"""Add WoRMS names to the pinned taxonomy snapshot without touching existing rows.

Usage: ``python scripts/taxonomy_snapshot_add.py <cache_dir> <name>[=Kingdom] ...``

Snapshot first: a name already present (any origin) is not queried. Missing names go
through the cached, <= 2 req/s :class:`~marinedata.worms_snapshot.WormsClient`; each new
row (and every ancestor needed for offline parent resolution) is stamped with today's
``retrieved_at``. Rows already in the snapshot are never rewritten. Codegen only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.parquet as pq

from marinedata.taxonomy import load_meta, taxonomy_dir
from marinedata.worms_snapshot import WormsClient, build_rows, write_parquet

REPO = Path(__file__).resolve().parents[1]


def main(cache: Path, specs: list[str]) -> None:
    root = REPO / "registry"
    path = taxonomy_dir(root) / load_meta(root)["snapshot"]
    rows = pq.read_table(path).to_pylist()
    have_ids = {int(r["aphia_id"]) for r in rows}
    have_names = {r["scientific_name"].lower() for r in rows}
    prefer: dict[str, tuple[str, ...]] = {}
    names = []
    for spec in specs:
        name, _, kingdom = spec.partition("=")
        if name.lower() in have_names:
            print(f"snapshot  {name}")
            continue
        names.append(name)
        if kingdom:
            prefer[name] = (kingdom,)
    if not names:
        return
    new, choices = build_rows(WormsClient(cache), ids=[], names=names, prefer=prefer)
    for c in choices:
        print(f"worms     {c.name} -> {c.aphia_id} ({c.reason})")
    added = [r for r in new if int(r["aphia_id"]) not in have_ids]
    merged = sorted(rows + added, key=lambda r: int(r["aphia_id"]))
    write_parquet(merged, path)
    print(f"added {len(added)} rows ({len(rows)} -> {len(merged)})")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
