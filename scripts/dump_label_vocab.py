"""Dump the label vocabulary of every published source (no image bytes).

Column projection over ``hf://`` parquet: only ``source``/``source_id`` and the label column
(``class_map`` / ``instances`` / ``boxes``) are fetched. Output JSON::

    {"<repo>/<config>": {"<source>": {"kind": ..., "rows": n, "labels": [{...}]}}}

Usage: ``python scripts/dump_label_vocab.py OUT.json`` (needs ``huggingface_hub`` login for the
gated ``marine-data-nc`` repo; read-only).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem

REPOS = ("reefsupport/marine-data", "reefsupport/marine-data-nc")
LABEL_COLUMN = {
    "coral-masks": ("class_map", "source"),
    "coral-masks-machine": ("class_map", "source"),
    "scene-masks": ("class_map", "source"),
    "instance-masks": ("instances", "source"),
    "fish-boxes": ("boxes", "source_id"),
}


def _read(path: str, columns: list[str]) -> list[dict]:
    with HfFileSystem().open(f"datasets/{path}") as handle:
        return pq.ParquetFile(handle).read(columns=columns).to_pylist()


def dump() -> dict:
    out: dict = {}
    jobs = []
    for repo in REPOS:
        for name in HfApi().list_repo_files(repo, repo_type="dataset"):
            parts = name.split("/")
            if len(parts) == 3 and parts[0] == "data" and parts[2].endswith(".parquet"):
                jobs.append((repo, parts[1], f"{repo}/{name}"))

    def run(job):
        repo, config, path = job
        column, source_column = LABEL_COLUMN[config]
        return repo, config, _read(path, [column, source_column]), column, source_column

    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(run, jobs))

    acc: dict = defaultdict(lambda: defaultdict(lambda: {"rows": 0, "labels": defaultdict(int)}))
    for repo, config, rows, column, source_column in results:
        for row in rows:
            entry = acc[f"{repo}/{config}"][row[source_column]]
            entry["rows"] += 1
            for item in row[column] or []:
                if column == "boxes":
                    key = (None, item["label"], None, None)
                else:
                    key = (
                        item.get("id") if column == "class_map" else None,
                        item["label_native"],
                        item.get("taxon_node"),
                        item.get("coarse"),
                    )
                entry["labels"][key] += 1
    for config_key, sources in acc.items():
        out[config_key] = {}
        for source, entry in sorted(sources.items()):
            labels = [
                {"id": k[0], "label_native": k[1], "taxon_node": k[2], "coarse": k[3], "count": n}
                for k, n in sorted(
                    entry["labels"].items(),
                    key=lambda kv: (kv[0][0] is None, kv[0][0] or 0, kv[0][1]),
                )
            ]
            out[config_key][source] = {
                "kind": LABEL_COLUMN[config_key.split("/", 2)[2]][0],
                "rows": entry["rows"],
                "labels": labels,
            }
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w", encoding="utf-8") as handle:
        json.dump(dump(), handle, indent=1, sort_keys=True)
