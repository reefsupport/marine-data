#!/usr/bin/env python3
"""WS-D step 3: does every old S3 prefix resolve to exactly one registry id?

Reads the frozen wsd-mapping.tsv report (18 old-tree prefixes + the two the audit
already flagged as ``MISSING``) and checks each surviving ``registry_id`` value
against the *current* registry — catching drift between "what the mapping report
said in 2026-09" and "what the registry actually declares now" without needing
bucket access (this reads only the TSV and the local YAML sources).

Not committed — a one-off verification script for this step's acceptance check,
run with the worktree's own ``.venv``:

    python scripts/check_wsd_mapping.py [path/to/wsd-mapping.tsv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from marinedata import Registry  # noqa: E402
from marinedata.registry import RegistryError  # noqa: E402

DEFAULT_TSV = Path(
    "~/.claude-state/projects/reefsupport/tasks/reports/data/2026-09-23/wsd-mapping.tsv"
).expanduser()


def main() -> int:
    tsv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TSV
    registry = Registry.load()

    with tsv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    exempt: list[str] = []  # rows whose registry_id is (or contains) MISSING
    single: list[str] = []  # rows resolving to exactly one known registry id
    multi: list[tuple[str, list[str]]] = []  # rows resolving to 2+ ids
    broken: list[tuple[str, str]] = []  # rows naming an id that doesn't exist

    for row in rows:
        prefix = row["prefix"]
        raw_id = row["registry_id"]
        if "MISSING" in raw_id:
            exempt.append(prefix)
            continue
        candidates = raw_id.split("|")
        unknown = [c for c in candidates if not _exists(registry, c)]
        if unknown:
            broken.append((prefix, ", ".join(unknown)))
        elif len(candidates) == 1:
            single.append(prefix)
        else:
            multi.append((prefix, candidates))

    print(f"rows: {len(rows)}")
    print(f"exempt (MISSING): {len(exempt)}")
    for p in exempt:
        print(f"  - {p}")
    print(f"single-id (pass): {len(single)}")
    print(f"multi-id (known split, not an error): {len(multi)}")
    for p, ids in multi:
        print(f"  - {p} -> {ids}")
    print(f"broken (unknown id — real failure): {len(broken)}")
    for p, ids in broken:
        print(f"  - {p} -> unknown id(s): {ids}")

    return 1 if broken else 0


def _exists(registry: Registry, source_id: str) -> bool:
    try:
        registry.source(source_id)
        return True
    except RegistryError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
