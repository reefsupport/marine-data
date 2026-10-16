"""G3 leak: the open upload set names no nc source id, no nc repo and no private bucket.

Scans every file of the build for the private names (:data:`marinedata.hf_export_v11.PRIVATE_NAMES`,
plus the nc repo id from ``registry/flavours.yaml``) and the ids of every source that is not
open-class: file names, text files (card, croissant, NOTICE, provenance), parquet footers and every
string column of every shard (the bytes columns are never read). The nc card may name the nc repo,
never a bucket. A failure line carries a hash of the leaked name, never the name: GATES.json is
uploaded under ``provenance/``."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path

from . import GateResult, finish
from .v11_common import tree_files

NAME = "leak"
TEXT_SUFFIXES = frozenset(
    {"", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".cff", ".lock", ".csv", ".tsv",
     ".toml", ".html", ".xml", ".ini", ".sha256"}
)  # fmt: skip


def _tag(kind: str, value: str) -> str:
    return f"{kind}#{hashlib.sha256(value.encode()).hexdigest()[:8]}"


class _Finder:
    def __init__(self, names: Iterable[str], ids: Iterable[str]) -> None:
        self.names = [(n, n.lower()) for n in names if n]
        self.ids = [
            (i, re.compile(rf"(?<![A-Za-z0-9_-]){re.escape(i)}(?![A-Za-z0-9_-])", re.I))
            for i in sorted(set(ids))
        ]

    def hits(self, text: str) -> list[str]:
        low = text.lower()
        found = [_tag("private_name", n) for n, lowered in self.names if lowered in low]
        return found + [_tag("nc_id", i) for i, rx in self.ids if rx.search(text)]


def _scan_parquet(path: Path, finder: _Finder) -> set[str]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(path)
    found: set[str] = set()
    for key, value in (shard.schema_arrow.metadata or {}).items():
        found.update(finder.hits(key.decode(errors="ignore") + " " + value.decode(errors="ignore")))
    text = (pa.types.is_string, pa.types.is_large_string)
    columns = [
        f.name
        for f in shard.schema_arrow
        if any(t(f.type) for t in text)
        or (
            (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
            and any(t(f.type.value_type) for t in text)
        )
    ]
    if not columns:
        return found
    for batch in shard.iter_batches(batch_size=8192, columns=columns):
        for array in batch.columns:
            if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
                array = pc.list_flatten(array)
            for value in pc.unique(array).to_pylist():
                if value:
                    found.update(finder.hits(value))
    return found


def run_leak_gate(
    build_dir: Path,
    *,
    nc_ids: Collection[str],
    flavour: str = "open",
    private_names: Collection[str] | None = None,
    nc_repo_names: Collection[str] | None = None,
) -> GateResult:
    """``nc_ids``: every source id that must not appear (open flavour only). ``private_names``
    (buckets, remotes) are banned in both flavours; ``nc_repo_names`` only in the open one."""
    from ..hf_export_v11 import PRIVATE_NAMES

    build = Path(build_dir)
    privates = tuple(PRIVATE_NAMES[1:] if private_names is None else private_names)
    repos = tuple(PRIVATE_NAMES[:1] if nc_repo_names is None else nc_repo_names)
    open_flavour = flavour == "open"
    finder = _Finder((*privates, *(repos if open_flavour else ())), nc_ids if open_flavour else ())
    counts: Counter = Counter()
    failures: list[str] = []
    for path in tree_files(build):
        rel = path.relative_to(build).as_posix()
        counts["files"] += 1
        found = set(finder.hits(rel))
        if path.suffix == ".parquet":
            counts["parquet_files"] += 1
            found |= _scan_parquet(path, finder)
        elif path.suffix.lower() in TEXT_SUFFIXES:
            counts["text_files"] += 1
            found.update(finder.hits(path.read_text(encoding="utf-8", errors="ignore")))
        else:
            counts["binary_skipped"] += 1
        for tag in sorted(found):
            counts["leaks"] += 1
            failures.append(f"{rel} leak: {tag} in the {flavour} upload set")
    if not counts["files"]:
        failures.append(f"no files under {build}")
    return finish(NAME, counts, failures)
