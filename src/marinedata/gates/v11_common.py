"""Readers shared by the v1.1.0 gates (G1-G9): table shards, the frozen v1.0 tree, the card."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection, Iterator, Sequence
from pathlib import Path
from typing import Any

import yaml

from . import GateResult, finish
from .shards import config_shards, shard_split

IMAGES, METADATA = "images", "metadata"
LABEL_TABLES = ("points", "bleaching", "boxes", "semseg", "instances")
MAIN_SPLITS = ("train", "validation", "test")
EVAL_SPLITS = frozenset({"validation", "test", "test_ood"})
HEX64 = re.compile(r"^[0-9a-f]{64}$")
_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n", re.S)
SKIP_DIRS = frozenset({".cache", ".git", ".huggingface", "__pycache__"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_files(root: Path) -> Iterator[Path]:
    """Every file under ``root`` (sorted), minus cache folders."""
    for path in sorted(Path(root).rglob("*")):
        if path.is_file() and not (set(path.relative_to(root).parts) & SKIP_DIRS):
            yield path


def shards(build_dir: Path, config: str) -> list[Path]:
    return config_shards(Path(build_dir)).get(config, [])


def column_values(files: Sequence[Path], column: str) -> Iterator[Any]:
    """Every value of ``column`` across ``files`` (a shard without the column is skipped)."""
    import pyarrow.parquet as pq

    for path in files:
        shard = pq.ParquetFile(path)
        if column in shard.schema_arrow.names:
            for batch in shard.iter_batches(batch_size=8192, columns=[column]):
                yield from batch.column(0).to_pylist()


def table_rows(files: Sequence[Path], columns: Sequence[str]) -> Iterator[dict]:
    """Rows with the requested columns each shard has, plus ``_split`` and ``_shard`` (name)."""
    import pyarrow.parquet as pq

    for path in files:
        shard = pq.ParquetFile(path)
        wanted = [c for c in columns if c in shard.schema_arrow.names]
        if not wanted:
            continue
        split = shard_split(path)
        for batch in shard.iter_batches(batch_size=4096, columns=wanted):
            for row in batch.to_pylist():
                row["_split"], row["_shard"] = split, path.name
                yield row


def v1_splits(v1_dir: Path) -> dict[str, set[str]]:
    """``image_sha256 -> {v1.0 splits}``, from ``<v1_dir>/data/*/<split>-*.parquet``."""
    out: dict[str, set[str]] = {}
    for files in config_shards(Path(v1_dir)).values():
        for path in files:
            split = shard_split(path)
            for sha in column_values([path], "image_sha256"):
                if sha:
                    out.setdefault(str(sha), set()).add(split)
    return out


def v1_config_shas(v1_dir: Path, configs: Collection[str] | None = None) -> dict[str, set[str]]:
    """``{config: {image_sha256}}`` of the frozen v1.0 tree, from
    ``<v1_dir>/data/<config>/*.parquet`` (only ``configs`` when given): the per-config twin of
    :func:`v1_splits`."""
    return {
        config: {str(sha) for sha in column_values(files, "image_sha256") if sha}
        for config, files in config_shards(Path(v1_dir)).items()
        if configs is None or config in configs
    }


def front_matter(readme: Path) -> dict:
    match = _FRONT.match(Path(readme).read_text(encoding="utf-8"))
    data = yaml.safe_load(match.group(1)) if match else None
    return data if isinstance(data, dict) else {}


def card_configs(readme: Path) -> dict[str, dict]:
    """``{config_name: entry}`` of the card's ``configs:`` front matter."""
    entries = front_matter(readme).get("configs") or []
    return {str(e["config_name"]): e for e in entries if isinstance(e, dict) and "config_name" in e}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


class Collector:
    """Counters plus capped failure lines. Exact totals live in ``counts`` as ``failed_<kind>``;
    the gate fails as soon as one line was recorded."""

    def __init__(self, cap: int = 200) -> None:
        self.counts: Counter = Counter()
        self.lines: list[str] = []
        self._cap = cap

    def fail(self, kind: str, line: str) -> None:
        self.counts[f"failed_{kind}"] += 1
        if len(self.lines) < self._cap:
            self.lines.append(line)

    def result(self, name: str, **extra: Any) -> GateResult:
        failed = sum(v for k, v in self.counts.items() if k.startswith("failed_"))
        return finish(name, {**self.counts, **extra, "failures_all": failed}, self.lines)
