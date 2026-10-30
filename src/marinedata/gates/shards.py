"""Shared readers for the gates: shards, split names, RELEASE.json, SPLIT_MAP.json, the card."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

_RELEASE = r"v\d+\.\d+\.\d+"
RELEASE_ID = re.compile(_RELEASE)
"""A release id as ``--release-id`` takes it and as an additive export puts it in a shard name."""
_SHARD = re.compile(
    r"^(?P<split>[A-Za-z][A-Za-z0-9_]*)"
    rf"(?:-(?!{_RELEASE}-\d{{5}}-of-)(?P<source>.+?))?"  # a lone release is not a source
    rf"(?:-(?P<release>{_RELEASE}))?"
    r"-(?P<index>\d{5})-of-(?P<total>\d{5})\.parquet$"
)
SPLIT_ALIASES = {"val": "validation", "valid": "validation", "probe": "validation"}


class CardError(ValueError):
    """README.md is missing or does not parse as a dataset card."""


def norm_split(name: str) -> str:
    """Hub split name of a release split (``val`` and ``probe`` are ``validation``)."""
    return SPLIT_ALIASES.get(name, name)


@dataclass(frozen=True)
class ShardName:
    """``<split>[-<source>][-<release>]-NNNNN-of-MMMMM.parquet`` taken apart.

    ``split`` is the first token as written (``val`` stays ``val``: :func:`norm_split` renames it).
    ``source`` is ``None`` for a per-split name (``train-00000-of-00004``). ``release`` is ``None``
    unless an additive export named the shard after its release
    (``train-<source>-v1.2.0-00000-of-00001``); it is never part of ``source``."""

    split: str
    source: str | None
    release: str | None
    index: int
    total: int


def parse_shard(path: Path | str) -> ShardName | None:
    """The parts of a shard file name, ``None`` when it is not of the form above. The one reader
    of that form: :func:`shard_split`, and with it every per-split count of the gates, uses it."""
    match = _SHARD.match(Path(path).name)
    if match is None:
        return None
    return ShardName(
        match["split"], match["source"], match["release"], int(match["index"]), int(match["total"])
    )


def shard_split(path: Path) -> str:
    """Hub split of a shard (first token of its name, aliases applied); a name that does not
    parse still has its first token."""
    shard = parse_shard(path)
    return norm_split(shard.split if shard else Path(path).stem.split("-")[0])


def config_shards(build_dir: Path) -> dict[str, list[Path]]:
    """``{config: [shard, ...]}`` of ``<build>/data/<config>/*.parquet``, sorted."""
    data = Path(build_dir) / "data"
    found = {}
    for folder in sorted(p for p in data.iterdir() if p.is_dir()) if data.is_dir() else ():
        files = sorted(folder.glob("*.parquet"))
        if files:
            found[folder.name] = files
    return found


def shard_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema_arrow.names)


def split_counts(files: Sequence[Path]) -> dict[str, int]:
    """Rows per split of one config, from the parquet footers (no row is read)."""
    import pyarrow.parquet as pq

    out: Counter = Counter()
    for path in files:
        out[shard_split(path)] += pq.ParquetFile(path).metadata.num_rows
    return dict(sorted(out.items()))


SOURCE_COLUMNS = ("source_id", "source")


def source_split_counts(files: Sequence[Path]) -> dict[str, dict[str, int]] | None:
    """``{source: {split: rows}}`` of one config, from its ``source_id`` (else ``source``) column.

    ``None`` when a shard has neither column. Only that column is read, never an image; a row
    with no source counts under ``?``."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    out: dict[str, Counter] = {}
    for path in files:
        shard = pq.ParquetFile(path)
        column = next((c for c in SOURCE_COLUMNS if c in shard.schema_arrow.names), None)
        if column is None:
            return None
        split = shard_split(path)
        for item in pc.value_counts(shard.read(columns=[column]).column(0)).to_pylist():
            out.setdefault(str(item["values"] or "?"), Counter())[split] += item["counts"]
    return {source: dict(sorted(by.items())) for source, by in sorted(out.items())}


def iter_rows(
    files: Sequence[Path], columns: Sequence[str], *, batch_size: int = 64
) -> Iterator[tuple[Path, int, str, dict]]:
    """``(shard, row index, split, row)`` with the requested columns that the shard has."""
    import pyarrow.parquet as pq

    for path in files:
        shard = pq.ParquetFile(path)
        wanted = [c for c in columns if c in shard.schema_arrow.names]
        split, index = shard_split(path), 0
        for batch in shard.iter_batches(batch_size=batch_size, columns=wanted):
            for row in batch.to_pylist():
                yield path, index, split, row
                index += 1


def read_release(build_dir: Path, release_dir: Path | None = None) -> dict | None:
    """RELEASE.json from ``release_dir`` if given, else from the build directory."""
    for folder in (release_dir, build_dir):
        path = Path(folder) / "RELEASE.json" if folder else None
        if path is not None and path.is_file():
            return json.loads(path.read_text())
    return None


def find_split_map(build_dir: Path, release_dir: Path | None = None) -> tuple[Path, str] | None:
    """``(path, "release_dir" | "build_dir")`` of ``SPLIT_MAP.json``: the release dir's if it has
    one, else the build directory's; ``None`` when neither does."""
    for where, folder in (("release_dir", release_dir), ("build_dir", build_dir)):
        path = Path(folder) / "SPLIT_MAP.json" if folder else None
        if path is not None and path.is_file():
            return path, where
    return None


def read_card(build_dir: Path) -> tuple[dict, dict, str]:
    """``(front matter, {config: normalised card config}, body)`` of ``<build>/README.md``,
    parsed with ``huggingface_hub.DatasetCard`` and ``datasets``' own config reader."""
    path = Path(build_dir) / "README.md"
    if not path.is_file():
        raise CardError(f"no README.md in {build_dir}")
    try:
        from datasets.utils.metadata import MetadataConfigs
        from huggingface_hub import DatasetCard

        card = DatasetCard.load(path)
        configs = dict(MetadataConfigs.from_dataset_card_data(card.data))
    except Exception as exc:
        raise CardError(f"README.md does not parse as a dataset card: {str(exc)[:160]}") from exc
    return card.data.to_dict(), configs, card.text
