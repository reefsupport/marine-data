"""Prior-release configs of a v1.1/v1.2 card: the Hub folders the build does not ship.

An additive publish uploads the new shards and leaves the rest of the repo as it is, so the Hub
keeps folders the build does not write (``condition-masks`` of v1.1.1 in the v1.2.0 repo). A card
that does not declare them leaves their files without a loader. ``--v1-dir`` (the repo before the
publish) names them: each gets a ``configs:`` entry whose splits come from its shard names
(:func:`marinedata.gates.shards.parse_shard`), and a row in the card marked as a frozen
prior-release config. Rows are read from the shard footers; nothing is rebuilt or rewritten."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .gates.shards import (
    SOURCE_COLUMNS,
    config_shards,
    norm_split,
    parse_shard,
    shard_columns,
    split_counts,
)
from .gates.v11_common import column_values
from .hf_export_v11 import V11ExportError

SPLIT_ORDER = ("train", "validation", "test", "test_ood")
V1_RELEASE = "v1.0"
INTRODUCED = {"condition-masks": "v1.1.1"}
"""The release that first shipped a prior-only config that is not one of the v1.0 configs."""


@dataclass(frozen=True)
class PriorConfig:
    """One ``data/<config>`` folder of the prior tree that the build does not ship."""

    name: str
    release: str | None  # the release that shipped it, when known
    splits: tuple[tuple[str, str, int], ...]  # (Hub split, shard prefix as written, rows)
    sources: tuple[str, ...]

    @property
    def rows(self) -> int:
        return sum(n for _, _, n in self.splits)

    @property
    def origin(self) -> str:
        return self.release or "the previous release"

    @property
    def split_text(self) -> str:
        return ", ".join(f"`{split}` {n:,}" for split, _, n in self.splits)

    @property
    def data_files(self) -> list[dict]:
        return [
            {"split": split, "path": f"data/{self.name}/{prefix}-*.parquet"}
            for split, prefix, _ in self.splits
        ]


def _sources(files: Sequence[Path]) -> tuple[str, ...]:
    """The source ids of a config: from its shard names, else from the ``source_id`` column."""
    named = {s.source for s in map(parse_shard, files) if s and s.source}
    if named:
        return tuple(sorted(named))
    column = next((c for c in SOURCE_COLUMNS if c in shard_columns(files[0])), None)
    if column is None:
        return ()
    return tuple(sorted({str(v) for v in column_values(files, column) if v}))


def _prior(name: str, files: Sequence[Path], v1_names: Collection[str]) -> PriorConfig:
    counts = split_counts(files)  # Hub split -> rows, read from the footers
    prefix: dict[str, str] = {}
    for path in files:
        shard = parse_shard(path)
        token = shard.split if shard else path.stem.split("-")[0]
        prefix.setdefault(norm_split(token), token)
    order = [s for s in SPLIT_ORDER if s in counts] + sorted(set(counts) - set(SPLIT_ORDER))
    splits = tuple((s, prefix.get(s, s), counts[s]) for s in order)
    release = V1_RELEASE if name in v1_names else INTRODUCED.get(name)
    return PriorConfig(name, release, splits, _sources(files))


def prior_configs(
    v1_dir: str | Path, shipped: Collection[str], v1_names: Collection[str] = ()
) -> list[PriorConfig]:
    """The configs of the prior tree at ``v1_dir`` that the build (``shipped``) does not ship: the
    v1.0 ones (``v1_names``, in that order) first, then the others by name."""
    found = config_shards(Path(v1_dir))
    if not found:
        raise V11ExportError("--v1-dir holds no data/<config>/*.parquet: nothing to carry")
    left = {name: files for name, files in found.items() if name not in shipped}
    order = [n for n in v1_names if n in left] + sorted(set(left) - set(v1_names))
    return [_prior(name, left[name], v1_names) for name in order]


def entries(prior: Sequence[PriorConfig], verbatim: Mapping[str, dict]) -> list[dict]:
    """``configs:`` entries: a v1.0 config keeps its v1.0 entry byte for byte, the others list the
    splits their shard names carry."""
    return [
        verbatim.get(p.name) or {"config_name": p.name, "data_files": p.data_files} for p in prior
    ]


def table_rows(prior: Sequence[PriorConfig]) -> list[tuple]:
    """Rows of the card's configuration table (Config, Task, Rows, Sources, Notes)."""
    return [
        (
            f"`{p.name}`",
            f"Frozen prior-release config ({p.origin})",
            f"{p.rows:,}",
            ", ".join(f"`{s}`" for s in p.sources),
            f"Carried unchanged from {p.origin}: {p.split_text} rows. Not rebuilt by this "
            "release; its files stay as published.",
        )
        for p in prior
    ]


def structure_note(prior: Sequence[PriorConfig]) -> str:
    """The paragraph that follows the structure list (empty without prior-only configs)."""
    if not prior:
        return ""
    listed = "; ".join(f"`{p.name}` ({p.origin}: {p.split_text} rows)" for p in prior)
    return (
        "\n\nFrozen prior-release configs, carried unchanged and not rebuilt by this release: "
        f"{listed}. Their files stay as published; this card declares them in its `configs:` "
        "header so that `load_dataset` finds them."
    )
