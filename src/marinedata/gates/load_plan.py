"""The files a card's ``configs:`` entries name, resolved against the build and the v1.0 tree.

``datasets`` infers the builder of a local folder from the FIRST ``configs:`` entry of the README
and ignores the config asked for, and an additive release's build dir does not hold the frozen v1.0
configs (their shards live in the v1.0 tree). The load gate therefore reads the card itself: every
entry's patterns are resolved here (no ``datasets`` needed) against the build, a pattern that
matches nothing there falls back to ``--v1-dir``, and one that matches in neither is
``config_files_missing``, unless it is an inherited v1.0 phantom (:func:`inherited_phantoms`): a
pattern of a frozen v1.0 config that the v1.0 card already listed, unchanged, and the v1.0 tree
never held a file for. That one is counted, not failed: no release can both keep the v1.0 entry
byte-identical (the frozen gate) and make a pattern the v1.0 repo never satisfied match."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from .v11_common import canonical
from .v11_common import card_configs as written_configs

FROZEN_V10_CONFIGS = frozenset({"coral-masks", "scene-masks", "instance-masks", "fish-boxes"})
"""The v1.0 configs an additive release never rebuilds: their shards live in the v1.0 tree only."""


@dataclass(frozen=True)
class ConfigFiles:
    config: str
    splits: Mapping[str, tuple[Path, ...]]  # card split name -> files (build first, else v1.0)
    missing: tuple[tuple[str, str], ...]  # (split, pattern) that matched in neither tree: they fail
    from_v1: int  # files taken from the v1.0 tree
    inherited: tuple[tuple[str, str], ...] = ()  # (split, pattern) missing in both, tolerated

    @property
    def resolved(self) -> bool:
        return bool(self.splits) and not self.missing

    @property
    def frozen(self) -> bool:
        return self.config in FROZEN_V10_CONFIGS


def config_patterns(entry: object) -> dict[str, list[str]]:
    """``{split: [pattern, ...]}`` of one ``configs:`` entry. ``data_files`` is a path, a list of
    paths, a list of ``{split, path}`` items (``path`` a string or a list) or a ``{split: path(s)}``
    mapping; a bare path is the ``train`` split."""
    raw = entry.get("data_files") if isinstance(entry, Mapping) else None
    out: dict[str, list[str]] = {}

    def add(split: object, path: object) -> None:
        paths = (
            [path]
            if isinstance(path, str)
            else list(path)
            if isinstance(path, (list, tuple))
            else []
        )
        out.setdefault(str(split), []).extend(p for p in paths if isinstance(p, str))

    if isinstance(raw, str):
        add("train", raw)
    elif isinstance(raw, Mapping):
        for split, path in raw.items():
            add(split, path)
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if isinstance(item, Mapping) and item.get("path") is not None:
                add(item.get("split") or "train", item["path"])
            elif isinstance(item, str):
                add("train", item)
    return out


def _matches(root: Path | None, pattern: str) -> list[Path]:
    if root is None or pattern.startswith("/") or ".." in Path(pattern).parts:
        return []  # a card pattern is relative to, and stays inside, the repo root
    return sorted(p for p in Path(root).glob(pattern) if p.is_file())


def resolve_config(
    config: str, entry: object, build_dir: Path, v1_dir: Path | None = None
) -> ConfigFiles:
    """Resolve each pattern of ``entry`` in the build, else (per pattern) in ``v1_dir``."""
    splits: dict[str, tuple[Path, ...]] = {}
    missing: list[tuple[str, str]] = []
    from_v1 = 0
    for split, patterns in config_patterns(entry).items():
        files: list[Path] = []
        for pattern in patterns:
            hit = _matches(build_dir, pattern)
            if not hit:
                hit = _matches(v1_dir, pattern)
                from_v1 += len(hit)
            if not hit:
                missing.append((split, pattern))
            files.extend(p for p in hit if p not in files)
        if files:
            splits[split] = tuple(files)
    if not splits and not missing:
        missing.append(("*", "<no data_files>"))
    return ConfigFiles(config, splits, tuple(missing), from_v1)


def _written_entries(folder: Path) -> dict[str, dict]:
    """``{config: entry}`` of ``<folder>/README.md`` as written (the reader of the frozen gate);
    ``{}`` when the card is missing or unreadable, so nothing is inherited from it."""
    try:
        return written_configs(Path(folder) / "README.md")
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def _is_v1_config(v1_dir: Path, config: str) -> bool:
    """``<v1_dir>/data/<config>`` is a folder (``config`` must be a plain folder name)."""
    plain = config not in {"", ".", ".."} and Path(config).name == config
    return plain and (Path(v1_dir) / "data" / config).is_dir()


def inherited_phantoms(
    item: ConfigFiles,
    build_entry: object,
    v1_dir: Path | None,
    v1_cards: Mapping[str, object],
) -> frozenset[tuple[str, str]]:
    """The ``(split, pattern)`` pairs of ``item.missing`` that v1.0 already shipped empty.

    A pair is inherited only if ALL hold, else it keeps failing as ``config_files_missing``:

    1. ``v1_dir`` is given and ``<v1_dir>/data/<config>`` is a folder, so the config is a frozen
       v1.0 config of THIS flavour (not the open-flavour :data:`FROZEN_V10_CONFIGS`);
    2. ``v1_cards`` (the v1.0 card's ``configs:``) has an entry for the config that is
       ``canonical``-equal to ``build_entry`` (v1.1.0 inherited it unchanged, the test of
       ``gates.frozen``);
    3. the pattern matches no file under ``v1_dir`` either (it is already a phantom in v1.0);
    4. the config still resolves at least one split: a config that loads nothing never passes.
    """
    config = item.config
    if (
        v1_dir is None
        or not item.splits
        or not _is_v1_config(v1_dir, config)
        or build_entry is None
        or config not in v1_cards
        or canonical(v1_cards[config]) != canonical(build_entry)
    ):
        return frozenset()
    return frozenset(pair for pair in item.missing if not _matches(v1_dir, pair[1]))


def plan_configs(
    card_configs: Mapping[str, object], build_dir: Path, v1_dir: Path | None = None
) -> list[ConfigFiles]:
    """Every card config in README order (the first one decides how the Hub infers the builder).
    With ``v1_dir``, the inherited v1.0 phantom patterns move from ``missing`` to ``inherited``."""
    build_dir = Path(build_dir)
    plan = [resolve_config(c, e, build_dir, v1_dir) for c, e in card_configs.items()]
    if v1_dir is None:
        return plan
    build_cards, v1_cards = _written_entries(build_dir), _written_entries(Path(v1_dir))
    out = []
    for item in plan:
        phantoms = inherited_phantoms(item, build_cards.get(item.config), Path(v1_dir), v1_cards)
        out.append(
            replace(
                item,
                missing=tuple(m for m in item.missing if m not in phantoms),
                inherited=tuple(m for m in item.missing if m in phantoms),
            )
            if phantoms
            else item
        )
    return out
