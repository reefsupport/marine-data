"""The files a card's ``configs:`` entries name, resolved against the build and the v1.0 tree.

``datasets`` infers the builder of a local folder from the FIRST ``configs:`` entry of the README
and ignores the config asked for, and an additive release's build dir does not hold the frozen v1.0
configs (their shards live in the v1.0 tree). The load gate therefore reads the card itself: every
entry's patterns are resolved here (no ``datasets`` needed) against the build, a pattern that
matches nothing there falls back to ``--v1-dir``, and one that matches in neither is
``config_files_missing``."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

FROZEN_V10_CONFIGS = frozenset({"coral-masks", "scene-masks", "instance-masks", "fish-boxes"})
"""The v1.0 configs an additive release never rebuilds: their shards live in the v1.0 tree only."""


@dataclass(frozen=True)
class ConfigFiles:
    config: str
    splits: Mapping[str, tuple[Path, ...]]  # card split name -> files (build first, else v1.0)
    missing: tuple[tuple[str, str], ...]  # (split, pattern) that matched in neither tree
    from_v1: int  # files taken from the v1.0 tree

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


def plan_configs(
    card_configs: Mapping[str, object], build_dir: Path, v1_dir: Path | None = None
) -> list[ConfigFiles]:
    """Every card config in README order (the first one decides how the Hub infers the builder)."""
    return [resolve_config(c, e, Path(build_dir), v1_dir) for c, e in card_configs.items()]
