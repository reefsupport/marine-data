"""Errors and helpers ``build_release`` relies on: skipped sources, split groups, records."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path

from ..loaders.generic import IMAGE_SUFFIXES
from ..models import Source
from ..registry import Registry
from .entries import release_skip_reason


class ReleaseSkipError(ValueError):
    """A releasable source with rows would be silently left out of a flavour build (WP-R2e)."""


def _loader_layout(source: Source) -> str | None:
    return source.loader.layout if source.loader is not None else None


def _memoized(digest: Callable[[Path], str]) -> Callable[[Path], str]:
    """``digest`` computed once per path: the group hook, the near-dup check and the manifest
    loop all hash the same files, and a 26 GB source must not be read three times."""
    seen: dict[Path, str] = {}

    def cached(path: Path) -> str:
        key = Path(path)
        if key not in seen:
            seen[key] = digest(key)
        return seen[key]

    return cached


def _with_release_group(  # type: ignore[no-untyped-def]
    sample, groups: dict[str, dict[str, str]], roots: dict[str, Path]
):
    """``sample`` with ``meta["split_group"]`` set from the enumeration the split map was
    generated from (a loader that is not ``staged-tree`` never sets one). A pair is one
    sample: its mask/depth file rides along with the primary image's group. The lookup is by
    image path relative to the source root, because ``Sample.key`` is layout-specific (a COCO
    ``file_name`` or a CSV image name is relative to ``images_dir``, not to the root); the key
    stays the fallback."""
    by_path = groups.get(sample.source_id)
    if by_path is None or sample.meta.get("split_group"):
        return sample
    group = None
    if sample.image is not None and sample.source_id in roots:
        try:
            group = by_path.get(str(Path(sample.image).relative_to(roots[sample.source_id])))
        except ValueError:
            group = None
    if group is None:
        group = by_path.get(sample.key)
    return sample if group is None else replace(sample, meta={**sample.meta, "split_group": group})


class SplitGroupError(ValueError):
    """A sample reached the group split without a ``split_group`` (fail closed, WP-R6)."""


def require_split_groups(samples: Iterable, task_id: str) -> None:  # type: ignore[type-arg]
    """Raise :class:`SplitGroupError` naming every source with a sample that carries no
    ``meta["split_group"]`` after :func:`_with_release_group`, instead of a bare error from the
    split. Shared by :func:`build_release` and ``hf_export`` so both fail on the same input."""
    missing: dict[str, int] = {}
    for sample in samples:
        if not sample.meta.get("split_group"):
            missing[sample.source_id] = missing.get(sample.source_id, 0) + 1
    if missing:
        listing = ", ".join(f"{sid} ({n} sample(s))" for sid, n in sorted(missing.items()))
        raise SplitGroupError(
            f"task {task_id!r}: no split_group for source(s) {listing}; the split map cannot place "
            "them (its loader sets none and the release enumeration has no entry for them)"
        )


def _has_rows(root: Path) -> bool:
    """True when the staged tree holds at least one row (``metadata.parquet``) or, without one,
    at least one image file."""
    index = root / "metadata.parquet"
    if index.is_file():
        import pyarrow.parquet as pq

        return pq.ParquetFile(index).metadata.num_rows > 0
    return root.is_dir() and any(
        p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES for p in root.rglob("*")
    )


def _skipped_sources(
    registry: Registry,
    admitted: list[str],
    roots: dict[str, str | Path],
    allow_skip: frozenset[str],
) -> dict[str, tuple[str, str]]:
    """``{source_id: (reason, allowed_by)}`` for every admitted source the build leaves out.

    The hard gate (WP-R2e): a source the split map cannot cover that has rows > 0 raises unless
    ``--allow-skip`` names it or its registry carries ``release_skip_reason``; a source with no
    rows is recorded (``allowed_by`` = ``no-rows``, reason ``no-staged-images`` when the split map
    would have covered it) and never blocks."""
    skipped: dict[str, tuple[str, str]] = {}
    blocked: list[str] = []
    for sid in admitted:
        source = registry.source(sid)
        reason = release_skip_reason(source)
        if sid in allow_skip:
            skipped[sid] = (reason or "--allow-skip: skipped on request", "allow-skip")
        elif reason is None:
            if not _has_rows(Path(roots[sid])):  # WP-R4: no rows to build, so nothing to judge
                skipped[sid] = (f"no-staged-images: no staged rows under {roots[sid]}", "no-rows")
            continue
        elif source.release_skip_reason:
            skipped[sid] = (reason, "registry")
        elif not _has_rows(Path(roots[sid])):
            skipped[sid] = (reason, "no-rows")
        else:
            blocked.append(sid)
    if blocked:
        raise ReleaseSkipError(
            "releasable source(s) with rows the split map cannot cover would be silently "
            "dropped: "
            + "; ".join(f"{sid} ({release_skip_reason(registry.source(sid))})" for sid in blocked)
            + " -- cover them, set a registry release_skip_reason, or pass --allow-skip SOURCE_ID"
        )
    return skipped


def _split_ratio_record(map_bytes: bytes) -> dict:
    """The train/validation/test share (percent) the split map was allocated with; the splits
    gate checks each config against it unless the registry (split_ratio.yaml) names one."""
    import json as _json

    ratios = _json.loads(map_bytes).get("ratios") or {}
    total = sum(ratios.values()) or 1.0
    return {
        "source": "split_map",
        "target_pct": {k: round(100.0 * v / total, 2) for k, v in sorted(ratios.items())},
    }
