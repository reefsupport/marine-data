"""Upstream split names -> ours (WP-R2b step 2, normaliser half).

``TEST``/``Test``/``testing`` -> ``test``; ``val``/``valid``/``validation`` -> ``val``;
``train``/``training`` -> ``train``; anything else (``dev``, ``trainval``, empty) -> ``None``
(unknown, never guessed).
:func:`upstream_test_groups` is the input a "honour upstream test" split rule needs: every split
group that contains at least one upstream-test row goes to OUR test split as a whole group.
:func:`honoured_test_groups` applies the per-source switch; ``release.generate_split_map`` feeds
the result to the allocator as forced-``test`` groups (WP-R2c).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

_CANON = {
    "test": "test", "testing": "test", "tst": "test",
    "val": "val", "valid": "val", "validation": "val",
    "train": "train", "training": "train",
}  # fmt: skip


def normalise_upstream_split(value: object) -> str | None:
    """``train`` | ``val`` | ``test`` for the common spellings, else ``None``."""
    if value is None:
        return None
    return _CANON.get(str(value).strip().lower().replace("_", "").replace("-", ""))


def upstream_test_groups(upstream_splits: Mapping[str, Iterable[str]]) -> frozenset[str]:
    """Split groups with any upstream-test row (``release.enumerate_release_rows`` fills
    ``{split_group: {raw upstream_split, ...}}``); the whole group is held out together."""
    return frozenset(
        group
        for group, raw in upstream_splits.items()
        if any(normalise_upstream_split(v) == "test" for v in raw)
    )


def honoured_test_groups(
    by_source: Mapping[str, Mapping[str, Iterable[str]]],
    *,
    honour: bool = True,
    off: Iterable[str] = (),
) -> frozenset[str]:
    """Upstream-test groups to force into our ``test`` split (WP-R2c).

    ``by_source`` = ``{source_id: {split_group: {raw upstream_split, ...}}}`` (what
    ``release.enumerate_release_rows`` fills). A source with no upstream-test row contributes
    nothing, so the rule is on by default only where an upstream test set exists. ``honour=False``
    or a ``"*"`` entry in ``off`` switches it off everywhere, any other ``off`` entry per source.
    """
    skip = set(off)
    if not honour or "*" in skip:
        return frozenset()
    groups: set[str] = set()
    for source_id, upstream_splits in by_source.items():
        if source_id not in skip:
            groups |= upstream_test_groups(upstream_splits)
    return frozenset(groups)
