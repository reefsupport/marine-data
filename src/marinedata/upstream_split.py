"""Upstream split names -> ours (WP-R2b step 2, normaliser half).

``TEST``/``Test``/``testing`` -> ``test``; ``val``/``valid``/``validation`` -> ``val``;
``train``/``training`` -> ``train``; anything else (``dev``, ``trainval``, empty) -> ``None``
(unknown, never guessed).
:func:`upstream_test_groups` is the input a "honour upstream test" split rule needs: every split
group that contains at least one upstream-test row goes to OUR test split as a whole group.
NOT yet wired into the split allocator (see the WP-R2b report).
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
