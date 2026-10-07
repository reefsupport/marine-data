"""WP-R2d: no single-group default for a source without a registry SplitGroupRule.

A staged tree whose ``split_group`` column is null and whose source declares no pattern gets
the ``metadata_norm`` chain at release time (never ``<source>/<partition>``), and a source
that still ends up with < 10 groups for >= 100 rows falls back to sha256 per row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from test_upstream_test_split import _registry, _source

from marinedata.release import enumerate_release_rows
from marinedata.tables import StagedImage, write_metadata_table


def _stage(root: Path, n: int, path_of) -> Path:  # type: ignore[no-untyped-def]
    staged = []
    for i in range(n):
        stem = f"s{i}"
        image = root / "images" / "p" / f"{stem}.img"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(f"img-{root.name}-{i}".encode())
        staged.append(
            StagedImage(
                stem=stem,
                partition="p",
                upstream_path=path_of(i),
                upstream_split="train",
                width=4,
                height=4,
                split_group=None,
            )
        )
    write_metadata_table(root / "metadata.parquet", staged)
    return root


def _groups(tmp_path: Path, n: int, path_of) -> list[str]:  # type: ignore[no-untyped-def]
    root = _stage(tmp_path / "src-a", n, path_of)
    registry = _registry({"src-a": _source("src-a")})
    return [g for _, g, _ in enumerate_release_rows(registry, {"src-a": root})]


def test_null_group_without_a_rule_uses_the_norm_chain_not_the_default(tmp_path: Path) -> None:
    groups = _groups(tmp_path, 40, lambda i: f"orig/f{i % 12}/s{i}.jpg")
    assert set(groups) == {f"src-a/train/f{k}" for k in range(12)}
    assert "src-a/p" not in groups


def test_few_groups_for_many_rows_warns_and_groups_per_image(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="marinedata.release"):
        groups = _groups(tmp_path, 120, lambda i: f"orig/f{i % 2}/s{i}.jpg")
    assert len(set(groups)) == 120
    assert all(g.startswith("src-a/sha:") for g in groups)
    assert any("src-a" in r.getMessage() and "120 rows" in r.getMessage() for r in caplog.records)


def test_guard_does_not_fire_below_the_row_floor_or_with_enough_groups(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="marinedata.release"):
        few = _groups(tmp_path / "a", 99, lambda i: f"orig/f{i % 2}/s{i}.jpg")
        many = _groups(tmp_path / "b", 120, lambda i: f"orig/f{i % 12}/s{i}.jpg")
    assert set(few) == {"src-a/train/f0", "src-a/train/f1"}
    assert len(set(many)) == 12
    assert not caplog.records
