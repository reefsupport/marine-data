"""Lineage report tests — T5: four defects found by reading the code, each confirmed
here before the fix and pinned after it.

The audit record's whole job is to let a stranger re-derive a compliance decision
without trusting the assertion. Before this pass it overstated what it proved: the
content hash claimed to bind a registry version it never received, item counts were the
registry's guess rather than what was actually read, the specific acts a licence barred
were dropped even though the tier alone cannot always tell you, and the sample digest
used to catch a wrong declared layout could not see a content change at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.builder import DatasetBuilder
from marinedata.fetch import sample_digest
from marinedata.lineage import build_lineage


@pytest.fixture
def one_source(registry: Registry):
    return [registry.source("reef-support-benthic-own")]


def test_registry_commit_is_actually_set(registry: Registry, one_source) -> None:
    """This repo IS a git checkout, so the commit must be real, not None."""
    assert registry.commit is not None
    lineage = build_lineage(
        one_source, registry.profile("research"), registry_commit=registry.commit
    )
    assert lineage.registry_commit == registry.commit
    assert lineage.registry_commit in lineage.to_json()


def test_registry_commit_absent_gracefully(one_source, registry: Registry, tmp_path: Path) -> None:
    """Outside a git checkout (a packaged install) this must not raise."""
    from marinedata.registry import _git_commit

    assert _git_commit(tmp_path) is None  # tmp_path is never a git repo


def test_content_hash_changes_when_registry_commit_does(registry: Registry, one_source) -> None:
    """The hash is supposed to bind the registry version — prove it actually varies."""
    profile = registry.profile("research")
    a = build_lineage(
        one_source, profile, registry_commit="aaaa", now=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    b = build_lineage(
        one_source, profile, registry_commit="bbbb", now=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    assert a.content_hash != b.content_hash


def test_items_reports_what_was_consumed_not_declared(registry: Registry, one_source) -> None:
    """⭐ The defect: `items` used to always be the registry's estimate, even when a real
    build read a different (e.g. sampled) number of items."""
    source = one_source[0]
    declared = source.primary_count
    assert declared and declared > 5, "fixture assumption: this source declares a real count"

    consumed = build_lineage(
        one_source, registry.profile("research"), items_consumed={source.id: 5}
    )
    assert consumed.datasets[0].items == 5
    assert consumed.datasets[0].items_source == "consumed"

    declared_only = build_lineage(one_source, registry.profile("research"))
    assert declared_only.datasets[0].items == declared
    assert declared_only.datasets[0].items_source == "declared"


def test_licence_flags_are_present_on_every_entry(registry: Registry) -> None:
    """⭐ The defect: two T3_NONCOMMERCIAL sources can carry different obligations
    (share-alike, no-derivatives) that the tier string alone cannot distinguish."""
    nc_share_alike = registry.source("seatizen-atlas")  # open since WP-R2
    lineage = build_lineage([nc_share_alike], registry.profile("research"))
    entry = lineage.datasets[0]
    assert entry.licence_flags == nc_share_alike.licence.flags.model_dump()
    assert isinstance(entry.licence_flags["no_derivatives"], bool)


def test_build_populates_registry_commit_and_consumed_items(
    tmp_path: Path, registry: Registry
) -> None:
    """End-to-end: a real build() must thread both fixes through, not just the helper.

    coralscapes' registry entry declares `loader.layout: staged-tree` (S62, D-O), so
    this fixture is a staged tree: `images/`+`labels/masks/`+`metadata.parquet`.
    """
    from marinedata.tables import StagedImage, write_metadata_table

    root = tmp_path / "coralscapes"
    rows = []
    for i in range(3):
        stem = f"f{i}"
        (root / "images" / "default").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "masks" / "default").mkdir(parents=True, exist_ok=True)
        (root / "images" / "default" / f"{stem}.jpg").write_bytes(b"\x89PNG")
        (root / "labels" / "masks" / "default" / f"{stem}.png").write_bytes(b"\x89PNG")
        rows.append(
            StagedImage(
                stem=stem,
                partition="default",
                upstream_path=f"orig/{stem}.jpg",
                upstream_split=None,
                width=10,
                height=10,
                split_group=f"coralscapes/site{i}",
            )
        )
    write_metadata_table(root / "metadata.parquet", rows)

    dataset = DatasetBuilder(registry, profile="research", roots={"coralscapes": root}).build()
    entry = next(d for d in dataset.lineage.datasets if d.id == "coralscapes")
    assert entry.items == 3
    assert entry.items_source == "consumed"
    assert dataset.lineage.registry_commit == registry.commit


# ── sample_digest: content, not just name+size ──────────────────────────────


def test_sample_digest_changes_when_content_changes_at_same_size(tmp_path: Path) -> None:
    """⭐ The defect: hashing name+size means two same-size files with different bytes
    (a relabelled mask, a re-exported image) digest identically."""
    root = tmp_path / "sample"
    root.mkdir()
    (root / "f.png").write_bytes(b"\x00" * 16)
    first = sample_digest(root)

    (root / "f.png").write_bytes(b"\xff" * 16)  # same name, same size, different bytes
    second = sample_digest(root)

    assert first != second


def test_sample_digest_is_stable_for_identical_content(tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "f.png").write_bytes(b"\x01\x02\x03")
    (b / "f.png").write_bytes(b"\x01\x02\x03")
    assert sample_digest(a) == sample_digest(b)
