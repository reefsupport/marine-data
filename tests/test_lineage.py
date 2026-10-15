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

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.builder import DatasetBuilder, SourcesDroppedError
from marinedata.cli import main as cli_main
from marinedata.fetch import sample_digest
from marinedata.lineage import build_lineage
from marinedata.query import find


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


# ── MD-3: four more defects — the record must describe the build that happened ──────
#
# `build_streaming` listed sources that contributed nothing (falling back to the
# *declared* count), `content_hash` omitted the one input the TDM path demands, a dirty
# registry tree could report a clean commit, and `sample_digest` was blind to a file
# moving between subdirectories at the same size.


def test_streaming_lineage_does_not_list_a_source_that_failed_to_load(
    registry: Registry, tmp_path: Path
) -> None:
    """⭐ The defect: `build_streaming` passed the gate-*permitted* list to
    `build_lineage`, not the list of sources that actually yielded a sample — a source
    whose loader raised `DataNotAvailable` was silently dropped from the corpus yet
    still appeared in `lineage.datasets`, falling back to its declared item count. D2:
    this must also fail loud (raise) rather than return as if nothing had happened."""
    from marinedata.tables import StagedImage, write_metadata_table

    # coralscapes declares `loader.layout: staged-tree` on main, so the good source is a
    # staged tree (images/ + labels/masks/ + metadata.parquet), as in test_builder.mixed_roots.
    labelled = tmp_path / "coralscapes"
    rows = []
    for i in range(3):
        stem = f"f{i}"
        (labelled / "images" / "default").mkdir(parents=True, exist_ok=True)
        (labelled / "labels" / "masks" / "default").mkdir(parents=True, exist_ok=True)
        (labelled / "images" / "default" / f"{stem}.jpg").write_bytes(b"\x89PNG\r\n\x1a\n")
        (labelled / "labels" / "masks" / "default" / f"{stem}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
        )
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
    write_metadata_table(labelled / "metadata.parquet", rows)
    empty = tmp_path / "sweet-corals"  # exists, but empty → DataNotAvailable
    empty.mkdir()

    builder = DatasetBuilder(
        registry,
        profile="research",
        roots={"coralscapes": labelled, "sweet-corals": empty},
    )

    with pytest.raises(SourcesDroppedError) as excinfo:
        builder.build_streaming(tolerance=None)

    assert set(excinfo.value.dropped) == {"sweet-corals"}
    lineage = excinfo.value.dataset.lineage
    assert [d.id for d in lineage.datasets] == ["coralscapes"], (
        "the failed source must not be reported as a contributor"
    )
    assert lineage.dropped == [
        {"id": "sweet-corals", "reason": excinfo.value.dropped["sweet-corals"]}
    ]
    assert "sweet-corals" in lineage.dropped[0]["reason"]  # a real reason, not a stub


def test_content_hash_changes_when_legal_opinion_ref_does(registry: Registry, one_source) -> None:
    """⭐ The defect: `content_hash` bound `{profile, datasets, excluded,
    registry_commit}` but not `legal_opinion_ref` — the one fact `gate.py`'s TDM path
    requires on record (source-level `legal_basis` is already inside `datasets`, but
    *which* counsel opinion, or whether one was supplied at all, was not)."""
    profile = registry.profile("research")
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with_ref_a = build_lineage(one_source, profile, legal_opinion_ref="opinion-1", now=stamp)
    with_ref_b = build_lineage(one_source, profile, legal_opinion_ref="opinion-2", now=stamp)
    without_ref = build_lineage(one_source, profile, legal_opinion_ref=None, now=stamp)

    assert with_ref_a.content_hash != with_ref_b.content_hash
    assert with_ref_a.content_hash != without_ref.content_hash


def test_git_commit_detects_a_dirty_tree(tmp_path: Path) -> None:
    """⭐ The defect: `_git_commit` never checked for uncommitted changes, so a dirty
    registry (e.g. a locally edited licence YAML never committed) reported a
    clean-looking hash. Uses a throwaway repo under `tmp_path` — never the real
    registry — so this leaves nothing dirty behind."""
    from marinedata.registry import _git_commit

    repo = tmp_path / "repo"
    repo.mkdir()

    def run(*args: str) -> None:
        subprocess.run(args, cwd=repo, check=True, capture_output=True)

    run("git", "init", "-q")
    run("git", "config", "user.email", "test@example.com")
    run("git", "config", "user.name", "Test")
    (repo / "licence.yaml").write_text("id: X\n")
    run("git", "add", ".")
    run("git", "commit", "-q", "-m", "initial")

    clean = _git_commit(repo)
    assert clean is not None
    assert not clean.endswith("+dirty")

    (repo / "licence.yaml").write_text("id: X\nno_derivatives: true\n")  # uncommitted edit
    dirty = _git_commit(repo)
    assert dirty is not None
    assert dirty.endswith("+dirty")
    assert dirty.removesuffix("+dirty") == clean, "same commit — only the caveat changes"


def test_sample_digest_changes_when_a_file_moves_between_subdirectories(
    tmp_path: Path,
) -> None:
    """⭐ The defect: `sample_digest` hashed `path.name` only, not the path relative to
    `root` — moving `a/img.jpg` to `b/img.jpg` left the digest unchanged."""
    root = tmp_path / "sample"
    (root / "a").mkdir(parents=True)
    (root / "a" / "img.jpg").write_bytes(b"\x01\x02\x03\x04")
    before = sample_digest(root)

    (root / "b").mkdir()
    (root / "a" / "img.jpg").rename(root / "b" / "img.jpg")
    after = sample_digest(root)

    assert before != after


def test_lineage_cli_exits_zero_on_denials_but_still_traces_them(
    registry: Registry, capsys: pytest.CaptureFixture[str]
) -> None:
    """E7: a denial is the gate working, not a failure. `ship-commercial` denies most of
    the registry by construction, so an exit code that fires on nearly every normal run
    gets suppressed within a week — the CLI must exit 0 here. A test that only checks
    `== 0` would also pass on a version that deleted the warning, so this pins the trace
    too: the denied ids must still show up on stderr *and* in the JSON `excluded`."""
    result = find(profile="ship-commercial", registry=registry)
    denied_ids = [d.source_id for d in result.excluded]
    assert denied_ids, "fixture assumption: ship-commercial denies at least one source"

    code = cli_main(["lineage", "--profile", "ship-commercial"])
    captured = capsys.readouterr()

    assert code == 0
    for source_id in denied_ids:
        assert f"✗ {source_id}:" in captured.err
    payload = json.loads(captured.out)
    assert {e["id"] for e in payload["excluded"]} == set(denied_ids)
