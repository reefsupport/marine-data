"""WP-R5/R5b: the canonical `atlantis-synthetic-depth` entry is rootable from the staged tree.

R4 found the registry entry had no s3 prefix and no root digest, so a release build could not
root it. It now declares the staged tree (`access.params.staged_*`, `sources/atlantis/<version>/`)
with the staged CHECKSUMS digest, keeps its licence class (restricted-nc, LICV 2026-10-05) and
keeps its fetch dispatch (`method: api`, `client: atlantis`; tests/test_fetch_api_dispatch.py
guards that).
"""

from __future__ import annotations

import marinedata.fetch as fetch_module
from marinedata.fetchers_remote import _is_pinned_staged_tree, staged_root_params
from marinedata.licence_class import source_class, staged_id
from marinedata.registry import Registry
from marinedata.release import release_skip_reason

CANONICAL = "atlantis-synthetic-depth"


def test_canonical_entry_is_a_pinned_staged_tree_at_the_staged_prefix() -> None:
    source = Registry.load().source(CANONICAL)
    staged = staged_root_params(source)
    assert staged is not None
    assert staged["prefix"] == f"sources/{staged_id(CANONICAL)}/{source.version}/"
    assert staged["bucket"] == "rs-storage-open"
    assert source.checksums is not None
    assert source.checksums.version == source.version
    assert len(source.checksums.root_digest) == 64
    assert _is_pinned_staged_tree(source)


def test_a_build_roots_it_and_its_licence_class_is_unchanged() -> None:
    source = Registry.load().source(CANONICAL)
    assert release_skip_reason(source) is None  # staged-tree layout: the split map can cover it
    assert source_class(CANONICAL) == "restricted-nc"
    assert source.access_class.value == "restricted-nc"


def test_fetch_dispatch_stays_api_but_a_release_roots_the_staged_tree(
    monkeypatch, tmp_path
) -> None:
    source = Registry.load().source(CANONICAL)
    assert source.access.method.value == "api"
    assert source.access.params["client"] == "atlantis"
    seen: list[tuple[str, int]] = []

    def fake_staged(src, root, limit):
        seen.append((src.id, limit))
        return fetch_module.FetchResult(src.id, root, 0, "s3-manifest", truncated=False)

    monkeypatch.setattr("marinedata.fetchers_remote.fetch_staged_root", fake_staged)
    result = fetch_module.fetch_sample(source, limit=7, root=tmp_path)
    assert seen == [(CANONICAL, 7)]
    assert result.method == "s3-manifest"
