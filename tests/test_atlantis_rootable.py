"""WP-R5: the canonical `atlantis-synthetic-depth` entry is rootable from the staged tree.

R4 found the registry entry had no s3 prefix and no root digest, so a release build could not root
it. It now points at the staged tree (`sources/atlantis/<version>/`) with the staged CHECKSUMS
digest, and keeps its licence class (restricted-nc, LICV 2026-10-05).
"""

from __future__ import annotations

from marinedata.fetchers_remote import _is_pinned_staged_tree
from marinedata.licence_class import source_class, staged_id
from marinedata.registry import Registry
from marinedata.release import release_skip_reason

CANONICAL = "atlantis-synthetic-depth"


def test_canonical_entry_is_a_pinned_staged_tree_at_the_staged_prefix() -> None:
    source = Registry.load().source(CANONICAL)
    assert source.access.method.value == "s3"
    assert source.access.params["prefix"] == f"sources/{staged_id(CANONICAL)}/{source.version}/"
    assert (
        source.access.uri
        == f"s3://{source.access.params['bucket']}/{source.access.params['prefix']}"
    )
    assert source.checksums is not None
    assert source.checksums.version == source.version
    assert len(source.checksums.root_digest) == 64
    assert _is_pinned_staged_tree(source)


def test_a_build_roots_it_and_its_licence_class_is_unchanged() -> None:
    source = Registry.load().source(CANONICAL)
    assert release_skip_reason(source) is None  # staged-tree layout: the split map can cover it
    assert source_class(CANONICAL) == "restricted-nc"
    assert source.access_class.value == "restricted-nc"
