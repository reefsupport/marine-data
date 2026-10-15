"""Malformed-registry-file error paths.

`registry.py`'s module docstring promises "an invalid entry fails the whole load
rather than being skipped, because a silently dropped source is indistinguishable
from one that was never added." A wrong top-level key (`source:` instead of
`sources:`) used to violate that silently: the file loads, validates as a dict,
`.get("sources", [])` returns `[]`, and nothing raises. These tests are the
regression guard.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.registry import Registry, RegistryError


def test_sources_file_with_wrong_top_level_key_raises(tmp_path: Path) -> None:
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    bad_file = sources_dir / "typo.yaml"
    bad_file.write_text("source:\n  - id: nope\n", encoding="utf-8")

    with pytest.raises(RegistryError) as exc_info:
        Registry._load_sources(sources_dir, {})
    assert str(bad_file) in str(exc_info.value)


def test_sources_file_with_correct_key_and_empty_list_does_not_raise(tmp_path: Path) -> None:
    """The per-file check must not regress the legitimate case of a placeholder file."""
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    (sources_dir / "empty.yaml").write_text("sources: []\n", encoding="utf-8")

    with pytest.raises(RegistryError, match="No sources found"):
        # Still raises — but from the *aggregate* empty check, not the per-file one.
        Registry._load_sources(sources_dir, {})


def test_collection_file_with_wrong_top_level_key_raises(tmp_path: Path) -> None:
    base = tmp_path / "schemas"
    base.mkdir()
    bad_file = base / "typo.yaml"
    bad_file.write_text("schema:\n  - id: nope\n", encoding="utf-8")

    with pytest.raises(RegistryError) as exc_info:
        Registry._load_collection(base, "schemas", object)
    assert str(bad_file) in str(exc_info.value)


def test_collection_file_with_correct_key_and_empty_list_does_not_raise(tmp_path: Path) -> None:
    base = tmp_path / "schemas"
    base.mkdir()
    (base / "empty.yaml").write_text("schemas: []\n", encoding="utf-8")

    assert Registry._load_collection(base, "schemas", object) == {}
