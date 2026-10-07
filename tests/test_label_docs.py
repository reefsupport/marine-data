"""docs/LABELS.md must match the tables generated from registry/label-schemes/*.yaml (offline)."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gen_label_docs", ROOT / "scripts" / "gen_label_docs.py"
)
assert SPEC is not None and SPEC.loader is not None
gen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gen)

DOC = (ROOT / "docs" / "LABELS.md").read_text()


def test_every_generated_block_is_present_once():
    for name in gen.BLOCKS:
        begin, end = gen.markers(name)
        assert DOC.count(begin) == 1 and DOC.count(end) == 1, name


def test_labels_doc_is_in_sync_with_the_registry():
    """Fails when a scheme, crosswalk or source table changed and the doc was not regenerated."""
    assert gen.render(DOC) == DOC, "docs/LABELS.md is stale: run `python scripts/gen_label_docs.py`"


def test_generated_tables_name_every_source_and_scheme():
    from marinedata import label_schemes as ls

    bodies = gen.generate()
    for scheme, spec in ls.schemes().items():
        for block in ("label-classes", "label-sources", "label-mapping"):
            assert f"#### `{scheme}`" in bodies[block]
        for source in spec.sources:
            assert f"`{source}`" in bodies["label-sources"]
            assert f"`{source}`" in bodies["label-mapping"]


def test_doc_does_not_name_the_private_repository():
    assert not re.search(r"marine-data-nc|\bnc\b", DOC)


def test_splice_rejects_a_doc_without_markers():
    with pytest.raises(ValueError, match="no <!-- BEGIN GENERATED"):
        gen.splice("# no markers\n", "label-classes", "x")
