"""``QueryResult.summary()`` and the ``list`` / ``show`` CLI read the source count fields.

Regression: these used a non-existent ``Source.items`` attribute and crashed with
``AttributeError`` as soon as a result held a source.
"""

from __future__ import annotations

import pytest

import marinedata as md
from marinedata.cli import main
from marinedata.registry import Registry


def _counted_source_id() -> str:
    registry = Registry.load()
    for source in registry.sources:
        if source.n_images:
            return source.id
    pytest.fail("no registry source carries n_images")


def test_summary_renders_matched_sources_and_total() -> None:
    result = md.find(profile="ship-commercial")
    assert len(result) > 0
    text = result.summary()
    assert text.startswith("profile=ship-commercial")
    for source in result:
        assert source.id in text
    assert result.total_items == sum(s.primary_count or 0 for s in result.sources)
    assert result.total_items > 0


def test_min_items_filters_on_image_count() -> None:
    everything = md.find(profile="ship-commercial")
    large = md.find(profile="ship-commercial", min_items=10_000)
    assert 0 < len(large) < len(everything)
    assert all((s.primary_count or 0) >= 10_000 for s in large)


def test_list_cli_prints_summary(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list", "--profile", "ship-commercial"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("profile=ship-commercial")
    assert "matched=" in out


def test_show_cli_prints_item_count(capsys: pytest.CaptureFixture[str]) -> None:
    source_id = _counted_source_id()
    assert main(["show", source_id]) == 0
    out = capsys.readouterr().out
    assert source_id in out
    assert "items" in out
