"""HK-4a: every registry source carries exactly one ``domain``; the five fish/fauna box
sources staged without an entry are registered with an evidence-backed class."""

from __future__ import annotations

import pytest

from marinedata.enums import Domain
from marinedata.registry import Registry


@pytest.fixture(scope="module")
def sources() -> dict:
    registry = Registry.load()
    return {s.id: s for s in registry.sources}


def test_every_registry_source_has_a_domain(sources) -> None:
    missing = sorted(sid for sid, s in sources.items() if s.domain is None)
    assert not missing, f"no domain on: {missing}"


def test_domain_vocabulary_is_the_eight_buckets() -> None:
    assert {d.value for d in Domain} == {
        "coral", "fish", "seagrass", "mangrove", "plankton", "deep-sea", "fauna", "imaging",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("sid", "access_class", "domain"),
    [
        ("uiis", "open", "fauna"),
        ("uiis10k", "open", "fauna"),
        ("usis10k", "open", "fauna"),
        ("roboflow-aquarium", "open", "fauna"),
        ("obsea-fish", "unknown", "fish"),  # licence not read from a primary object: excluded
    ],
)
def test_staged_fish_box_sources_are_registered(sources, sid, access_class, domain) -> None:
    source = sources[sid]
    assert source.access_class.value == access_class
    assert source.domain is not None and source.domain.value == domain
