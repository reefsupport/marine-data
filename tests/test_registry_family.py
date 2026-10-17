"""``Source.family`` (D-FORK B): the optional slug the dedup gate reads from the registry."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from conftest import make_source
from pydantic import ValidationError

from marinedata import licence_class as lc
from marinedata.models import Source
from marinedata.registry import Registry, RegistryError

REGISTRY = Path(__file__).resolve().parents[1] / "registry"
FAMILY = "roboflow-coral-universe"
FORKS = (
    "roboflow-coral-bleaching-final-v6i",
    "roboflow-coral-bleaching-general-v1-yolov8s",
    "roboflow-coral-classification-copy-changed-v13i",
    "roboflow-coral-reef-classification-v3i",
)
SEAVIEW_FAMILY = "seaview-survey"
SEAVIEW = ("seaview-survey-imagery", "reef-support-seaview-labels")
SLUGS = ("roboflow-coral-universe", "a", "a_b-c9", "9lives")
NOT_SLUGS = ("", " ", "Roboflow", "-x", "_x", "a b", "a.b", "a/b", "ü")


def _entry(**fields) -> dict:
    """A valid registry source entry, as the YAML loader hands it to ``Source.model_validate``."""
    return {**make_source("image-mask-pairs").model_dump(mode="json"), **fields}


@pytest.fixture(scope="module")
def real_registry() -> Registry:
    return Registry.load(REGISTRY)


def test_the_four_roboflow_forks_declare_one_family(real_registry):
    assert {sid: real_registry.source(sid).family for sid in FORKS} == dict.fromkeys(FORKS, FAMILY)
    assert {sid: lc.source_family(sid, REGISTRY) for sid in FORKS} == dict.fromkeys(FORKS, FAMILY)


def test_the_two_seaview_sources_declare_one_family(real_registry):
    expected = dict.fromkeys(SEAVIEW, SEAVIEW_FAMILY)
    assert {sid: real_registry.source(sid).family for sid in SEAVIEW} == expected
    assert {sid: lc.source_family(sid, REGISTRY) for sid in SEAVIEW} == expected


def test_the_registry_has_exactly_the_roboflow_and_the_seaview_family():
    members: dict[str, set[str]] = {}
    for sid, family in lc.source_families(REGISTRY).items():
        members.setdefault(family, set()).add(sid)
    assert members == {FAMILY: set(FORKS), SEAVIEW_FAMILY: set(SEAVIEW)}


def test_light_scan_and_validated_registry_declare_the_same_families(real_registry):
    declared = {s.id: s.family for s in real_registry.sources if s.family}
    assert declared and lc.source_families(REGISTRY) == declared


def test_family_defaults_to_none_and_accepts_a_slug():
    assert Source.model_validate(_entry()).family is None
    assert Source.model_validate(_entry(family=FAMILY)).family == FAMILY


@pytest.mark.parametrize("family", NOT_SLUGS)
def test_family_must_be_a_slug(family):
    with pytest.raises(ValidationError):
        Source.model_validate(_entry(family=family))


@pytest.mark.parametrize("family", [*SLUGS, *NOT_SLUGS])
def test_light_scan_applies_the_same_slug_rule_as_the_model(tmp_path, family):
    (tmp_path / "sources").mkdir()
    entry = {"id": "fork-x", "access_class": "open", "family": family}
    (tmp_path / "sources" / "x.yaml").write_text(yaml.safe_dump({"sources": [entry]}))
    try:
        Source.model_validate(_entry(id="fork-x", family=family))
        valid = True
    except ValidationError:
        valid = False
    assert valid == (family in SLUGS)
    assert lc.source_families(tmp_path) == ({"fork-x": family} if valid else {})
    assert lc.source_family("fork-x", tmp_path) == (family if valid else None)


def test_sources_without_a_family_are_absent_from_the_map(tmp_path):
    (tmp_path / "sources").mkdir()
    entries = [
        {"id": "fork-a", "access_class": "open", "family": FAMILY},
        {"id": "fork-b", "access_class": "open", "family": FAMILY},
        {"id": "alone", "access_class": "open"},
    ]
    (tmp_path / "sources" / "x.yaml").write_text(yaml.safe_dump({"sources": entries}))
    assert lc.source_families(tmp_path) == {"fork-a": FAMILY, "fork-b": FAMILY}
    assert lc.source_family("alone", tmp_path) is None
    assert lc.source_family("not-a-source", tmp_path) is None


def test_registry_loader_accepts_a_family_and_rejects_a_bad_one(tmp_path):
    sources = tmp_path / "sources"
    sources.mkdir()
    entries = [_entry(id="fork-a", family=FAMILY), _entry(id="fork-b")]
    (sources / "ok.yaml").write_text(yaml.safe_dump({"sources": entries}))
    loaded = Registry._load_sources(sources, {})
    assert loaded["fork-a"].family == FAMILY and loaded["fork-b"].family is None

    (sources / "bad.yaml").write_text(
        yaml.safe_dump({"sources": [_entry(id="fork-c", family="Not A Slug")]})
    )
    with pytest.raises(RegistryError, match=r"bad\.yaml"):
        Registry._load_sources(sources, {})
