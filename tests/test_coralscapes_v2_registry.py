"""v1.2.0 registry facts of Coralscapes V2: open flavour, family, stem groups, link validation."""

from __future__ import annotations

import pytest

from marinedata.flavours import ships_in
from marinedata.gates.v11_runner import nc_source_ids
from marinedata.models import SourceLink
from marinedata.registry import Registry, RegistryError

V1, V2 = "coralscapes", "coralscapes-v2"


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


def test_v2_is_an_open_source_and_ships_in_the_open_flavour(reg):
    assert reg.source(V2).access_class == "open"
    assert ships_in(reg, V2, "open")
    assert V2 not in nc_source_ids()


def test_both_coralscapes_entries_declare_one_family(reg):
    assert reg.source(V1).family == reg.source(V2).family == "coralscapes"


def test_v2_split_group_is_one_group_per_stem(reg):
    v2 = reg.source(V2)
    stems = ("site12_000345_017682", "site12_000346_017690", "site36_000001_000002")
    groups = [
        v2.split_group_for(stem=s, upstream_path=f"images/default/{s}.png", partition="default")
        for s in stems
    ]
    assert groups == [f"{V2}/{s}" for s in stems]  # same site, different stems: different groups
    assert len(set(groups)) == 3


def test_v2_links_name_v1_by_stem(reg):
    links = reg.source(V2).links
    assert [(x.source_id, x.key, x.relation) for x in links] == [(V1, "stem", "re-annotation")]


def _check_links(reg: Registry, *links: SourceLink) -> None:
    sources = {s.id: s for s in reg.sources}
    sources[V2] = sources[V2].model_copy(update={"links": tuple(links)})
    Registry._check_references(sources, reg._schemas, reg._crosswalks)


def test_a_link_to_an_unknown_source_fails_at_load(reg):
    bad = SourceLink(source_id="no-such-source", key="stem", relation="re-annotation")
    with pytest.raises(RegistryError, match="links references 'no-such-source'"):
        _check_links(reg, bad)


def test_a_link_to_itself_fails_at_load(reg):
    with pytest.raises(RegistryError, match="links references 'coralscapes-v2'"):
        _check_links(reg, SourceLink(source_id=V2, key="stem", relation="re-annotation"))


def test_the_declared_link_passes_validation(reg):
    _check_links(reg, *reg.source(V2).links)
