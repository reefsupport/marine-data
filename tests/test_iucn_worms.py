"""WoRMS IUCN attribute extraction: category-name normalization and the nested
AphiaAttributes tree walk. No live network calls — WoRMS' actual response shapes are
reproduced as fixtures (verified live, 2026-09-25: `measurementValue` is the full IUCN
English name, e.g. "Critically Endangered", never the bare "CR" code)."""

from __future__ import annotations

from marinedata import iucn_worms as iw


def test_normalize_category_maps_full_name_to_code():
    assert iw._normalize_category("Critically Endangered") == "CR"
    assert iw._normalize_category("least concern") == "LC"
    assert iw._normalize_category("Data Deficient") == "DD"


def test_normalize_category_passes_through_unknown_values():
    assert iw._normalize_category("Some New Category") == "Some New Category"


def test_extract_category_walks_nested_children():
    attributes = [
        {
            "measurementType": "Habitat",
            "children": [
                {"measurementType": "IUCN Red List Category", "measurementValue": "Endangered"}
            ],
        }
    ]
    assert iw._extract_category(attributes) == "EN"


def test_extract_category_returns_none_when_attribute_absent():
    other = [{"measurementType": "Habitat", "measurementValue": "reef"}]
    assert iw._extract_category(other) is None
    assert iw._extract_category([]) is None
