"""WP-L1a: licence classes, per-row resolver, flavour filter, applied registry classes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from marinedata import licence_class as lc
from marinedata.licence_class import (
    INTERNAL_ONLY,
    NC,
    ND,
    OPEN,
    UNKNOWN,
    flavour_filter,
    licence_class_of,
    resolve_row_class,
    source_class,
    source_classes,
)
from marinedata.registry import Registry

REGISTRY = Path(__file__).resolve().parents[1] / "registry"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CC0-1.0", OPEN),
        ("CC0", OPEN),
        ("PDM", OPEN),
        ("PD", OPEN),
        ("US-GOV-PD", OPEN),
        ("public domain (US government work)", OPEN),
        ("CC-BY-4.0", OPEN),
        ("CC BY 3.0", OPEN),
        ("CC-BY-SA-4.0", OPEN),
        ("cc-by-sa-4.0", OPEN),
        ("Apache-2.0", OPEN),
        ("MIT", OPEN),
        ("ODbL-1.0", OPEN),
        ("Creative Commons Attribution 4.0 International", OPEN),
        ("CC-BY-NC-4.0", NC),
        ("CC BY-NC-SA 3.0", NC),
        ("Creative Commons Attribution-NonCommercial 4.0", NC),
        ("CC-BY-ND-4.0", ND),
        ("CC-BY-NC-ND-4.0", ND),
        ("Creative Commons Attribution-Noncommercial-No Derivatives 4.0", ND),
        ("CC-BY-4.0 images, CC-BY-NC-4.0 labels", NC),
        ("", UNKNOWN),
        (None, UNKNOWN),
        ("   ", UNKNOWN),
        ("FathomNet", UNKNOWN),
        ("none stated", UNKNOWN),
        ("permit", UNKNOWN),
        ("all rights reserved", UNKNOWN),
        ("CC-BY-4.0, all rights reserved", UNKNOWN),
    ],
)
def test_licence_class_of_families(text, expected):
    assert licence_class_of(text) == expected


def test_resolver_source_class_vs_row_licence():
    # not per-row: the row wins only when stricter; missing/unparseable = no information
    assert resolve_row_class(OPEN, "CC-BY-NC-4.0") == NC
    assert resolve_row_class(OPEN, "CC-BY-NC-ND-4.0") == ND
    assert resolve_row_class(NC, "CC0-1.0") == NC
    assert resolve_row_class(OPEN, None) == OPEN
    assert resolve_row_class(OPEN, "FathomNet") == OPEN
    assert resolve_row_class(INTERNAL_ONLY, "CC0-1.0") == INTERNAL_ONLY
    assert resolve_row_class("garbage", None) == UNKNOWN
    # per-row: the row wins outright; empty / unparseable = unknown
    assert resolve_row_class(ND, "CC0-1.0", per_row=True) == OPEN
    assert resolve_row_class(OPEN, "CC-BY-NC-4.0", per_row=True) == NC
    assert resolve_row_class(OPEN, None, per_row=True) == UNKNOWN
    assert resolve_row_class(OPEN, "FathomNet", per_row=True) == UNKNOWN
    assert resolve_row_class(OPEN, "CC0-1.0", "CC-BY-NC-SA-4.0", per_row=True) == NC


def _row(source: str, cls: str | None, **extra):
    attrs = {} if cls is None else {"licence_class": cls}
    return {
        "source_id": source,
        "ann_id": f"{source}:{extra.get('n', 0)}",
        "attrs": json.dumps(attrs),
    }


MIXED = [
    _row("a-open", OPEN, n=1),
    _row("a-open", OPEN, n=2),
    _row("b-nc", NC, n=1),
    _row("b-nc", NC, n=2),
    _row("c-nd", ND),
    _row("d-internal", INTERNAL_ONLY),
    _row("e-unknown", UNKNOWN),
    _row("f-nothing", None),
]


def test_flavour_filter_open_keeps_only_open():
    got = flavour_filter(MIXED, "open", classes={})
    assert [r["source_id"] for r in got] == ["a-open", "a-open"]


def test_flavour_filter_nc_is_the_delta_only():
    got = flavour_filter(MIXED, "nc", classes={})
    assert [r["source_id"] for r in got] == ["b-nc", "b-nc"]
    assert not {r["ann_id"] for r in got} & {
        r["ann_id"] for r in flavour_filter(MIXED, "open", classes={})
    }


@pytest.mark.parametrize("flavour", ["open", "nc"])
def test_nothing_nd_internal_or_unknown_leaks(flavour):
    leaked = {r["source_id"] for r in flavour_filter(MIXED, flavour, classes={})}
    assert not leaked & {"c-nd", "d-internal", "e-unknown", "f-nothing"}


def test_flavour_filter_rejects_unknown_flavour():
    with pytest.raises(ValueError):
        flavour_filter(MIXED, "full")


def test_flat_class_field_and_objects_and_source_fallback():
    class Obj:
        source_id = "zz-open"
        licence_class = None

    assert flavour_filter([{"source_id": "x", "licence_class": NC}], "nc") == [
        {"source_id": "x", "licence_class": NC}
    ]
    assert flavour_filter([Obj()], "open", classes={"zz-open": OPEN}) != []
    assert flavour_filter([Obj()], "nc", classes={"zz-open": OPEN}) == []


def test_fathomnet_per_row_and_never_a_source_class_fallback():
    assert lc.per_row_source("fathomnet", REGISTRY)
    assert source_class("fathomnet", root=REGISTRY) == ND  # the bound, never released as such
    rows = [
        {"source_id": "fathomnet", "licence_class": resolve_row_class(ND, s, per_row=True)}
        for s in ("CC0-1.0", "CC-BY-4.0", "CC-BY-NC-4.0", "CC-BY-NC-ND-4.0", None, "FathomNet")
    ]
    assert [r["licence_class"] for r in rows] == [OPEN, OPEN, NC, ND, UNKNOWN, UNKNOWN]
    assert len(flavour_filter(rows, "open")) == 2
    assert len(flavour_filter(rows, "nc")) == 1
    # a fathomnet row that carries no class of its own is never released (no source fallback)
    bare = [{"source_id": "fathomnet"}, {"source_id": "inat-marine"}, {"source_id": "qut-fish"}]
    assert flavour_filter(bare, "open") == [] and flavour_filter(bare, "nc") == []


def test_kaggle_bleaching_is_internal_and_dropped():
    assert source_class("kaggle-healthy-bleached-corals", root=REGISTRY) == INTERNAL_ONLY
    row = {"source_id": "kaggle-healthy-bleached-corals", "attrs": {"licence_class": INTERNAL_ONLY}}
    assert flavour_filter([row], "open") == [] and flavour_filter([row], "nc") == []
    bare = {"source_id": "kaggle-healthy-bleached-corals"}
    assert flavour_filter([bare], "open") == [] and flavour_filter([bare], "nc") == []


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("large-scale-fish", OPEN),
        ("marineevt", OPEN),
        ("seatizen-atlas", OPEN),
        ("mouss-seg", OPEN),
        ("noaa-coral-icra", OPEN),
        ("plasticinwater", OPEN),
        ("caddy", NC),
        ("mermaid-aws", NC),
        ("kaggle-healthy-bleached-corals", INTERNAL_ONLY),
        ("coralscop-masks-rs", INTERNAL_ONLY),
        ("eilat-rsmas", INTERNAL_ONLY),
        ("deepseagrass", ND),
    ],
)
def test_lic_ab_corrections_applied(source, expected):
    assert source_class(source, root=REGISTRY) == expected


def test_every_registry_and_ingest_spec_id_has_a_valid_class():
    import yaml

    registry = Registry.load(REGISTRY)
    spec_ids = []
    for path in sorted((REGISTRY / "ingest-specs").glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        if isinstance(data, dict) and "id" in data:
            assert data.get("access_class") in lc.ACCESS_CLASSES, path.name
            spec_ids.append(data["id"])
    classes = source_classes(REGISTRY)
    assert len(spec_ids) >= 200 and all(s in classes for s in spec_ids)
    assert all(s.access_class.value in lc.ACCESS_CLASSES for s in registry.sources)
    # the registry and its ingest spec never disagree
    for s in registry.sources:
        if s.id in spec_ids:
            assert classes[s.id] == s.access_class.value


@pytest.mark.parametrize("source", ["floating-marine-debris", "noaa-dsc-csv", "large-scale-fish"])
@pytest.mark.parametrize("flavour", ["open", "nc"])
def test_release_excluded_sources_ship_in_no_flavour(source, flavour):
    assert lc.release_excluded(source, root=REGISTRY)
    row = {"source_id": source, "licence_class": OPEN if flavour == "open" else NC}
    assert flavour_filter([row], flavour) == []


def test_release_exclusion_flag_does_not_touch_other_sources():
    assert not lc.release_excluded("mermaid-aws", root=REGISTRY)
    assert not lc.release_excluded("not-a-source", root=REGISTRY)
    row = {"source_id": "zz-open", "licence_class": OPEN}
    assert flavour_filter([row], "open") == [row]
