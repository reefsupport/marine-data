"""WP-13 Tier A: template determinism, "no fact -> no clause", and coverage counting."""

import pandas as pd

from marinedata.captions.facts import (
    CaptionFacts,
    build_facts_table,
    depth_band,
    rollup_points_to_facts,
)
from marinedata.captions.template import build_caption, coverage_report

FULL = CaptionFacts(
    image_sha256="abc123",
    source_id="noaa-pifsc-bleaching",
    habitat="coral_reef",
    depth_band="shallow (5-10 m)",
    meow_ecoregion="Hawaii",
    benthic_dominant="HC",
    benthic_present=("SD", "ALG"),
    bleaching_status="BLEACHED",
    top_taxa=("Acropora",),
)


def test_build_caption_is_deterministic():
    assert build_caption(FULL) == build_caption(FULL)
    # A second, independently-constructed identical facts object gives the same string.
    twin = CaptionFacts(**{**FULL.__dict__})
    assert build_caption(FULL) == build_caption(twin)


def test_build_caption_contains_every_populated_fact():
    caption = build_caption(FULL)
    for expected in (
        "noaa-pifsc-bleaching",
        "coral reef",
        "shallow (5-10 m)",
        "Hawaii",
        "hc",
        "bleached",
        "Acropora",
    ):
        assert expected.lower() in caption.lower()


def test_no_fact_no_clause_empty_facts_yields_empty_caption():
    bare = CaptionFacts(image_sha256="onlysha")
    assert build_caption(bare) == ""


def test_no_fact_no_clause_each_absent_field_omits_its_clause():
    only_habitat = CaptionFacts(image_sha256="x", habitat="coral_reef")
    caption = build_caption(only_habitat)
    assert "coral reef" in caption
    for absent in ("ecoregion", "bleach", "dominated", "visible"):
        assert absent not in caption.lower()


def test_depth_band_boundaries_and_none():
    assert depth_band(None) is None
    assert depth_band(3.0) == "very shallow (under 5 m)"
    assert depth_band(5.0) == "shallow (5-10 m)"
    assert depth_band(39.9).startswith("upper mesophotic")
    assert depth_band(100.0) == "deep (40 m or more)"


def test_n_facts_counts_source_as_one_fact():
    only_source = CaptionFacts(image_sha256="x", source_id="src")
    assert only_source.n_facts() == 1
    bare = CaptionFacts(image_sha256="x")
    assert bare.n_facts() == 0


def test_build_facts_table_gates_bleaching_on_resolved_label_only():
    metadata = pd.DataFrame(
        {
            "image_sha256": ["s1", "s2"],
            "source_id": ["noaa-pifsc-bleaching", "noaa-pifsc-bleaching"],
            "habitat": ["coral_reef", "coral_reef"],
        }
    )
    bleaching = pd.DataFrame(
        {
            "image_sha256": ["s1", "s2"],
            "label": ["BLEACHED", None],  # s2 is unresolved -> no fact, never guessed
        }
    )
    facts = build_facts_table(metadata, bleaching_condition=bleaching)
    by_sha = {f.image_sha256: f for f in facts}
    assert by_sha["s1"].bleaching_status == "BLEACHED"
    assert by_sha["s2"].bleaching_status is None


def test_rollup_points_to_facts_dominant_and_present():
    points = pd.DataFrame(
        {
            "image_id": ["i1"] * 10 + ["i2"] * 10,
            "class": ["HC"] * 6
            + ["SD"] * 3
            + ["ALG"] * 1  # i1: HC=60% (dominant)
            + ["HC"] * 4
            + ["SD"] * 4
            + ["ALG"] * 2,  # i2: no class >= 50% -> mixed
        }
    )
    facts = rollup_points_to_facts(
        points, image_col="image_id", class_col="class", source_id="mermaid-aws"
    )
    by_id = {f.image_sha256: f for f in facts}
    i1 = by_id["native:mermaid-aws:i1"]
    assert i1.benthic_dominant == "HC"
    assert i1.benthic_present == ("ALG", "HC", "SD")  # ALG at exactly 10% meets the threshold
    i2 = by_id["native:mermaid-aws:i2"]
    assert i2.benthic_dominant == "mixed"


def test_rollup_points_to_facts_drops_image_with_majority_unknown():
    points = pd.DataFrame({"image_id": ["i1"] * 10, "class": ["unknown"] * 6 + ["HC"] * 4})
    facts = rollup_points_to_facts(
        points,
        image_col="image_id",
        class_col="class",
        source_id="src",
        unknown_classes=frozenset({"unknown"}),
    )
    assert facts == []  # > 50% unknown -> excluded entirely


def test_coverage_report_counts_rows_at_or_above_threshold():
    facts = [
        FULL,
        CaptionFacts(image_sha256="bare"),
        CaptionFacts(image_sha256="x", habitat="coral_reef"),
    ]
    report = coverage_report(facts, min_facts=3)
    assert report["rows"] == 3
    assert report["at_or_above"] == 1  # only FULL has >= 3 facts
    assert report["pct"] == round(100 / 3, 2)
