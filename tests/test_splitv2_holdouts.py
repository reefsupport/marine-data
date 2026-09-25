"""P3 acceptance: any-member-OOD group rule, missing metadata stays ID (design §3.2)."""

from __future__ import annotations

from pathlib import Path

from marinedata.splitv2.holdouts import Sample, assign_ood
from marinedata.splitv2.rules import load_rules

RULES_PATH = Path(__file__).resolve().parents[1] / "registry" / "splits" / "v2.yaml"


def _deepfish_pool(extra: list[Sample]) -> list[Sample]:
    """5 deepfish groups x 100 images = 500 images, 5 groups — exactly the size guard
    floor (§3.2: >= 500 images and >= 5 groups) so the holdout is constructible."""
    pool = []
    for g in range(5):
        for i in range(100):
            pool.append(
                Sample(
                    sha256=f"df{g}-{i}",
                    split_group_id=f"sg-df{g}",
                    source_id="deepfish",
                )
            )
    return pool + extra


def test_any_member_ood_rule_holds_out_the_whole_group():
    """A group with one deepfish member and one non-deepfish member goes OOD as a
    unit — the group is the assignment atom, not the sample (design §3.1/§3.2)."""
    config = load_rules(RULES_PATH)
    mixed_group = [
        Sample(sha256="mix-a", split_group_id="sg-mixed", source_id="deepfish"),
        Sample(sha256="mix-b", split_group_id="sg-mixed", source_id="some-other-source"),
    ]
    samples = _deepfish_pool(mixed_group)

    result = assign_ood(samples, config)

    assert result.group_split["sg-mixed"] == "ood-source-deepfish"
    assert "ood-source-deepfish" in result.group_tags["sg-mixed"]
    # every pure deepfish group is also held out
    for g in range(5):
        assert result.group_split[f"sg-df{g}"] == "ood-source-deepfish"


def test_missing_geography_stays_in_distribution():
    """No meow_realm/meow_province and a source absent from geo_fallback: the sample
    (and its group) never goes OOD — design's "missing metadata never sends a sample
    OOD; it stays ID" (§3.2)."""
    config = load_rules(RULES_PATH)
    samples = _deepfish_pool(
        [
            Sample(
                sha256="unk-a",
                split_group_id="sg-unknown-geo",
                source_id="totally-unregistered-source",
                meow_realm=None,
                meow_province=None,
                depth_m=None,
                platform=None,
                capture_datetime=None,
            )
        ]
    )

    result = assign_ood(samples, config)

    assert "sg-unknown-geo" not in result.group_split
    assert "sg-unknown-geo" not in result.group_tags


def test_geo_fallback_fills_missing_realm_before_matching():
    """deepseagrass has no per-sample meow_realm but a registered fallback of
    'Temperate Australasia' — the fallback must fill the field before the equality
    check, not bypass it (§3.2)."""
    config = load_rules(RULES_PATH)
    # 5 groups x 101 images = 505, >= 500 images and >= 5 groups (size guard floor).
    fallback_pool = []
    for g in range(5):
        for i in range(101):
            fallback_pool.append(
                Sample(
                    sha256=f"dsg-{g}-{i}",
                    split_group_id=f"sg-dsg{g}",
                    source_id="deepseagrass",
                    meow_realm=None,
                )
            )

    result = assign_ood(fallback_pool, config)

    for g in range(5):
        assert result.group_split[f"sg-dsg{g}"] == "ood-geo-temperate-australasia"


def test_precedence_first_match_wins_split_but_all_matches_are_tagged():
    """A group matching both the deepfish rule (#1) and, hypothetically, a later rule
    is assigned the earlier rule's split; every constructible match still appears in
    ood_tags."""
    config = load_rules(RULES_PATH)
    # 5 deepfish groups (constructible on their own); sg-df0's members are also >=
    # 800m deep, so it matches both #1 (deepfish) and #4 (depth). 5 more >= 800m
    # groups from another source make #4 constructible too (600 images, 6 groups).
    deepfish_pool = [
        Sample(
            sha256=f"df{g}-{i}",
            split_group_id=f"sg-df{g}",
            source_id="deepfish",
            depth_m=900 if g == 0 else None,
        )
        for g in range(5)
        for i in range(100)
    ]
    other_deep = [
        Sample(
            sha256=f"od-{g}-{i}",
            split_group_id=f"sg-deep-{g}",
            source_id="other-deep-source",
            depth_m=900,
        )
        for g in range(5)
        for i in range(100)
    ]
    samples = deepfish_pool + other_deep

    result = assign_ood(samples, config)

    assert result.group_split["sg-df0"] == "ood-source-deepfish"
    assert set(result.group_tags["sg-df0"]) >= {"ood-source-deepfish", "ood-depth-deep"}
