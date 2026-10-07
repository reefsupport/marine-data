"""P3b acceptance: charter D-P(1)'s data-driven tropical-province OOD holdout —
closest-to-5% selection, the >= 500 guard, the Tropical Atlantic exclusion, the
tie-break, the inert case, and wiring the resolved value back into `assign_ood`
(design §3.2, as amended by D-P after it was written)."""

from __future__ import annotations

from pathlib import Path

from marinedata.splitv2.dp_province import (
    EXCLUDED_REALM,
    DPChoice,
    labelled_total,
    province_labelled_counts,
    resolve_tropical_province_rule,
    select_tropical_province,
)
from marinedata.splitv2.holdouts import Sample, assign_ood
from marinedata.splitv2.rules import load_rules

RULES_PATH = Path(__file__).resolve().parents[1] / "registry" / "splits" / "v2.yaml"
TROPICAL_REALMS = (
    "Central Indo-Pacific",
    "Western Indo-Pacific",
    "Eastern Indo-Pacific",
    "Tropical Eastern Pacific",
)


def test_v2_yaml_declares_seven_holdouts_and_the_tropical_realm_list():
    config = load_rules(RULES_PATH)
    assert len(config.holdouts) == 7
    assert set(TROPICAL_REALMS) <= set(config.tropical_realms)
    rule = next(h for h in config.holdouts if h.name == "ood-geo-tropical-province")
    assert rule.data_driven is True
    assert rule.value is None  # unresolved until `splits check` picks a province


def test_selects_the_province_closest_to_five_percent_of_the_labelled_pool():
    # total labelled pool 21,000; target 5% = 1,050.
    counts = {
        "Coral Triangle Core": ("Central Indo-Pacific", 950),  # |950-1050| = 100
        "Solomon Sea": ("Western Indo-Pacific", 1500),  # |1500-1050| = 450
    }
    choice = select_tropical_province(
        counts,
        tropical_realms=TROPICAL_REALMS,
        labelled_total=18_550 + 950 + 1500,
        min_images=500,
    )
    assert choice.province == "Coral Triangle Core"
    assert choice.n_images == 950


def test_below_500_images_never_qualifies_even_when_closest():
    # target = 100; "Tiny Province" (90) is closer but under the floor.
    counts = {
        "Tiny Province": ("Central Indo-Pacific", 90),
        "Big Enough Province": ("Western Indo-Pacific", 500),
    }
    choice = select_tropical_province(
        counts,
        tropical_realms=TROPICAL_REALMS,
        labelled_total=2_000,
        min_images=500,
    )
    assert choice.province == "Big Enough Province"


def test_tropical_atlantic_is_never_chosen_even_when_it_is_closest():
    # target = 1000; the Tropical Atlantic candidate sits exactly on target, but is
    # excluded by construction (D-P(1)), even though the caller's realm allow-list
    # (mis-configured, on purpose, for this test) still lists it as tropical.
    counts = {
        "Our Own Reefs": (EXCLUDED_REALM, 1000),
        "Far East Reef": ("Eastern Indo-Pacific", 600),
    }
    choice = select_tropical_province(
        counts,
        tropical_realms=(*TROPICAL_REALMS, EXCLUDED_REALM),
        labelled_total=20_000,
        min_images=500,
    )
    assert choice.province == "Far East Reef"
    assert choice.province != "Our Own Reefs"


def test_ties_break_by_province_name_ascending():
    # Both candidates land exactly 50 away from the target of 1,000.
    counts = {
        "Zanzibar Channel": ("Eastern Indo-Pacific", 950),
        "Andaman Coast": ("Central Indo-Pacific", 950),
    }
    choice = select_tropical_province(
        counts,
        tropical_realms=TROPICAL_REALMS,
        labelled_total=20_000,
        min_images=500,
    )
    assert choice.province == "Andaman Coast"


def test_inert_case_prints_the_exact_warning_line():
    choice = select_tropical_province(
        {},
        tropical_realms=TROPICAL_REALMS,
        labelled_total=10_000,
        min_images=500,
    )
    assert choice.province is None
    line = choice.describe()
    assert line.startswith("D-P holdout: none qualifies (")
    assert line.endswith(")")


def test_chosen_case_describe_reports_province_count_and_percent():
    choice = DPChoice(province="Fiji Islands", n_images=1000, labelled_total=20_000)
    assert choice.describe() == "D-P holdout: Fiji Islands (1000, 5.00%)"


def _sample(sha, gid, source, realm=None, province=None):
    return Sample(
        sha256=sha,
        split_group_id=gid,
        source_id=source,
        meow_realm=realm,
        meow_province=province,
    )


def test_province_labelled_counts_counts_tagged_samples_and_excludes_never_eval():
    """Real per-sample `meow_realm`/`meow_province` (the common case; per-source
    fallback filling itself is already covered by
    `test_geo_fallback_fills_missing_realm_before_matching`)."""
    config = load_rules(RULES_PATH)
    samples = [
        _sample(f"soc-{i}", f"sg-soc-{i}", "mlc-moorea", "Eastern Indo-Pacific", "Society Islands")
        for i in range(5)
    ] + [
        # never-eval source: must not count toward the labelled pool at all.
        _sample(
            "pretrain-0",
            "sg-pretrain",
            "some-pretrain-source",
            "Central Indo-Pacific",
            "Some Province",
        ),
    ]
    counts = province_labelled_counts(samples, config, never_eval={"some-pretrain-source"})
    assert counts["Society Islands"] == ("Eastern Indo-Pacific", 5)
    assert "Some Province" not in counts
    assert labelled_total(samples, never_eval={"some-pretrain-source"}) == 5


def test_resolved_rule_wires_into_assign_ood_and_holds_out_the_chosen_province():
    """End to end: once D-P picks a province, patching the yaml rule's value makes
    `assign_ood` hold out exactly that province's groups, same as any authored rule."""
    config = load_rules(RULES_PATH)
    # 5 groups x 100 = 500 images, 5 groups: exactly the size-guard floor.
    samples = [
        Sample(
            sha256=f"soc-{g}-{i}",
            split_group_id=f"sg-soc-{g}",
            source_id="mlc-moorea",
            meow_realm="Eastern Indo-Pacific",
            meow_province="Society Islands",
        )
        for g in range(5)
        for i in range(100)
    ]
    choice = select_tropical_province(
        province_labelled_counts(samples, config, never_eval=set()),
        tropical_realms=config.tropical_realms,
        labelled_total=labelled_total(samples, never_eval=set()),
        min_images=config.size_guards["min_images"],
    )
    assert choice.province == "Society Islands"

    resolved = resolve_tropical_province_rule(config, choice)
    result = assign_ood(samples, resolved)

    for g in range(5):
        assert result.group_split[f"sg-soc-{g}"] == "ood-geo-tropical-province"


def test_inert_choice_leaves_the_rule_permanently_unmatched():
    config = load_rules(RULES_PATH)
    inert = DPChoice(province=None, n_images=0, labelled_total=100, reason="test")
    resolved = resolve_tropical_province_rule(config, inert)
    samples = [
        Sample(
            sha256=f"soc-{i}",
            split_group_id=f"sg-soc-{i}",
            source_id="mlc-moorea",
            meow_realm="Eastern Indo-Pacific",
            meow_province="Society Islands",
        )
        for i in range(600)
    ]
    result = assign_ood(samples, resolved)
    assert not any(v == "ood-geo-tropical-province" for v in result.group_split.values())
