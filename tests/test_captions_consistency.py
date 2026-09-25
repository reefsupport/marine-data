"""WP-13: the automated consistency-flagger fixtures and the Wilson-CI helper."""

import pytest

from marinedata.captions.consistency import check_caption, wilson_ci


def test_agreeing_caption_raises_no_flag():
    flags = check_caption(
        "A healthy hard coral colony on the reef flat.",
        bleaching_status="HEALTHY",
        benthic_dominant="HC",
    )
    assert flags == []


def test_silent_caption_raises_no_flag():
    # The caption never names a bleaching or benthic state -> nothing to contradict.
    flags = check_caption(
        "A wide-angle photo of the seafloor.",
        bleaching_status="BLEACHED",
        benthic_dominant="HC",
    )
    assert flags == []


def test_contradicting_bleaching_state_is_flagged():
    flags = check_caption(
        "A healthy-looking coral colony.",
        bleaching_status="BLEACHED",
        benthic_dominant=None,
    )
    assert len(flags) == 1
    assert flags[0].startswith("bleaching:")
    assert "label=BLEACHED" in flags[0]


def test_contradicting_benthic_dominant_is_flagged():
    flags = check_caption(
        "The frame is mostly sand with a little algae.",
        bleaching_status=None,
        benthic_dominant="HC",
    )
    assert len(flags) == 1
    assert flags[0].startswith("benthic:")


def test_empty_caption_raises_no_flag():
    assert check_caption("", bleaching_status="BLEACHED", benthic_dominant="HC") == []


def test_two_axis_contradiction_raises_two_flags():
    flags = check_caption(
        "A healthy patch of sand.",
        bleaching_status="BLEACHED",
        benthic_dominant="HC",
    )
    axes = {f.split(":")[0] for f in flags}
    assert axes == {"bleaching", "benthic"}


def test_wilson_ci_zero_n_returns_zero_zero():
    assert wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_matches_known_value_within_tolerance():
    # 54/62 ~ 87.1% (D-U2's audited precision) -> a textbook Wilson interval.
    lo, hi = wilson_ci(54, 62)
    assert lo == pytest.approx(0.766, abs=0.01)
    assert hi == pytest.approx(0.933, abs=0.01)
    assert lo < 54 / 62 < hi


def test_wilson_ci_is_narrower_with_more_data_same_rate():
    lo_small, hi_small = wilson_ci(9, 10)
    lo_big, hi_big = wilson_ci(900, 1000)
    assert (hi_big - lo_big) < (hi_small - lo_small)
