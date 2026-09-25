"""Tests for :mod:`marinedata.task_layers.rollup` (D-Y, WP-8).

Every case below is hand-derived from the D-Y bullet list, not from the function's own
output: the 50% dominant cutoff (inclusive), the 10% present cutoff (inclusive), the
unknown-point/pixel exclusion from the denominator, the > 50%-unknown image exclusion,
and the < 10 known-points-or-pixels "cover only" case. The last case is exercised once
with point-sized counts and once with pixel-sized counts, since D-Y says masks get "the
same three fields" as points from the same rule.
"""

from __future__ import annotations

import pytest

from marinedata.task_layers.rollup import rollup_counts


def test_dominant_at_exactly_50_percent_counts_as_dominant() -> None:
    # 5/10 = 50% exactly, meets ">= 50%" (D-Y says "a class with >= 50%").
    result = rollup_counts({"HC": 5, "ALGAE": 3, "SC": 2})
    assert result.dominant == "HC"
    assert result.cover["HC"] == pytest.approx(0.5)


def test_no_class_at_50_percent_is_mixed() -> None:
    result = rollup_counts({"HC": 4, "ALGAE": 4, "SC": 2})
    assert result.dominant == "mixed"
    assert result.cover_only is False


def test_present_set_uses_10_percent_cutoff_inclusive() -> None:
    # Total known = 100: HC 50%, ALGAE 10% (included), SC 9% (excluded), ABIOTIC 31%.
    result = rollup_counts({"HC": 50, "ALGAE": 10, "SC": 9, "ABIOTIC": 31})
    assert result.present == frozenset({"HC", "ALGAE", "ABIOTIC"})
    assert "SC" not in result.present


def test_unknown_points_excluded_from_denominator_but_not_from_image_exclusion() -> None:
    # 5 known / 10 total = 50% unknown: NOT > 50%, so the image is not excluded, but the
    # 5 unknown points must not count toward the cover fractions of the known classes.
    result = rollup_counts({"HC": 5, "unknown": 5})
    assert result.excluded is False
    assert result.n_known == 5
    assert result.cover == {"HC": pytest.approx(1.0)}


def test_image_excluded_when_more_than_50_percent_unknown() -> None:
    result = rollup_counts({"HC": 2, "unknown": 8})
    assert result.excluded is True
    assert result.cover == {}
    assert result.dominant is None
    assert result.present == frozenset()


def test_fewer_than_10_points_gets_cover_only() -> None:
    # D-Y: "Images with < 10 points get cover only."
    result = rollup_counts({"HC": 3, "ALGAE": 2})
    assert result.n_known == 5
    assert result.cover_only is True
    assert result.dominant is None
    assert result.present == frozenset()
    assert result.cover == {"HC": pytest.approx(0.6), "ALGAE": pytest.approx(0.4)}


def test_exactly_10_points_is_not_cover_only() -> None:
    result = rollup_counts({"HC": 6, "ALGAE": 4})
    assert result.n_known == 10
    assert result.cover_only is False
    assert result.dominant == "HC"


def test_mask_pixel_counts_get_the_same_three_fields_as_points() -> None:
    # D-Y: masks -> image uses "the same three fields, from pixel fractions". Pixel
    # counts are orders of magnitude larger than point counts but the rule is identical.
    result = rollup_counts({"HC": 600_000, "ALGAE": 350_000, "unknown": 50_000})
    assert result.excluded is False
    assert result.dominant == "HC"
    assert result.cover["HC"] == pytest.approx(0.6315789, rel=1e-6)
    assert result.present == frozenset({"HC", "ALGAE"})


def test_negative_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        rollup_counts({"HC": -1})


def test_all_unknown_image_is_excluded() -> None:
    result = rollup_counts({"unknown": 20})
    assert result.excluded is True
    assert result.n_known == 0
