"""``SplitGroupRule`` — the per-source ``split_group`` derivation (WS-D step 3).

``group_key(sample, "group")`` (see ``tests/test_scan.py``) already trusts a
``split_group`` value sitting in ``Sample.meta``; these tests cover the other end —
deriving that value from a source's registry rule, and confirming the value ends up
unchanged in ``metadata.parquet``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_source

from marinedata.models import SplitGroupRule
from marinedata.tables import StagedImage, write_metadata_table


def test_default_rule_uses_source_id_and_partition_fallback() -> None:
    """No ``pattern`` set: the fallback template is ``<source_id>/<partition>`` — the
    same grouping ``group_key(by="site")`` already used before this rule existed."""
    rule = SplitGroupRule()
    resolved = rule.resolve(
        source_id="my-source", stem="img1", upstream_path="a/img1.jpg", partition="train"
    )
    assert resolved == "my-source/train"


def test_pattern_over_stem_extracts_group() -> None:
    rule = SplitGroupRule(pattern=r"^(.{5})", match_field="stem", template="seaview/{group}")
    assert (
        rule.resolve(source_id="seaview", stem="12345_001", upstream_path="x", partition="train")
        == "seaview/12345"
    )


def test_pattern_over_upstream_path_extracts_group() -> None:
    rule = SplitGroupRule(
        pattern=r"reef_support/([^/]+)/",
        match_field="upstream_path",
        template="rs-colombia/{group}",
    )
    assert (
        rule.resolve(
            source_id="reef-support-benthic-own",
            stem="img1",
            upstream_path="mask_labels/reef_support/site-a/img1.jpg",
            partition="train",
        )
        == "rs-colombia/site-a"
    )


def test_non_matching_pattern_raises() -> None:
    rule = SplitGroupRule(pattern=r"^(nomatch)", match_field="stem")
    with pytest.raises(ValueError, match="did not match"):
        rule.resolve(source_id="s", stem="other", upstream_path="x", partition="train")


def test_pattern_without_capturing_group_rejected() -> None:
    with pytest.raises(ValueError, match="capturing group"):
        SplitGroupRule(pattern=r"^abc")


def test_invalid_regex_rejected() -> None:
    with pytest.raises(ValueError, match="is invalid"):
        SplitGroupRule(pattern=r"(unterminated")


def test_invalid_match_field_rejected() -> None:
    with pytest.raises(ValueError, match="match_field"):
        SplitGroupRule(match_field="not-a-field")


def test_source_split_group_for_delegates_to_its_rule() -> None:
    src = make_source(layout="image-folder", source_id="my-source")
    assert src.split_group_for(stem="img1", upstream_path="a/img1.jpg", partition="val") == (
        "my-source/val"
    )


def test_benthic_own_station_bay_rule() -> None:
    """`reef-support-benthic-own`'s split_group moved from site (4 groups) to
    station/bay (31 groups, WS-D S22 / R3 Q9 / 7l): known stems from each of the
    four Colombian sites must land in the expected `rs-colombia/<site>/<station>`
    group, and the pattern must not raise on any of them (WS-D S22, template uses
    the per-row `partition`, not the regex, to disambiguate the site)."""
    from marinedata.registry import Registry

    source = Registry.load().source("reef-support-benthic-own")
    cases = [
        ("20220912_AnB_CB10_103_", "SEAFLOWER_BOLIVAR", "rs-colombia/SEAFLOWER_BOLIVAR/CB10"),
        ("P9150320", "SEAFLOWER_BOLIVAR", "rs-colombia/SEAFLOWER_BOLIVAR/P9150"),
        ("E0_T1_C10_Corr_30sep22", "SEAFLOWER_COURTOWN", "rs-colombia/SEAFLOWER_COURTOWN/E0"),
        (
            "C10_BC_PM_T1_29nov24_CDaza_corr",
            "UNAL_BLEACHING_TAYRONA",
            "rs-colombia/UNAL_BLEACHING_TAYRONA/BC",
        ),
        ("G0088299", "TETES_PROVIDENCIA", "rs-colombia/TETES_PROVIDENCIA/G"),
    ]
    for stem, partition, expected in cases:
        assert (
            source.split_group_for(stem=stem, upstream_path=f"x/{stem}", partition=partition)
            == expected
        )

    # Group count must match R3/7l: Bolivar 7, Courtown 19, Tayrona 4, Tetes 1 = 31.
    # Best-effort against the real local staged tree; skips where that tree is absent
    # (e.g. a fresh checkout) rather than failing the suite on missing local data.
    staged = (
        Path.home() / "dev/reefsupport/data/_stage/sources/reef-support-benthic-own/2026-08"
        "/metadata.parquet"
    )
    if not staged.exists():
        pytest.skip(f"local staged tree not present at {staged}")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pq.read_table(staged)
    groups_by_partition: dict[str, set[str]] = {}
    for stem, partition in zip(
        table.column("stem").to_pylist(), table.column("partition").to_pylist(), strict=True
    ):
        group = source.split_group_for(stem=stem, upstream_path=f"x/{stem}", partition=partition)
        groups_by_partition.setdefault(partition, set()).add(group)
    counts = {k: len(v) for k, v in groups_by_partition.items()}
    assert counts == {
        "SEAFLOWER_BOLIVAR": 7,
        "SEAFLOWER_COURTOWN": 19,
        "UNAL_BLEACHING_TAYRONA": 4,
        "TETES_PROVIDENCIA": 1,
    }
    assert sum(counts.values()) == 31


def test_bleaching_shares_benthic_own_station_bay_rule() -> None:
    """`reef-support-bleaching`'s 658 images are byte-identical to
    `reef-support-benthic-own`'s Tayrona partition (WS-D S30, R3 Q11), so its
    `split_group` rule must be the exact same station/bay pattern/template as
    `reef-support-benthic-own`'s — otherwise a shared image would land in different
    groups under the two sources, defeating group-disjoint splitting."""
    from marinedata.registry import Registry

    registry = Registry.load()
    bleaching = registry.source("reef-support-bleaching")
    benthic_own = registry.source("reef-support-benthic-own")

    assert bleaching.split_group.pattern == benthic_own.split_group.pattern
    assert bleaching.split_group.match_field == benthic_own.split_group.match_field
    assert bleaching.split_group.template == benthic_own.split_group.template

    stem = "C10_BC_PM_T1_29nov24_CDaza_corr"
    expected = "rs-colombia/UNAL_BLEACHING_TAYRONA/BC"
    assert (
        bleaching.split_group_for(
            stem=stem, upstream_path=f"x/{stem}", partition="UNAL_BLEACHING_TAYRONA"
        )
        == expected
        == benthic_own.split_group_for(
            stem=stem, upstream_path=f"y/{stem}", partition="UNAL_BLEACHING_TAYRONA"
        )
    )


def test_split_group_rule_matches_metadata_column(tmp_path: Path) -> None:
    """The value ``source.split_group_for(...)`` computes for a staged row must be the
    exact value that lands in ``metadata.parquet``'s ``split_group`` column — no
    ingest path is allowed to recompute or normalise it differently on the way in."""
    pq = pytest.importorskip("pyarrow.parquet")

    rule = SplitGroupRule(pattern=r"^(.{5})", match_field="stem", template="seaview/{group}")
    src = make_source(layout="image-folder", source_id="seaview-survey-imagery").model_copy(
        update={"split_group": rule}
    )

    stems = ["12345_001", "12345_002", "67890_001"]
    expected = {
        stem: src.split_group_for(stem=stem, upstream_path=f"x/{stem}.jpg", partition="train")
        for stem in stems
    }

    rows = [
        StagedImage(
            stem=stem,
            partition="train",
            upstream_path=f"x/{stem}.jpg",
            upstream_split=None,
            width=10,
            height=10,
            split_group=expected[stem],
        )
        for stem in stems
    ]
    path = tmp_path / "metadata.parquet"
    write_metadata_table(path, rows)

    table = pq.read_table(path)
    by_stem = dict(
        zip(
            table.column("stem").to_pylist(),
            table.column("split_group").to_pylist(),
            strict=True,
        )
    )
    assert by_stem == expected
