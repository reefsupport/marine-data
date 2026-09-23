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
