"""WP-2 metadata config: field derivation, coverage report, licence filter, no-null-licence
CI gate. All fixtures are in-memory/tmp_path — no dependency on the real corpus."""

from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pytest

pytest.importorskip("pyarrow")

from marinedata import metadata_release as mr


def _source(**kw):
    base = dict(
        id="src-a",
        name="Source A",
        version="v1",
        licence=SimpleNamespace(id="CC-BY-4.0"),
        tags=(),
        verification=SimpleNamespace(verified_on=dt.date(2026, 9, 20)),
        checksums=SimpleNamespace(root_digest="a" * 64),
        images_from=(),
        location_sensitive=False,
        habitat=None,
    )
    return SimpleNamespace(**{**base, **kw})


class _Registry:
    def __init__(self, sources: dict[str, object]):
        self._sources = sources

    def source(self, source_id: str):
        return self._sources[source_id]


def test_attribution_is_never_empty():
    assert mr.attribution_for(_source()) == "Source A (CC-BY-4.0)"


def test_is_location_sensitive_reads_the_tag_convention():
    assert not mr.is_location_sensitive(_source())
    assert mr.is_location_sensitive(_source(tags=(mr.LOCATION_SENSITIVE_TAG,)))


def test_is_location_sensitive_first_class_field_wins_over_tag_absence():
    assert mr.is_location_sensitive(_source(location_sensitive=True))


def test_is_location_sensitive_cr_en_taxon_gate():
    """WP-2b: a sample carrying a CR/EN-labelled taxon is sensitive even when its
    source is not flagged at all."""
    source = _source()
    cr_en = frozenset({"Epinephelus striatus"})  # Nassau grouper, IUCN CR
    assert not mr.is_location_sensitive(
        source, sample_labels=("Acropora cervicornis",), cr_en_labels=cr_en
    )
    assert mr.is_location_sensitive(
        source, sample_labels=("Epinephelus striatus",), cr_en_labels=cr_en
    )


def test_generalize_rounds_and_flags_only_when_sensitive_and_positioned():
    assert mr._generalize(1.234, 5.678, sensitive=False) == (1.234, 5.678, False)
    assert mr._generalize(None, None, sensitive=True) == (None, None, False)
    assert mr._generalize(1.234, 5.678, sensitive=True) == (1.2, 5.7, True)


def test_lineage_root_digest_single_parent_only():
    parent = _source(id="parent", checksums=SimpleNamespace(root_digest="b" * 64))
    registry = _Registry({"parent": parent})
    derived = _source(images_from=("parent",))
    assert mr.lineage_root_digest_for(derived, registry) == "b" * 64

    two_parents = _source(images_from=("parent", "other"))
    assert mr.lineage_root_digest_for(two_parents, registry) is None

    first_hop = _source(images_from=())
    assert mr.lineage_root_digest_for(first_hop, registry) is None


def _rows(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    staged_dir = tmp_path / "src-a" / "v1"
    staged_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "stem": ["stem1"],
                "partition": ["default"],
                "upstream_path": ["orig/stem1.jpg"],
            }
        ),
        staged_dir / "metadata.parquet",
    )
    registry = _Registry({"src-a": _source()})
    from pathlib import Path

    refs = [mr.ImageRef("s" * 64, "src-a", Path("/x/default/stem1.jpg"), "train")]
    quality = {"s" * 64: {"min_side": 512, "q_blur": 10.0, "flags": ["blur_low"]}}
    return mr.build_rows(refs, registry, tmp_path, quality)


def test_build_rows_never_leaves_license_null(tmp_path):
    rows = _rows(tmp_path)
    assert len(rows) == 1
    assert all(r["license"] is not None for r in rows)
    assert rows[0]["source_version"] == "v1"
    assert rows[0]["upstream_id"] == "orig/stem1.jpg"
    assert rows[0]["quality_flags"] == "blur_low"
    assert rows[0]["upstream_url"] is None  # D-K: no per-item URL resolves
    assert rows[0]["location_generalized"] is False


def test_coverage_report_is_0_or_100_for_a_single_uniform_source(tmp_path):
    cov = mr.coverage(_rows(tmp_path))
    assert cov["n_rows"] == 1
    assert cov["overall"]["license"] == 100.0
    assert cov["overall"]["lat"] == 0.0
    assert cov["by_source"]["src-a"]["license"] == 100.0


def test_filter_by_license_keeps_only_allowed(tmp_path):
    import pyarrow as pa

    table = pa.table({"license": ["CC-BY-4.0", "CC-BY-NC-4.0"], "n": [1, 2]})
    filtered = mr.filter_by_license(table, ["CC-BY-4.0"])
    assert filtered.column("n").to_pylist() == [1]


def test_no_row_may_ship_with_a_null_license(tmp_path):
    """The CI gate item 5 asks for: any row built by WP-2 has a non-null `license`."""
    rows = _rows(tmp_path)
    offenders = [r["image_sha256"] for r in rows if not r.get("license")]
    assert offenders == []


def test_render_coverage_markdown_documents_every_partial_field(tmp_path):
    cov = mr.coverage(_rows(tmp_path))
    doc = mr.render_coverage_markdown(cov)
    assert "# Metadata coverage — v1" in doc
    assert "| `license` | 100.0% |" in doc
    assert "| `lat` | 0.0% |" in doc
    # every WP-2-owned field below 100% coverage must carry a documented null reason
    for field, pct in cov["overall"].items():
        if pct < 100.0 and field in mr.REQUIRED_NULL_REASONS:
            assert f"`{field}`:" in doc


def test_project_v2_coverage_reads_generically_and_handles_missing_columns(tmp_path):
    import pandas as pd

    full = tmp_path / "full.parquet"
    pd.DataFrame(
        {
            "lat": [1.0, None],
            "lon": [2.0, None],
            "depth_m": [3.0, 4.0],
            "capture_datetime": [None, None],
            "platform": [None, None],
            "camera": [None, None],
        }
    ).to_parquet(full)
    bare = tmp_path / "bare.parquet"
    pd.DataFrame({"stem": ["a", "b", "c"]}).to_parquet(bare)

    proj = mr.project_v2_coverage({"has-some": full, "has-none": bare})
    assert proj["by_source"]["has-some"]["lat"] == 50.0
    assert proj["by_source"]["has-some"]["depth_m"] == 100.0
    assert proj["by_source"]["has-none"]["lat"] == 0.0
    assert proj["by_source"]["has-none"]["camera"] == 0.0
    assert proj["n_rows"] == 5
    md = mr.render_v2_projection_markdown(proj)
    assert "has-some" in md and "has-none" in md


def test_build_rows_joins_geo_backfill_and_generalizes_sensitive(tmp_path):
    from pathlib import Path

    from marinedata import geo_backfill as gb

    bf = tmp_path / "bf"
    rec = gb.GeoRecord(
        "default/stem1",
        12.4237,
        -81.4825,
        "source_centroid",
        "fixture",
        capture_datetime="2022-09-26",
    )
    gb.write_backfill("src-a", [rec], bf)
    registry = _Registry({"src-a": _source()})
    refs = [mr.ImageRef("s" * 64, "src-a", Path("/x/default/stem1.jpg"), "train")]
    (row,) = mr.build_rows(refs, registry, tmp_path, {}, backfill_root=bf)
    assert (row["lat"], row["lon"], row["geo_precision"]) == (12.4237, -81.4825, "source_centroid")
    assert row["capture_datetime"] == "2022-09-26" and row["geo_source"] == "fixture"
    (gen,) = mr.build_rows(
        refs,
        registry,
        tmp_path,
        {},
        backfill_root=bf,
        sample_labels={"s" * 64: ["Acropora palmata"]},
        cr_en_labels=frozenset({"Acropora palmata"}),
    )
    assert (gen["lat"], gen["lon"], gen["location_generalized"]) == (12.4, -81.5, True)
    (none,) = mr.build_rows(refs, registry, tmp_path, {}, backfill_root=tmp_path / "empty")
    assert none["geo_precision"] == "none" and none["lat"] is None


def _group_fixture(tmp_path, staged_group):
    import pyarrow as pa
    import pyarrow.parquet as pq

    staged_dir = tmp_path / "src-a" / "v1"
    staged_dir.mkdir(parents=True)
    cols = {"stem": ["stem1"], "partition": ["default"], "upstream_path": ["o/stem1.jpg"]}
    if staged_group is not None:
        cols["split_group"] = [staged_group]
    pq.write_table(pa.table(cols), staged_dir / "metadata.parquet")
    return _Registry({"src-a": _source()})


def _ref(group):
    from pathlib import Path

    return [mr.ImageRef("s" * 64, "src-a", Path("/x/default/stem1.jpg"), "train", group)]


def test_split_group_falls_back_to_the_release_group_and_staged_wins(tmp_path):
    """WP-R9: suim/mermaid/atlantis had a NULL split_group in the metadata config because only
    the staged metadata.parquet was read; the release's own group is the fallback."""
    registry = _group_fixture(tmp_path, None)
    (row,) = mr.build_rows(_ref("site-7"), registry, tmp_path, {})
    assert row["split_group"] == "site-7"
    mr.require_split_groups([row])
    registry = _group_fixture(tmp_path / "b", "staged-1")
    (row,) = mr.build_rows(_ref("site-7"), registry, tmp_path / "b", {})
    assert row["split_group"] == "staged-1"


def test_a_null_split_group_cannot_reach_the_export_silently(tmp_path):
    registry = _group_fixture(tmp_path, None)
    (row,) = mr.build_rows(_ref(None), registry, tmp_path, {})
    assert row["split_group"] is None
    with pytest.raises(mr.MetadataBuildError, match=r"NULL split_group.*src-a \(1 rows\)"):
        mr.require_split_groups([row])
    # the export entry point enforces it too
    from pathlib import Path

    from marinedata.hf_export import ExportRow

    layout = {
        "images": (
            None,
            {
                "train": [
                    ExportRow(
                        values={"image_sha256": "s" * 64, "source_id": "src-a"},
                        file=Path("/x/default/stem1.jpg"),
                    )
                ]
            },
        )
    }
    with pytest.raises(mr.MetadataBuildError, match="src-a"):
        mr.add_metadata_config(layout, registry, {"src-a": tmp_path / "src-a" / "v1"})


def test_metadata_rows_carry_privacy_flag_and_face_score(tmp_path):
    """WP-R9: the 0.60-0.85 band is kept and flagged; everything else has neither column set."""
    registry = _group_fixture(tmp_path, "g1")
    (flagged,) = mr.build_rows(_ref("g1"), registry, tmp_path, {}, privacy_flags={"s" * 64: 0.7})
    assert (flagged["privacy_flag"], flagged["face_score"]) == ("possible_face", 0.7)
    (plain,) = mr.build_rows(_ref("g1"), registry, tmp_path, {})
    assert (plain["privacy_flag"], plain["face_score"]) == (None, None)
