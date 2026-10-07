"""WP-U14a: metadata normalisation framework (default + per-row normalisers + CLI)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from marinedata import cli
from marinedata.metadata_norm import OUTPUT_COLUMNS, NormContext, normalise
from marinedata.metadata_norm import stage as stage_mod
from marinedata.metadata_norm.fill import fill_rates
from marinedata.sample_schema import FIELD_NAMES

SHA = "a" * 64


def _ctx(**kw) -> NormContext:
    base = {
        "registry_licence": "CC-BY-4.0",
        "attribution": "Authors",
        "citation": "Cite 2026",
        "homepage": "https://example.org/ds",
        "source_class": "open",
        "checksums": {"images/train/a.png": SHA, "images/val/b.jpg": "b" * 64},
        "ingest": {"fetch_date": "2026-10-02", "version": "v9"},
    }
    return NormContext(**{**base, **kw})


def test_output_has_all_sample_columns_plus_class_and_provenance():
    assert (*FIELD_NAMES, "licence_class", "provenance") == OUTPUT_COLUMNS
    assert len(FIELD_NAMES) == 37
    t = normalise("src", "v1", [{"stem": "a", "partition": "train"}], _ctx())
    assert tuple(t.column_names) == OUTPUT_COLUMNS


def test_default_fills_registry_checksum_and_ingest_fields():
    staged = [
        {"stem": "a", "partition": "train", "upstream_path": "d/a.png", "upstream_split": "train",
         "width": 10, "height": 20, "split_group": "g1"},
    ]  # fmt: skip
    row = normalise("src", "v1", staged, _ctx()).to_pylist()[0]
    assert row["license"] == "CC-BY-4.0"
    assert row["attribution"] == "Authors | Cite 2026 | https://example.org/ds"
    assert row["image_sha256"] == SHA and row["image_path"] == "images/train/a.png"
    assert row["image_format"] == "png" and row["source_version"] == "v1"
    assert row["fetch_date"] == dt.date(2026, 10, 2)
    assert (row["width"], row["height"], row["split_group"]) == (10, 20, "g1")
    assert row["upstream_id"] == "d/a.png" and row["sample_id"] == "src/a"
    assert row["lat"] is None and row["depth_m"] is None and row["licence_class"] == "open"
    prov = dict(row["provenance"])
    assert prov["license"] == "registry" and prov["image_sha256"] == "CHECKSUMS.sha256"
    assert prov["fetch_date"] == "INGEST.json" and prov["width"] == "staged"


def test_default_synthesises_rows_from_checksums_and_keeps_staged_values():
    t = normalise("src", "v1", None, _ctx())
    assert t.num_rows == 2 and sorted(t["stem"].to_pylist()) == ["a", "b"]
    kept = normalise(
        "src", "v1", [{"stem": "a", "partition": "train", "license": "CC-BY-NC-4.0"}], _ctx()
    )
    row = kept.to_pylist()[0]
    assert row["license"] == "CC-BY-NC-4.0" and row["licence_class"] == "restricted-nc"


def test_fill_rates_counts_non_null_per_column():
    t = normalise("src", "v1", None, _ctx())
    fr = fill_rates(t)
    assert fr["license"] == 100.0 and fr["lat"] == 0.0 and set(fr) == set(OUTPUT_COLUMNS)


def test_inat_marine_maps_photo_licence_and_observer():
    manifest = [
        {"photo_id": "11", "license": "CC-BY-NC", "observer_id": "7", "extension": "jpeg",
         "width": "2048.0", "height": "1396.0"},
        {"photo_id": "12", "license": "", "observer_id": "8", "extension": "jpg"},
    ]  # fmt: skip
    ctx = _ctx(
        per_row=True, source_class="restricted-nd", registry_licence="per photo", checksums={}
    )
    rows = normalise("inat-marine", "v1", manifest, ctx).to_pylist()
    assert rows[0]["license"] == "CC-BY-NC" and rows[0]["licence_class"] == "restricted-nc"
    assert "iNaturalist observer 7" in rows[0]["attribution"]
    assert rows[0]["upstream_url"].endswith("/photos/11") and rows[0]["width"] == 2048
    assert dict(rows[0]["provenance"])["license"] == "inat-manifest"
    assert rows[1]["licence_class"] == "unknown"  # per-row source, no per-photo licence


def test_fathomnet_uses_box_licences_and_never_copies_the_email():
    rec = {"uuid": "u1", "contributorsEmail": "a@mbari.org", "width": 5, "height": 6,
           "boundingBoxes": [
               {"annotationLicense": "CC-BY-4.0"},
               {"annotationLicense": "CC-BY-NC-4.0"},
           ]}  # fmt: skip
    ctx = _ctx(per_row=True, source_class="restricted-nd", labels={"u1": rec}, checksums={})
    row = normalise("fathomnet", "v1", None, ctx).to_pylist()[0]
    assert row["license"] == "CC-BY-4.0; CC-BY-NC-4.0" and row["licence_class"] == "restricted-nc"
    assert "mbari.org" in row["attribution"] and "a@mbari.org" not in row["attribution"]
    assert (row["upstream_id"], row["width"]) == ("u1", 5)


def test_planktonzilla_split_from_stem_and_licence_fails_closed():
    staged = [{"stem": "data_train-00000-of-00189_parquet_1000", "partition": "default"}]
    ctx = _ctx(per_row=True, source_class="open", checksums={})
    row = normalise("planktonzilla", "v1", staged, ctx).to_pylist()[0]
    assert row["upstream_split"] == "train" and row["upstream_path"].endswith(".parquet#1000")
    assert row["licence_class"] == "unknown"


def test_cli_writes_a_local_parquet_and_never_needs_the_bucket(tmp_path, monkeypatch, capsys):
    ctx = _ctx()

    def fake_load(source_id, version, limit, **_):
        return "v1", [{"stem": "a", "partition": "train"}][:limit], ctx

    monkeypatch.setattr(stage_mod, "load_inputs", fake_load)
    assert cli.main(["metadata", "normalise", "--source", "src", "--out", str(tmp_path)]) == 0
    out = Path(tmp_path) / "src" / "v1" / "metadata.normalised.parquet"
    assert tuple(pq.read_table(out).column_names) == OUTPUT_COLUMNS
    assert "1 rows" in capsys.readouterr().out


def test_cli_rejects_a_bad_limit(tmp_path):
    argv = ["metadata", "normalise", "--source", "src", "--out", str(tmp_path), "--limit", "0"]
    assert cli.main(argv) == 2


def test_cli_requires_source_and_out():
    with pytest.raises(SystemExit):
        cli.main(["metadata", "normalise"])
