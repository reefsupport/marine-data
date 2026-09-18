"""Pipeline-level ingest tests (I2, D1 §7): content and manifest-truth correctness —
split from ``test_ingest.py`` (D2e, 503 ln) by topic. What lands where (image+points,
no unlabelled path, partition vs upstream split, split-map keys, byte-identical images,
licence gating, orphan annotation rows) and the D2c manifest-truth fixes
(``raster_ignore_value``, ``fetched_uri``). No network: a synthetic zip built in
``tmp_path`` mimics ``SUIM/{train_val,TEST}/{images,masks}``, and ``fetch.get_bytes``
is monkeypatched to return its bytes instead of hitting the network. Shared builders
live in ``tests/_ingest_helpers.py``.

Staging integrity and error paths (re-ingest no-op, checksum-guard failures, stem
collisions, timestamp leakage) live in ``tests/test_ingest_staging.py``.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from _ingest_helpers import _build_zip, _profile, _source, _stage, _suim_plan

from marinedata import fetch
from marinedata.enums import Tier
from marinedata.gate import LicenceViolation
from marinedata.ingest import IngestError, _stage_with_plan
from marinedata.sample import Sample
from marinedata.scan import group_key
from marinedata.tables import PointRow


def test_an_image_with_mask_and_points_appears_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {}, "b": {}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=2, expected_masks=2)
    source = _source()
    points = (
        PointRow(
            stem="a", partition="default", row=1, col=2, label="HC", schema_id="dataset-native"
        ),
    )

    result = _stage_with_plan(
        source,
        plan,
        cache_root=tmp_path / "cache",
        out_root=tmp_path / "out",
        profile=_profile(),
        points=points,
    )

    assert len(list((result.root / "images" / "default").glob("a.jpg"))) == 1
    assert len(list((result.root / "labels" / "masks" / "default").glob("a.png"))) == 1

    import pyarrow.parquet as pq

    table = pq.read_table(result.root / "labels" / "points.parquet")
    assert table.num_rows == 1
    assert table.column("stem").to_pylist() == ["a"]


def test_no_unlabelled_path_is_ever_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {"mask": True}, "b": {"mask": False}, "c": {"mask": False}}
    result, _source, _plan = _stage(tmp_path, monkeypatch, specs)

    # Relative to result.root, not the absolute path — pytest's own tmp_path directory
    # name (derived from this test's name) contains "unlabelled" as a substring.
    relative_paths = [
        p.relative_to(result.root).as_posix() for p in result.root.rglob("*") if p.is_file()
    ]
    assert not any("unlabelled" in rel for rel in relative_paths)
    assert len(list((result.root / "images" / "default").glob("*.jpg"))) == 3

    import json

    counts = json.loads((result.root / "ANNOTATIONS.json").read_text(encoding="utf-8"))
    assert counts["images"] == 3
    assert counts["images_without_annotation"] == 2


def test_partition_is_not_the_upstream_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {"split": "train_val"}, "b": {"split": "TEST"}}
    result, _source, _plan = _stage(tmp_path, monkeypatch, specs)

    for path in result.root.rglob("*"):
        if path.is_file():
            rel = path.relative_to(result.root).as_posix()
            assert "train_val" not in rel
            assert "/TEST/" not in f"/{rel}/"

    import pyarrow.parquet as pq

    table = pq.read_table(result.root / "metadata.parquet")
    assert set(table.column("partition").to_pylist()) == {"default"}
    assert set(table.column("upstream_split").to_pylist()) == {"train_val", "TEST"}


def test_split_map_group_key_matches_the_staged_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})

    import pyarrow.parquet as pq

    table = pq.read_table(result.root / "metadata.parquet")
    assert set(table.column("partition").to_pylist()) == {plan.partition}

    sample = Sample(source_id=source.id, key="x", meta={"partition": plan.partition})
    assert group_key(sample, "site") == f"{source.id}/{plan.partition}"


def test_images_are_byte_identical_to_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {}, "b": {}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=2, expected_masks=2)
    source = _source()
    result = _stage_with_plan(
        source, plan, cache_root=tmp_path / "cache", out_root=tmp_path / "out", profile=_profile()
    )

    import hashlib

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        upstream_sha = hashlib.sha256(archive.read("SUIM/train_val/images/a.jpg")).hexdigest()
    staged_path = result.root / "images" / "default" / "a.jpg"
    staged_sha = hashlib.sha256(staged_path.read_bytes()).hexdigest()
    assert staged_sha == upstream_sha


def test_a_denied_licence_raises_before_any_byte_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = _build_zip({"a": {}})
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=1, expected_masks=1)
    source = _source(tier=Tier.PROHIBITED)
    out_root = tmp_path / "out"

    with pytest.raises(LicenceViolation):
        _stage_with_plan(
            source, plan, cache_root=tmp_path / "cache", out_root=out_root, profile=_profile()
        )

    assert not out_root.exists() or not any(out_root.rglob("*"))


def test_every_annotation_row_stem_resolves_to_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = _build_zip({"a": {}, "b": {}})
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=2, expected_masks=2)
    source = _source()
    out_root = tmp_path / "out"
    points = (
        PointRow(
            stem="ghost", partition="default", row=0, col=0, label="HC", schema_id="dataset-native"
        ),
    )

    with pytest.raises(IngestError, match="ghost"):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=out_root,
            profile=_profile(),
            points=points,
        )


def test_ignore_index_written_from_plan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``ANNOTATIONS.json``'s ``raster_ignore_value`` comes from ``plan.ignore_index``,
    not a hardcoded ``255`` (the defect D2c fixes)."""
    result, _source_obj, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}}, ignore_index=7)

    import json

    counts = json.loads((result.root / "ANNOTATIONS.json").read_text(encoding="utf-8"))
    assert counts["geometries"][0]["raster_ignore_value"] == 7


def test_ignore_index_none_writes_null(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan with no reserved ignore index writes JSON ``null``, not a made-up value."""
    result, _source_obj, _plan = _stage(
        tmp_path, monkeypatch, {"a": {}, "b": {}}, ignore_index=None
    )

    import json

    counts = json.loads((result.root / "ANNOTATIONS.json").read_text(encoding="utf-8"))
    assert counts["geometries"][0]["raster_ignore_value"] is None


def test_source_json_has_fetched_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``SOURCE.json`` names the URL bytes were actually requested from, so a re-host
    fetch is visible in the manifest rather than only in the registry's prose."""
    result, source, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})

    import json

    payload = json.loads((result.root / "SOURCE.json").read_text(encoding="utf-8"))
    assert payload["_ingest"]["fetched_uri"] == source.access.params["sample_url"]


def test_suim_plan_ignore_index_is_none() -> None:
    """The real (non-fixture) suim plan: BW 0-7 are all real classes, 255 never
    occurs — no ignore index."""
    from marinedata.ingest import _PLANS

    assert _PLANS["suim"].ignore_index is None
