"""Pipeline-level ingest tests (I2, D1 §7). No network: a synthetic zip built in
``tmp_path`` mimics ``SUIM/{train_val,TEST}/{images,masks}``, and ``fetch.get_bytes``
is monkeypatched to return its bytes instead of hitting the network.

``tests/test_checksums.py`` already proves the manifest-layer properties this reuses
(no-op rewrite, mutation raises, ``shasum -c`` passes, recorded-not-read — tests 13, 14,
16 in D1 §7 live in ``tests/test_ingest_primitives.py`` instead) — not duplicated here.
"""

from __future__ import annotations

import io
import re
import subprocess
import zipfile
from datetime import date
from pathlib import Path

import pytest

from marinedata import fetch
from marinedata.checksums import ChecksumError
from marinedata.enums import (
    AccessMethod,
    AnnotationKind,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.gate import LicenceViolation
from marinedata.ingest import ArchivePlan, IngestError, _stage_with_plan, stage_source
from marinedata.models import (
    Access,
    Annotation,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    Verification,
)
from marinedata.sample import Sample
from marinedata.scan import group_key
from marinedata.tables import PointRow

# --------------------------------------------------------------------------- fixtures


def _jpeg_bytes(
    size: tuple[int, int] = (4, 3), color: tuple[int, int, int] = (10, 20, 30)
) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def _bmp_bytes(pixels: list[int], size: tuple[int, int] = (4, 3)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    im = Image.new("L", size)
    im.putdata(pixels)
    im.save(buf, format="BMP")
    return buf.getvalue()


def _build_zip(specs: dict[str, dict]) -> bytes:
    """Build an in-memory zip mimicking ``SUIM/{split}/{images,masks}/<stem>.<ext>``.

    ``specs`` maps a raw upstream stem to ``{"split": "train_val"|"TEST", "mask":
    bool, "pixels": [...]}``; unset keys default to ``split="train_val"``,
    ``mask=True``, a small deterministic pixel list.
    """
    members: dict[str, bytes] = {}
    for stem, spec in specs.items():
        split = spec.get("split", "train_val")
        members[f"SUIM/{split}/images/{stem}.jpg"] = _jpeg_bytes()
        if spec.get("mask", True):
            pixels = spec.get("pixels", [0, 1, 2, 3] * 3)
            members[f"SUIM/{split}/masks/{stem}.bmp"] = _bmp_bytes(pixels)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _suim_plan(
    *, expected_images: int, expected_masks: int, classes: int = 4, ignore_index: int | None = None
) -> ArchivePlan:
    return ArchivePlan(
        image_pattern=re.compile(r"^SUIM/(?P<split>train_val|TEST)/images/(?P<stem>[^/]+)\.jpg$"),
        mask_pattern=re.compile(r"^SUIM/(?P<split>train_val|TEST)/masks/(?P<stem>[^/]+)\.bmp$"),
        expected_images=expected_images,
        expected_masks=expected_masks,
        partition="default",
        version="v1",
        classes=classes,
        license_text="MIT License (fixture)\n",
        ignore_index=ignore_index,
    )


def _source(
    sample_url: str = "https://example.invalid/SUIM.zip",
    *,
    source_id: str = "fixturesrc",
    tier: Tier = Tier.PERMISSIVE,
    classes: int = 4,
) -> Source:
    return Source(
        id=source_id,
        name="Fixture",
        description="Synthetic ingest fixture source.",
        version="v1",
        licence=Licence(id="MIT", name="MIT License", tier=tier),
        notes="fixture: denied on purpose for test coverage" if tier == Tier.PROHIBITED else None,
        verification=Verification(
            verified_on=date(2026, 8, 17), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.HTTP,
            uri="https://example.invalid",
            params={"sample_url": sample_url},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(
            layout="image-mask-pairs", schema_id="dataset-native", crosswalk_id="fixture-8class"
        ),
        annotations=(
            Annotation(kind=AnnotationKind.DENSE_MASK, classes=classes, supervises=("taxon",)),
        ),
    )


def _profile() -> Profile:
    return Profile(
        id="research",
        description="fixture profile",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )


def _stage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, specs: dict[str, dict], **plan_kwargs):
    images = sum(1 for s in specs.values())
    masks = sum(1 for s in specs.values() if s.get("mask", True))
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(
        expected_images=plan_kwargs.pop("expected_images", images),
        expected_masks=plan_kwargs.pop("expected_masks", masks),
        **plan_kwargs,
    )
    source = _source()
    result = _stage_with_plan(
        source, plan, cache_root=tmp_path / "cache", out_root=tmp_path / "out", profile=_profile()
    )
    return result, source, plan


# ------------------------------------------------------------------------------ tests


def test_reingesting_identical_bytes_is_a_no_op(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    specs = {"a": {}, "b": {}, "c": {}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=3, expected_masks=3)
    source = _source()
    cache_root, out_root, profile = tmp_path / "cache", tmp_path / "out", _profile()

    v1 = _stage_with_plan(source, plan, cache_root=cache_root, out_root=out_root, profile=profile)
    v2 = _stage_with_plan(source, plan, cache_root=cache_root, out_root=out_root, profile=profile)

    assert v1.manifest.root_digest == v2.manifest.root_digest
    assert v1.images == v2.images == 3


def test_staged_root_is_out_sources_id_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, source, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})

    assert result.root == tmp_path / "out" / "sources" / source.id / source.version


def test_a_changed_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    jpg = next((result.root / "images" / "default").glob("*.jpg"))
    data = bytearray(jpg.read_bytes())
    data[0] ^= 0xFF
    jpg.write_bytes(bytes(data))

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


def test_an_added_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    (result.root / "images" / "default" / "stray.jpg").write_bytes(b"not a real image")

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


def test_a_removed_file_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result, source, plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    mask = next((result.root / "labels" / "masks" / "default").glob("*.png"))
    mask.unlink()

    with pytest.raises(ChecksumError):
        _stage_with_plan(
            source,
            plan,
            cache_root=tmp_path / "cache",
            out_root=tmp_path / "out",
            profile=_profile(),
        )


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


def test_shasum_c_passes_on_the_staged_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _source_obj, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "CHECKSUMS.sha256"],
        cwd=result.root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_stem_collision_raises_rather_than_suffixing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "img#1" and "img_1" both sanitise (non [A-Za-z0-9._-] -> "_") to "img_1".
    specs = {"img#1": {"mask": False}, "img_1": {"mask": False}}
    zip_bytes = _build_zip(specs)
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=2, expected_masks=0)
    source = _source()
    out_root = tmp_path / "out"

    with pytest.raises(IngestError, match="stem collision"):
        _stage_with_plan(
            source, plan, cache_root=tmp_path / "cache", out_root=out_root, profile=_profile()
        )

    assert not any(out_root.rglob("*_1*"))


def test_no_timestamp_reaches_the_staged_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _source_obj, _plan = _stage(tmp_path, monkeypatch, {"a": {}, "b": {}})
    for path in result.root.rglob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert "fetched_at" not in text
        assert "staged_at" not in text


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


def test_stage_source_without_a_plan_raises(tmp_path: Path) -> None:
    source = _source(source_id="no-such-plan")
    with pytest.raises(IngestError, match="no ingest plan"):
        stage_source(
            source, cache_root=tmp_path / "cache", out_root=tmp_path / "out", profile=_profile()
        )


def test_wrong_member_count_raises_and_stages_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zip_bytes = _build_zip({"a": {}, "b": {}})
    monkeypatch.setattr(fetch, "get_bytes", lambda url, **kw: zip_bytes)
    plan = _suim_plan(expected_images=5, expected_masks=5)  # zip only has 2 + 2
    source = _source()
    out_root = tmp_path / "out"

    with pytest.raises(IngestError, match="expected 5"):
        _stage_with_plan(
            source, plan, cache_root=tmp_path / "cache", out_root=out_root, profile=_profile()
        )

    assert not (out_root / "sources" / source.id).exists()


# ------------------------------------------------------------------------- D2c: manifest truth


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
