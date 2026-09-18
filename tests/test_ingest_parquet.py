"""Pipeline-level tests for the HF-parquet ingest path (D2a). No network: fixture
shards are tiny in-memory parquet files built with pyarrow + PIL and written straight
to ``tmp_path`` — :func:`marinedata.ingest_parquet._stage_with_parquet_plan` never
calls :func:`marinedata.fetch.get_bytes`/``get_stream`` itself (that is
:func:`marinedata.ingest_parquet.fetch_parquet_shards`'s job, exercised by neither this
module nor D2b), so there is nothing to monkeypatch for these tests to stay offline.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata import gate
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
from marinedata.ingest import IngestError
from marinedata.ingest_parquet import ParquetPlan, _finish_staging, _stage_with_parquet_plan
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

# --------------------------------------------------------------------------- fixtures

_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
_SCHEMA = pa.schema([pa.field("image", _STRUCT), pa.field("label", _STRUCT)])


def _jpeg_bytes(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buf, format="JPEG")
    return buf.getvalue()


def _mask_png_bytes(pixels: list[int]) -> bytes:
    buf = io.BytesIO()
    im = Image.new("L", (4, 3))
    im.putdata(pixels)
    im.save(buf, format="PNG")
    return buf.getvalue()


def _row(index: int, *, path: str | None = None, mask_value: int | None = 1) -> dict:
    row = {
        "image_bytes": _jpeg_bytes(color=(index, index, index)),
        "image_path": path,
        "mask_bytes": _mask_png_bytes([mask_value] * 12) if mask_value is not None else None,
        "mask_path": f"{path}_mask.png" if path else None,
    }
    return row


def _write_shard(path: Path, rows: list[dict]) -> Path:
    table = pa.table(
        {
            "image": [{"bytes": r["image_bytes"], "path": r["image_path"]} for r in rows],
            "label": [{"bytes": r["mask_bytes"], "path": r["mask_path"]} for r in rows],
        },
        schema=_SCHEMA,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _plan(*, expected_rows: int, classes: int = 4) -> ParquetPlan:
    return ParquetPlan(
        shard_pattern=re.compile(r"^(?P<split>train|validation|test)-\d{5}-of-\d{5}\.parquet$"),
        image_column="image",
        mask_column="label",
        expected_rows=expected_rows,
        partition="default",
        version="v1",
        classes=classes,
        license_text="Apache License 2.0 (fixture)\n",
    )


def _source(
    *, source_id: str = "fixtureparquet", tier: Tier = Tier.PERMISSIVE, classes: int = 4
) -> Source:
    return Source(
        id=source_id,
        name="Fixture Parquet",
        description="Synthetic parquet ingest fixture source.",
        version="v1",
        licence=Licence(id="Apache-2.0", name="Apache License 2.0", tier=tier),
        notes="fixture: denied on purpose for test coverage" if tier == Tier.PROHIBITED else None,
        verification=Verification(
            verified_on=date(2026, 9, 18), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.HUGGINGFACE,
            uri="https://huggingface.co/datasets/fixture/parquet",
            params={"hf_id": "fixture/parquet", "config": "default", "split": "train"},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(
            layout="image-mask-pairs", schema_id="dataset-native", crosswalk_id="fixture-40class"
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


def _stage(tmp_path: Path, shard_specs: dict[str, list[dict]], **plan_kwargs):
    shard_dir = tmp_path / "shards"
    shards = [_write_shard(shard_dir / name, rows) for name, rows in shard_specs.items()]
    total = sum(len(rows) for rows in shard_specs.values())
    plan = _plan(expected_rows=plan_kwargs.pop("expected_rows", total), **plan_kwargs)
    source = _source(classes=plan.classes)
    result = _stage_with_parquet_plan(
        source, plan, shards, out_root=tmp_path / "out", profile=_profile()
    )
    return result, source, plan


# ------------------------------------------------------------------------------ tests


def test_parquet_tree_matches_archive_tree_shape(tmp_path: Path) -> None:
    specs = {
        "train-00000-of-00001.parquet": [_row(0, path="a"), _row(1, path="b"), _row(2, path="c")],
        "validation-00000-of-00001.parquet": [_row(3, path="d"), _row(4, path="e")],
    }
    result, _, plan = _stage(tmp_path, specs)

    root = result.root
    assert (root / "images" / plan.partition).is_dir()
    assert (root / "labels" / "masks" / plan.partition).is_dir()
    assert (root / "metadata.parquet").is_file()
    assert (root / "SOURCE.json").is_file()
    assert (root / "ANNOTATIONS.json").is_file()
    assert (root / "LICENSE").is_file()
    assert (root / "CHECKSUMS.sha256").is_file()
    assert result.images == 5
    assert len(list((root / "images" / plan.partition).glob("*"))) == 5
    assert len(list((root / "labels" / "masks" / plan.partition).glob("*"))) == 5


def test_image_bytes_are_verbatim(tmp_path: Path) -> None:
    payload = _jpeg_bytes(color=(7, 8, 9))
    row = {"image_bytes": payload, "image_path": "only", "mask_bytes": None, "mask_path": None}
    specs = {"train-00000-of-00001.parquet": [row]}
    result, _, plan = _stage(tmp_path, specs)

    staged = next((result.root / "images" / plan.partition).glob("*"))
    assert staged.read_bytes() == payload


def test_mask_is_indexed_png_below_classes(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a", mask_value=2)]}
    result, _, plan = _stage(tmp_path, specs, classes=4)

    staged_mask = next((result.root / "labels" / "masks" / plan.partition).glob("*"))
    with Image.open(staged_mask) as im:
        assert im.mode == "P"
        assert max(im.tobytes()) < 4


def test_mask_index_out_of_range_raises(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a", mask_value=4)]}
    with pytest.raises(ValueError, match="out of range"):
        _stage(tmp_path, specs, classes=4)


def test_row_count_mismatch_raises_and_stages_nothing(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a"), _row(1, path="b")]}
    with pytest.raises(IngestError, match="expected"):
        _stage(tmp_path, specs, expected_rows=99)

    assert not (tmp_path / "out" / "sources").exists()


def test_stem_from_path_else_deterministic_fallback(tmp_path: Path) -> None:
    specs = {
        "train-00000-of-00001.parquet": [
            _row(0, path="named_row"),
            _row(1, path=None),
        ]
    }
    result, _, plan = _stage(tmp_path, specs)

    staged = {p.stem for p in (result.root / "images" / plan.partition).glob("*")}
    assert "named_row" in staged
    assert "train-00000-000001" in staged


def test_duplicate_stem_raises(tmp_path: Path) -> None:
    specs = {
        "train-00000-of-00001.parquet": [
            _row(0, path="same"),
            _row(1, path="same"),
        ]
    }
    with pytest.raises(IngestError, match="duplicate stem"):
        _stage(tmp_path, specs)


def test_upstream_split_in_metadata_not_path(tmp_path: Path) -> None:
    specs = {
        "train-00000-of-00001.parquet": [_row(0, path="a")],
        "validation-00000-of-00001.parquet": [_row(1, path="b")],
    }
    result, _, plan = _stage(tmp_path, specs)

    for staged in (result.root / "images" / plan.partition).glob("*"):
        assert "train" not in staged.name and "validation" not in staged.name

    table = pq.read_table(result.root / "metadata.parquet")
    splits = set(table.column("upstream_split").to_pylist())
    assert splits == {"train", "validation"}
    assert set(table.column("partition").to_pylist()) == {plan.partition}


def test_restage_is_noop_same_root_digest(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a"), _row(1, path="b")]}
    shard_dir = tmp_path / "shards"
    shards = [_write_shard(shard_dir / name, rows) for name, rows in specs.items()]
    plan = _plan(expected_rows=2)
    source = _source(classes=plan.classes)
    out_root = tmp_path / "out"
    profile = _profile()

    v1 = _stage_with_parquet_plan(source, plan, shards, out_root=out_root, profile=profile)
    v2 = _stage_with_parquet_plan(source, plan, shards, out_root=out_root, profile=profile)

    assert v1.manifest.root_digest == v2.manifest.root_digest
    assert v1.images == v2.images == 2


def test_no_timestamp_in_tree(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a")]}
    result, _, _ = _stage(tmp_path, specs)

    for name in ("SOURCE.json", "ANNOTATIONS.json"):
        text = (result.root / name).read_text(encoding="utf-8")
        assert "fetched_at" not in text
        assert "staged_at" not in text


def test_shasum_c_passes(tmp_path: Path) -> None:
    specs = {"train-00000-of-00001.parquet": [_row(0, path="a"), _row(1, path="b")]}
    result, _, _ = _stage(tmp_path, specs)

    proc = subprocess.run(
        ["shasum", "-a", "256", "-c", "CHECKSUMS.sha256"],
        cwd=result.root,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_coralscapes_plan_ignore_index_is_zero() -> None:
    """The real (non-fixture) coralscapes plan: index 0 is the upstream
    unlabelled/ignore value (17.15% of pixels), not the 255 that never occurs."""
    from marinedata.ingest_parquet import _PARQUET_PLANS

    assert _PARQUET_PLANS["coralscapes"].ignore_index == 0


# ---------------------------------------------------------- _finish_staging (D3a1)


def _call_finish_staging(tmp_path: Path, *, mask_count: int, geometries=None):
    """Call :func:`_finish_staging` directly with the minimal fixture arguments it
    needs — no shard/plan machinery, since ``geometries`` and ``mask_encoder`` are its
    own concerns, not the staging loop's."""
    version_root = tmp_path / "out"
    version_root.mkdir(parents=True, exist_ok=True)
    source = _source(classes=4)
    decision = gate.Decision(source_id=source.id, allowed=True, reason="fixture")
    result = _finish_staging(
        version_root,
        source,
        _profile(),
        decision,
        staged_rows=[],
        recorded={},
        classes=4,
        license_text="Apache License 2.0 fixture text",
        mask_count=mask_count,
        images_without_annotation=0,
        observed_indices=frozenset(),
        upstream=[],
        ignore_index=None,
        fetched_uri="",
        stem_rule="hf-struct-path-basename-else-split-shard-row",
        geometries=geometries,
    )
    return result, version_root


def test_finish_staging_writes_given_geometries(tmp_path: Path) -> None:
    """A caller-supplied ``geometries`` sequence is written to ``ANNOTATIONS.json``
    verbatim — the points path: no mask-derived geometry is built or mixed in."""
    given = [
        {
            "kind": "point",
            "path": "labels/points.parquet",
            "format": "parquet-points",
            "schema_id": "mermaid-attributes",
            "crosswalk_id": None,
            "classes": 0,
            "raster_ignore_value": None,
            "images_covered": 3,
            "rows": 25,
            "supervises": ["taxon", "form"],
            "observed_indices": [],
        }
    ]
    _, root = _call_finish_staging(tmp_path, mask_count=0, geometries=given)

    payload = json.loads((root / "ANNOTATIONS.json").read_text(encoding="utf-8"))
    assert payload["geometries"] == given


def test_finish_staging_omits_mask_encoder_without_masks(tmp_path: Path) -> None:
    """``mask_encoder`` would be false provenance on a path that never ran pillow over
    anything — a points-only stage with ``mask_count == 0``."""
    _, root = _call_finish_staging(tmp_path, mask_count=0, geometries=None)

    payload = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    assert "mask_encoder" not in payload["_ingest"]


def test_finish_staging_default_geometries_unchanged(tmp_path: Path) -> None:
    """``geometries=None`` with masks present reproduces today's mask-derived
    geometry byte-for-byte — including that ``mask_encoder`` IS present here."""
    _, root = _call_finish_staging(tmp_path, mask_count=3, geometries=None)

    payload = json.loads((root / "ANNOTATIONS.json").read_text(encoding="utf-8"))
    assert payload["geometries"] == [
        {
            "kind": "dense-mask",
            "path": "labels/masks/",
            "format": "png-indexed",
            "schema_id": "dataset-native",
            "crosswalk_id": "fixture-40class",
            "classes": 4,
            "raster_ignore_value": None,
            "images_covered": 3,
            "rows": None,
            "supervises": ["taxon"],
            "observed_indices": [],
        }
    ]

    source_payload = json.loads((root / "SOURCE.json").read_text(encoding="utf-8"))
    assert source_payload["_ingest"]["mask_encoder"] == "pillow/11.0.0"
