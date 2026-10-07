"""Shared fixtures for the ingest pipeline tests (I2, D1 §7), split out of
``test_ingest.py`` (D2e) so ``tests/test_ingest_staging.py`` and
``tests/test_ingest_content.py`` both stay under the 400-line norm without duplicating
the SUIM zip/plan/source builders.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date
from pathlib import Path

import pytest

from marinedata import fetch
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
from marinedata.ingest import ArchivePlan, _stage_with_plan
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
