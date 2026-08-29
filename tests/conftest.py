"""Shared fixtures.

Synthetic fixtures are how ~40 loaders get honest coverage without downloading
terabytes. Each builder writes the minimal tree a layout declares, so a layout reader is
exercised against real filesystem behaviour rather than mocks.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.enums import (
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import (
    Access,
    Annotation,
    Coverage,
    Licence,
    LoaderSpec,
    Source,
    Verification,
)


@pytest.fixture(scope="session")
def registry() -> Registry:
    return Registry.load()


def make_source(
    layout: str,
    params: dict | None = None,
    source_id: str = "fixture",
    annotations: tuple[Annotation, ...] = (),
    modalities: tuple[Modality, ...] = (Modality.IMAGE,),
) -> Source:
    """A minimal source declaring a given layout."""
    return Source(
        id=source_id,
        name="Fixture",
        description="Synthetic fixture source.",
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 8, 17),
            verified_by="synthetic fixture",
            method="licence-file",
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(method=AccessMethod.HTTP, uri="https://example.invalid"),
        modalities=modalities,
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout=layout, params=params or {}),
        annotations=annotations,
    )


def _touch_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Loaders reference images lazily, so a valid PNG header is unnecessary — but
    # writing bytes keeps the file non-empty and the suffix is what dispatch uses.
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


@pytest.fixture
def image_folder_root(tmp_path: Path) -> Path:
    root = tmp_path / "image-folder"
    for class_name, count in (("Hard Coral", 3), ("Soft Coral", 2)):
        for i in range(count):
            _touch_image(root / class_name / f"img_{i}.jpg")
    return root


@pytest.fixture
def image_mask_root(tmp_path: Path) -> Path:
    root = tmp_path / "image-mask"
    for i in range(3):
        _touch_image(root / "images" / f"frame_{i}.jpg")
        _touch_image(root / "masks" / f"frame_{i}.png")
    return root


@pytest.fixture
def coco_root(tmp_path: Path) -> Path:
    root = tmp_path / "coco"
    for i in range(2):
        _touch_image(root / "images" / f"im{i}.jpg")
    doc = {
        "images": [{"id": 0, "file_name": "im0.jpg"}, {"id": 1, "file_name": "im1.jpg"}],
        "categories": [{"id": 1, "name": "Hard Coral"}, {"id": 2, "name": "fish"}],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [1, 2, 30, 40]},
            {"id": 2, "image_id": 0, "category_id": 2, "bbox": [5, 6, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5]},
        ],
    }
    (root / "annotations.json").write_text(json.dumps(doc), encoding="utf-8")
    return root


@pytest.fixture
def yolo_root(tmp_path: Path) -> Path:
    root = tmp_path / "yolo"
    for i in range(2):
        _touch_image(root / "images" / f"f{i}.jpg")
    (root / "labels").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "f0.txt").write_text("0 0.5 0.5 0.2 0.2\n1 0.1 0.1 0.05 0.05\n")
    (root / "labels" / "f1.txt").write_text("1 0.3 0.3 0.1 0.1\n")
    return root


@pytest.fixture
def opencv_cascade_root(tmp_path: Path) -> Path:
    root = tmp_path / "cascade"
    for name in ("a.jpg", "b.jpg"):
        _touch_image(root / name)
    (root / "annotations.dat").write_text(
        "a.jpg 2 10 20 30 40 50 60 70 80\nb.jpg 1 1 2 3 4\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def csv_points_root(tmp_path: Path) -> Path:
    root = tmp_path / "points"
    for name in ("a.jpg", "b.jpg"):
        _touch_image(root / "images" / name)
    with (root / "annotations.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Name", "Row", "Column", "Label"])
        writer.writerows(
            [
                ["a.jpg", 10, 20, "Hard Coral"],
                ["a.jpg", 30, 40, "Soft Coral"],
                ["b.jpg", 5, 5, "sand"],
            ]
        )
    return root


@pytest.fixture
def labelbox_root(tmp_path: Path) -> Path:
    root = tmp_path / "labelbox"
    for name in ("s1.jpg", "s2.jpg"):
        _touch_image(root / "images" / name)

    def row(external_id: str, names: list[str]) -> dict:
        return {
            "data_row": {"external_id": external_id},
            "projects": {
                "p1": {
                    "labels": [
                        {"annotations": {"objects": [{"name": n, "value": n} for n in names]}}
                    ]
                }
            },
        }

    lines = [
        json.dumps(row("s1.jpg", ["Hard Coral", "Hard Coral", "Soft Coral"])),
        json.dumps(row("s2.jpg", ["SCALE"])),
    ]
    (root / "export-result.ndjson").write_text("\n".join(lines), encoding="utf-8")
    return root


@pytest.fixture
def audio_root(tmp_path: Path) -> Path:
    root = tmp_path / "audio"
    for class_name in ("fish_chorus", "snapping_shrimp"):
        d = root / class_name
        d.mkdir(parents=True, exist_ok=True)
        (d / "clip0.wav").write_bytes(b"RIFF")
    return root
