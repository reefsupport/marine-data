"""Ingest primitives (I1): digesting writers, mask normaliser, tables, manifests.

Pipeline-level tests (re-ingest no-op, ``shasum -c``, stem collisions, ...) live in
``tests/test_ingest.py`` (I2, D1 §7) and are not duplicated here. This file only proves
the primitives each of those steps is built from: a write-side digest is correct, a
copied image is byte-identical, a re-encoded mask preserves pixel indices, a table's
bytes do not depend on row order, and a manifest carries no timestamp.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from conftest import make_source

from marinedata.checksums import copy_digest, file_digest, write_digest
from marinedata.manifests import AnnotationCounts, write_annotations_json, write_source_json
from marinedata.normalise import bmp_mask_to_indexed_png, copy_image
from marinedata.tables import BoxRow, PointRow, StagedImage, write_boxes_table, write_points_table


def _jpeg(path: Path, size: tuple[int, int] = (6, 4)) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (10, 20, 30)).save(path, format="JPEG")


def _bmp_mask(path: Path, pixels: list[int], size: tuple[int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("L", size)
    im.putdata(pixels)
    im.save(path, format="BMP")


def test_write_digest_returns_sha256_of_written_bytes(tmp_path: Path) -> None:
    payload = b"some staged bytes"
    dest = tmp_path / "out.bin"
    digest = write_digest(dest, payload)
    assert dest.read_bytes() == payload
    assert digest == file_digest(dest)


def test_copy_digest_is_byte_identical_and_returns_source_sha256(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"a" * 5000)  # bigger than a single chunk would be for tiny data
    dest = tmp_path / "copies" / "dest.bin"
    digest = copy_digest(src, dest)
    assert dest.read_bytes() == src.read_bytes()
    assert digest == file_digest(src)
    assert digest == file_digest(dest)


def test_copy_image_does_not_reencode(tmp_path: Path) -> None:
    src = tmp_path / "src.jpg"
    _jpeg(src, size=(6, 4))
    dest = tmp_path / "staged" / "src.jpg"
    digest, width, height = copy_image(src, dest)
    assert digest == file_digest(src)
    assert digest == file_digest(dest)
    assert (width, height) == (6, 4)


def test_mask_is_indexed_png_within_the_declared_class_count(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "mask.bmp"
    _bmp_mask(src, [0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 1, 1], size=(4, 3))
    dest = tmp_path / "mask.png"
    digest, observed = bmp_mask_to_indexed_png(src, dest, classes=8)
    assert observed == frozenset({0, 1, 2, 3})
    assert digest == file_digest(dest)
    with Image.open(dest) as im:
        assert im.mode == "P"
        assert set(im.getdata()) == observed

    bad = tmp_path / "bad.bmp"
    _bmp_mask(bad, [0, 1, 2, 9], size=(2, 2))
    with pytest.raises(ValueError, match="out of range"):
        bmp_mask_to_indexed_png(bad, tmp_path / "bad.png", classes=8)


def test_mask_png_bytes_are_deterministic(tmp_path: Path) -> None:
    src = tmp_path / "mask.bmp"
    _bmp_mask(src, [0, 1, 2, 3] * 3, size=(4, 3))
    digest1, _ = bmp_mask_to_indexed_png(src, tmp_path / "out1.png", classes=8)
    digest2, _ = bmp_mask_to_indexed_png(src, tmp_path / "out2.png", classes=8)
    assert digest1 == digest2


def test_parquet_bytes_are_row_order_independent(tmp_path: Path) -> None:
    image_rows = [
        StagedImage(
            stem=f"s{i}",
            partition="default",
            upstream_path=f"p{i}",
            upstream_split=None,
            width=10 + i,
            height=20 + i,
        )
        for i in range(6)
    ]
    point_rows = [
        PointRow(
            stem=f"s{i % 3}",
            partition="default",
            row=i,
            col=i * 2,
            label="HC",
            schema_id="dataset-native",
        )
        for i in range(6)
    ]
    box_rows = [
        BoxRow(
            stem=f"s{i % 3}",
            partition="default",
            x=i,
            y=i,
            w=5,
            h=5,
            label="fish",
            schema_id="dataset-native",
        )
        for i in range(6)
    ]

    from marinedata.tables import write_metadata_table

    for name, rows, writer in (
        ("metadata", image_rows, write_metadata_table),
        ("points", point_rows, write_points_table),
        ("boxes", box_rows, write_boxes_table),
    ):
        shuffled = rows[:]
        random.Random(0).shuffle(shuffled)
        assert shuffled != rows  # the shuffle actually reordered something
        d1 = writer(tmp_path / f"{name}_a.parquet", rows)
        d2 = writer(tmp_path / f"{name}_b.parquet", shuffled)
        assert d1 == d2, f"{name} table bytes depend on row order"


def test_source_json_carries_null_checksums(tmp_path: Path) -> None:
    source = make_source("image-mask-pairs")
    path = tmp_path / "SOURCE.json"
    write_source_json(path, source, {"ingest_version": 1})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checksums"] is None
    assert payload["_ingest"] == {"ingest_version": 1}


def test_source_json_rejects_timestamp_keys(tmp_path: Path) -> None:
    source = make_source("image-mask-pairs")
    with pytest.raises(ValueError, match="fetched_at"):
        write_source_json(tmp_path / "SOURCE.json", source, {"fetched_at": "2026-01-01"})


def test_every_writer_returns_the_digest_of_the_file_it_wrote(tmp_path: Path) -> None:
    source = make_source("image-mask-pairs")

    img_dest = tmp_path / "img.jpg"
    _jpeg(tmp_path / "src.jpg")
    image_digest, _w, _h = copy_image(tmp_path / "src.jpg", img_dest)
    assert image_digest == file_digest(img_dest)

    mask_src = tmp_path / "mask.bmp"
    _bmp_mask(mask_src, [0, 1, 2, 3], size=(2, 2))
    mask_dest = tmp_path / "mask.png"
    mask_digest, _observed = bmp_mask_to_indexed_png(mask_src, mask_dest, classes=8)
    assert mask_digest == file_digest(mask_dest)

    table_dest = tmp_path / "metadata.parquet"
    rows = [
        StagedImage(
            stem="s0",
            partition="default",
            upstream_path="p0",
            upstream_split=None,
            width=1,
            height=1,
        )
    ]
    from marinedata.tables import write_metadata_table

    table_digest = write_metadata_table(table_dest, rows)
    assert table_digest == file_digest(table_dest)

    source_dest = tmp_path / "SOURCE.json"
    source_digest = write_source_json(source_dest, source, {"ingest_version": 1})
    assert source_digest == file_digest(source_dest)

    annotations_dest = tmp_path / "ANNOTATIONS.json"
    counts = AnnotationCounts(images=1, images_without_annotation=0, geometries=())
    annotations_digest = write_annotations_json(annotations_dest, source, counts)
    assert annotations_digest == file_digest(annotations_dest)


def test_missing_ingest_extra_error_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "src.jpg"
    src.write_bytes(b"not actually decoded before the import check")
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError, match="ingest"):
        copy_image(src, tmp_path / "missing-dest.jpg")

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    from marinedata.tables import write_metadata_table

    with pytest.raises(ImportError, match="ingest"):
        write_metadata_table(tmp_path / "unused.parquet", [])
