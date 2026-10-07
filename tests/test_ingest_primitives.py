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


def _rgb_bmp_mask(path: Path, pixels: list[tuple[int, int, int]], size: tuple[int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", size)
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


@pytest.mark.parametrize(
    "key",
    [
        "fetched_on",
        "ingest_time",
        "timestamp",
        "Created",
        "modified",
        "updated",
        "mtime",
        "ctime",
        "staged_at",
        "run_ts",
    ],
)
def test_timestamp_guard_rejects_date_like_keys(tmp_path: Path, key: str) -> None:
    source = make_source("image-mask-pairs")
    with pytest.raises(ValueError, match=key):
        write_source_json(tmp_path / "SOURCE.json", source, {key: "2026-01-01"})


def test_timestamp_guard_rejects_nested_date_like_key(tmp_path: Path) -> None:
    source = make_source("image-mask-pairs")
    with pytest.raises(ValueError, match="fetched_on"):
        write_source_json(
            tmp_path / "SOURCE.json",
            source,
            {"nested": {"fetched_on": "2026-01-01"}},
        )


def test_timestamp_guard_accepts_fetched_uri(tmp_path: Path) -> None:
    source = make_source("image-mask-pairs")
    path = tmp_path / "SOURCE.json"
    write_source_json(path, source, {"fetched_uri": "https://example.test/data.zip"})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["_ingest"]["fetched_uri"] == "https://example.test/data.zip"


def test_timestamp_guard_allows_registry_date_keys(tmp_path: Path) -> None:
    from marinedata.manifests import _REGISTRY_DATE_KEYS, _assert_no_timestamp_keys

    for key in _REGISTRY_DATE_KEYS:
        _assert_no_timestamp_keys({key: "2026-01-01"})  # must not raise


def test_allowlist_keys_are_registry_model_fields() -> None:
    import inspect

    from pydantic import BaseModel

    from marinedata import models
    from marinedata.manifests import _REGISTRY_DATE_KEYS

    all_fields: set[str] = set()
    for _name, cls in inspect.getmembers(models, inspect.isclass):
        if issubclass(cls, BaseModel) and cls.__module__ == models.__name__:
            all_fields.update(cls.model_fields)

    for key in _REGISTRY_DATE_KEYS:
        assert key in all_fields, f"{key} is not a field of any model in models.py"


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


def test_rgb_bitpacked_mask_decodes_to_schema_indices(tmp_path: Path) -> None:
    # index = 4*R + 2*G + 1*B (each channel thresholded at 127) — one pixel per class.
    pixels = [
        (0, 0, 0),  # 0
        (0, 0, 255),  # 1 (HD)
        (0, 255, 0),  # 2
        (0, 255, 255),  # 3 (WR)
        (255, 0, 0),  # 4 (RO)
        (255, 0, 255),  # 5 (RI)
        (255, 255, 0),  # 6 (FV)
        (255, 255, 255),  # 7
    ]
    src = tmp_path / "mask.bmp"
    _rgb_bmp_mask(src, pixels, size=(8, 1))
    dest = tmp_path / "mask.png"
    digest, observed = bmp_mask_to_indexed_png(src, dest, classes=8)
    assert observed == frozenset(range(8))
    assert digest == file_digest(dest)

    from PIL import Image

    with Image.open(dest) as im:
        assert im.mode == "P"
        assert list(im.getdata()) == list(range(8))


def test_rgb_mask_noisy_channel_values_threshold_at_127(tmp_path: Path) -> None:
    # Off-pure values either side of the 127 threshold must still resolve to 0/1 bits.
    pixels = [(0, 0, 128), (10, 5, 255), (0, 127, 0), (5, 200, 20)]
    src = tmp_path / "mask.bmp"
    _rgb_bmp_mask(src, pixels, size=(4, 1))
    dest = tmp_path / "mask.png"
    _digest, observed = bmp_mask_to_indexed_png(src, dest, classes=8)
    # (0,0,128)->1, (10,5,255)->1, (0,127,0)->0 (127 is not > 127), (5,200,20)->2
    assert observed == frozenset({0, 1, 2})


def test_unsupported_mask_mode_raises(tmp_path: Path) -> None:
    from PIL import Image

    src = tmp_path / "mask.bmp"
    src.parent.mkdir(parents=True, exist_ok=True)
    # BMP's native 1-bit bilevel mode ("1") round-trips losslessly but is neither
    # 'L'/'P' (single-channel index) nor 'RGB' (bitpacked) — must raise, not silently
    # coerce.
    Image.new("1", (2, 2), 0).save(src, format="BMP")
    with pytest.raises(ValueError, match="not a supported mask mode"):
        bmp_mask_to_indexed_png(src, tmp_path / "mask.png", classes=8)


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


# ---------------------------------------------------------- PointRow extras (D3a1)


def test_points_table_nullable_extras_roundtrip(tmp_path: Path) -> None:
    """``label_id``/``form``/``region`` round-trip through the parquet file, nullable,
    while the rest of the row shape is untouched."""
    import pyarrow.parquet as pq

    rows = [
        PointRow(
            stem="s0",
            partition="default",
            row=10,
            col=20,
            label="Porites",
            schema_id="mermaid-attributes",
            label_id="attr-1",
            form="Branching",
            region="Pacific",
        ),
        PointRow(
            stem="s1",
            partition="default",
            row=5,
            col=6,
            label="Sand",
            schema_id="mermaid-attributes",
        ),
    ]
    path = tmp_path / "points.parquet"
    write_points_table(path, rows)

    table = pq.read_table(path)
    assert table.column_names == [
        "stem",
        "partition",
        "row",
        "col",
        "label",
        "schema_id",
        "label_id",
        "form",
        "region",
    ]
    by_stem = dict(zip(table.column("stem").to_pylist(), range(table.num_rows), strict=True))
    row0 = by_stem["s0"]
    assert table.column("label_id")[row0].as_py() == "attr-1"
    assert table.column("form")[row0].as_py() == "Branching"
    assert table.column("region")[row0].as_py() == "Pacific"
    row1 = by_stem["s1"]
    assert table.column("label_id")[row1].as_py() is None
    assert table.column("form")[row1].as_py() is None
    assert table.column("region")[row1].as_py() is None
    assert table.field("label_id").nullable
    assert table.field("form").nullable
    assert table.field("region").nullable


def test_points_table_two_writes_byte_identical(tmp_path: Path) -> None:
    """Writing the same rows twice — including the new nullable extras — yields the
    same digest, matching the determinism the other two tables already guarantee."""
    rows = [
        PointRow(
            stem=f"s{i % 3}",
            partition="default",
            row=i,
            col=i * 2,
            label="HC",
            schema_id="mermaid-attributes",
            label_id=f"attr-{i}" if i % 2 else None,
            form="Branching" if i % 2 else None,
            region="Pacific",
        )
        for i in range(6)
    ]
    d1 = write_points_table(tmp_path / "a.parquet", rows)
    d2 = write_points_table(tmp_path / "b.parquet", rows)
    assert d1 == d2


def test_points_table_existing_callers_unaffected(tmp_path: Path) -> None:
    """A caller that only knows the old six fields — no ``label_id``/``form``/
    ``region`` keyword — still constructs and writes exactly as before."""
    rows = [
        PointRow(
            stem="s0",
            partition="default",
            row=1,
            col=2,
            label="HC",
            schema_id="dataset-native",
        )
    ]
    digest = write_points_table(tmp_path / "points.parquet", rows)
    assert digest == file_digest(tmp_path / "points.parquet")


def test_image_labels_table_roundtrip(tmp_path: Path) -> None:
    """``ImageLabelRow`` round-trips through ``write_image_labels_table``: schema,
    the nullable ``confidence`` extra, and multiple rows per ``(partition, stem)`` for
    a multi-label image (WS-D S23)."""
    import pyarrow.parquet as pq

    from marinedata.tables import ImageLabelRow, write_image_labels_table

    rows = [
        ImageLabelRow(
            stem="s0", partition="default", label="Healthy", schema_id="roboflow-bleaching-native"
        ),
        ImageLabelRow(
            stem="s1",
            partition="default",
            label="Unhealthy",
            schema_id="roboflow-bleaching-native",
            confidence=0.9,
        ),
        ImageLabelRow(
            stem="s1",
            partition="default",
            label="Bleached",
            schema_id="roboflow-bleaching-native",
            confidence=0.8,
        ),
    ]
    path = tmp_path / "image_labels.parquet"
    write_image_labels_table(path, rows)

    table = pq.read_table(path)
    assert table.column_names == ["stem", "partition", "label", "schema_id", "confidence"]
    assert table.field("confidence").nullable
    by_stem = table.column("stem").to_pylist()
    assert by_stem.count("s1") == 2
    confidences = dict(
        zip(table.column("stem").to_pylist(), table.column("confidence").to_pylist(), strict=True)
    )
    assert confidences["s0"] is None


def test_image_labels_table_two_writes_byte_identical(tmp_path: Path) -> None:
    """Same determinism guarantee as the other three tables."""
    from marinedata.tables import ImageLabelRow, write_image_labels_table

    rows = [
        ImageLabelRow(
            stem=f"s{i}",
            partition="default",
            label="Healthy",
            schema_id="roboflow-bleaching-native",
        )
        for i in range(5)
    ]
    d1 = write_image_labels_table(tmp_path / "a.parquet", rows)
    d2 = write_image_labels_table(tmp_path / "b.parquet", rows)
    assert d1 == d2
