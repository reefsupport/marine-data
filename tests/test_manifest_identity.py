"""D-X v1 byte-identity, proved at manifest level with no image bytes (INT-core2b).

No test here fetches an image, or even a real ``CHECKSUMS.sha256`` over the network:
:func:`marinedata.manifest_identity.build_v1_row_manifest` takes its checksum lookup as
a callable, so the row-reconstruction and comparison logic is exercised end to end with
a fake fetcher. :func:`marinedata.manifest_identity.fetch_source_checksums` itself (the
one real network call this module makes) is exercised separately with ``_get``
monkeypatched — see ``test_fetch_source_checksums_*`` below.

The real, network-backed run (charter D-E: anonymous HTTPS is allowed) was done once by
hand for the INT-core2b report — see ``docs/integration-v2.md``'s "INT-core2b" section
for the row/column counts it produced.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from marinedata.enums import (
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.manifest_identity import (
    ManifestRow,
    build_v1_row_manifest,
    compare_v1_manifest,
    compare_v1_metadata_columns,
    fetch_source_checksums,
    load_published_v1_metadata,
    load_published_v1_metadata_table,
    rebuild_v1_metadata_rows,
)
from marinedata.models import (
    Access,
    Checksums,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    Verification,
)
from marinedata.registry import Registry
from marinedata.splitmap import SplitMap
from marinedata.tables import StagedImage, write_metadata_table


def _source(source_id: str) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the D-X manifest-identity test.",
        version="v1",
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 8, 17), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.S3,
            uri="s3://rs-storage-open",
            params={"bucket": "rs-storage-open", "prefix": f"sources/{source_id}/v1/"},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout="staged-tree", params={}),
        checksums=Checksums(version="v1", root_digest="0" * 64, files=2, size_bytes=10),
        annotations=(),
    )


def _registry(sources: dict[str, Source]) -> Registry:
    profile = Profile(
        id="research",
        description="fixture",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )
    return Registry(
        sources=sources, licences={}, profiles={"research": profile}, schemas={}, tasks={}
    )


def _stage_metadata(root: Path, rows: list[tuple[str, str, int, int]]) -> None:
    """``rows``: ``(stem, split_group, width, height)`` — no image bytes are ever
    written; ``metadata.parquet`` is the staging pipeline's sample index, not a claim
    that ``images/`` exists on disk."""
    staged = [
        StagedImage(
            stem=stem,
            partition="p",
            upstream_path=f"orig/{stem}.jpg",
            upstream_split=None,
            width=width,
            height=height,
            split_group=group,
        )
        for stem, group, width, height in rows
    ]
    write_metadata_table(root / "metadata.parquet", staged)


def test_build_v1_row_manifest_uses_checksums_not_image_bytes(tmp_path: Path) -> None:
    source = _source("fixture-src")
    registry = _registry({"fixture-src": source})
    cache_dir = tmp_path / "cache"
    _stage_metadata(cache_dir / "fixture-src", [("s1", "g1", 10, 20), ("s2", "g2", 30, 30)])
    # No `images/` directory anywhere under cache_dir — proves no image bytes are read.
    assert not (cache_dir / "fixture-src" / "images").exists()

    checksums = {"images/p/s1.jpg": "a" * 64, "images/p/s2.png": "b" * 64}
    split_map = SplitMap(
        by="group",
        seed=0,
        ratios={"train": 0.7, "val": 0.15, "test": 0.15},
        assignments={"g1": "train", "g2": "val"},
    )
    skipped: dict[str, str] = {}
    rows = build_v1_row_manifest(
        registry,
        split_map,
        cache_dir=cache_dir,
        fetch_checksums=lambda src: checksums,
        skipped=skipped,
    )

    assert not skipped
    assert sorted(rows, key=lambda r: r.sha256) == [
        ManifestRow(
            sha256="a" * 64,
            split="train",
            source_id="fixture-src",
            width=10,
            height=20,
            path="images/p/s1.jpg",
        ),
        ManifestRow(
            sha256="b" * 64,
            split="val",
            source_id="fixture-src",
            width=30,
            height=30,
            path="images/p/s2.png",
        ),
    ]


def test_build_v1_row_manifest_skips_source_with_no_local_metadata(tmp_path: Path) -> None:
    registry = _registry({"fixture-src": _source("fixture-src")})
    skipped: dict[str, str] = {}
    rows = build_v1_row_manifest(
        registry,
        SplitMap(by="group", seed=0, ratios={"train": 1.0}),
        cache_dir=tmp_path / "empty-cache",
        fetch_checksums=lambda src: {},
        skipped=skipped,
    )
    assert rows == []
    assert "fixture-src" in skipped


def test_compare_v1_manifest_flags_missing_split_and_min_side_mismatches() -> None:
    published = {
        "a" * 64: ("train", 20),  # matches
        "b" * 64: ("val", 30),  # rebuild disagrees on split
        "c" * 64: ("test", 5),  # rebuild disagrees on min_side
        "d" * 64: ("train", 1),  # missing from the rebuild entirely
    }
    rows = [
        ManifestRow(sha256="a" * 64, split="train", source_id="s", width=20, height=99),
        ManifestRow(sha256="b" * 64, split="train", source_id="s", width=30, height=99),
        ManifestRow(sha256="c" * 64, split="test", source_id="s", width=8, height=99),
        ManifestRow(sha256="e" * 64, split="train", source_id="s", width=1, height=1),
    ]
    report = compare_v1_manifest(rows, published)
    assert report.missing_from_rebuild == ("d" * 64,)
    assert report.extra_in_rebuild == ("e" * 64,)
    assert report.split_mismatches == ("b" * 64,)
    assert report.min_side_mismatches == ("c" * 64,)
    assert not report.identical


def test_compare_v1_manifest_identical_when_everything_matches() -> None:
    published = {"a" * 64: ("train", 20)}
    rows = [ManifestRow(sha256="a" * 64, split="train", source_id="s", width=20, height=99)]
    assert compare_v1_manifest(rows, published).identical


def test_fetch_source_checksums_verifies_root_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    import marinedata.manifest_identity as mod

    manifest_text = "a" * 64 + "  images/p/s1.jpg\n"
    manifest_bytes = manifest_text.encode()
    expected_digest = hashlib.sha256(manifest_bytes).hexdigest()

    calls: list[str] = []

    def fake_get(url: str, **kwargs: object) -> bytes:
        calls.append(url)
        return manifest_bytes

    monkeypatch.setattr(mod, "_get", fake_get)
    source = _source("fixture-src").model_copy(
        update={
            "checksums": Checksums(version="v1", root_digest=expected_digest, files=1, size_bytes=1)
        }
    )

    result = fetch_source_checksums(source)

    assert result == {"images/p/s1.jpg": "a" * 64}
    assert len(calls) == 1
    assert "CHECKSUMS.sha256" in calls[0]


def test_fetch_source_checksums_raises_on_digest_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    import marinedata.manifest_identity as mod

    monkeypatch.setattr(mod, "_get", lambda url, **kwargs: b"a" * 64 + b"  images/p/s1.jpg\n")
    source = _source("fixture-src")  # root_digest is "0" * 64 — never matches

    with pytest.raises(ValueError, match="does not match"):
        fetch_source_checksums(source)


def test_load_published_v1_metadata_reads_non_image_columns_only(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "hf_v1"
    (root / "data" / "metadata").mkdir(parents=True)
    table = pa.table({"image_sha256": ["a" * 64], "min_side": [42], "license": ["CC-BY-4.0"]})
    pq.write_table(table, root / "data" / "metadata" / "train-00000-of-00001.parquet")

    published = load_published_v1_metadata(root)

    assert published == {"a" * 64: ("train", 42)}


def test_load_published_v1_metadata_table_keeps_every_column_plus_split(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = tmp_path / "hf_v1"
    (root / "data" / "metadata").mkdir(parents=True)
    table = pa.table({"image_sha256": ["b" * 64], "min_side": [7], "license": ["CC0-1.0"]})
    pq.write_table(table, root / "data" / "metadata" / "validation-00000-of-00001.parquet")

    published = load_published_v1_metadata_table(root)

    assert published == {
        "b" * 64: {
            "image_sha256": "b" * 64,
            "min_side": 7,
            "license": "CC0-1.0",
            "split": "validation",
        }
    }


def test_compare_v1_metadata_columns_flags_only_differing_columns() -> None:
    published = {
        "a" * 64: {
            "image_sha256": "a" * 64,
            "license": "CC0-1.0",
            "q_blur": float("nan"),
            "split": "train",
        },
        "b" * 64: {"image_sha256": "b" * 64, "license": "CC0-1.0", "q_blur": 1.5, "split": "test"},
    }
    rebuilt = [
        {"image_sha256": "a" * 64, "license": "CC0-1.0", "q_blur": float("nan")},
        {"image_sha256": "b" * 64, "license": "CC-BY-4.0", "q_blur": 1.5},
        {"image_sha256": "c" * 64, "license": "x", "q_blur": 0.0},  # not published: ignored
    ]

    mismatches = compare_v1_metadata_columns(rebuilt, published)

    assert mismatches == {"image_sha256": (), "license": ("b" * 64,), "q_blur": ()}


def test_rebuild_v1_metadata_rows_passes_checksum_path_not_image_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import marinedata.metadata_release as metadata_release

    seen: dict[str, object] = {}

    def fake_build_rows(refs, registry, stage_root, quality_by_sha, meow_polygons, backfill_root):
        seen.update(refs=list(refs), stage_root=stage_root, backfill_root=backfill_root)
        return [{"image_sha256": r.sha256} for r in refs]

    monkeypatch.setattr(metadata_release, "build_rows", fake_build_rows)
    row = ManifestRow(
        sha256="d" * 64,
        split="train",
        source_id="fixture-src",
        width=4,
        height=3,
        path="images/p1/stem1.jpg",
    )

    out = rebuild_v1_metadata_rows(
        [row], _registry({}), stage_root=tmp_path, quality_by_sha={}, backfill_root=tmp_path / "bf"
    )

    assert out == [{"image_sha256": "d" * 64}]
    (ref,) = seen["refs"]
    assert (ref.file.parent.name, ref.file.stem, ref.split) == ("p1", "stem1", "train")
    assert not ref.file.exists()  # a CHECKSUMS key, never a file on disk
    assert seen["backfill_root"] == tmp_path / "bf"
