"""WP2: the Coralscapes V2 stager. No network: shards are tiny synthetic parquet files built
with pyarrow + PIL in ``tmp_path``; ``snapshot_download`` is a stub."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata import checksums
from marinedata import ingest_coralscapes_v2 as cs
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
from marinedata.sample_schema import V2_FIELD_NAMES, validate_table

_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
FRAMES = [
    ("site1_000001_016200", "site1", "train"),
    ("site1_000002_002600", "site1", "train"),
    ("site2_000001_000100", "site2", "test"),
]
EXPECT = {"train": 2, "test": 1}
STEM1 = "site1_000001_016200"


def _png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _frame(index: int, stem: str, site: str) -> dict:
    label = Image.new("L", (8, 4))
    label.putdata([0, 1, 2, 3] * 8)
    instance = np.full((4, 8), 25 * 2000 + index, dtype=np.uint16)
    return {
        "stem": stem,
        "site": site,
        "image": _png(Image.new("RGB", (8, 4), (index * 20, 10, 30))),
        "label": _png(label),
        "instance_array": instance,
        "instance": _png(Image.fromarray(instance)),
    }


def _write_shard(path: Path, frames: list[dict], *, instance_as_array: bool) -> Path:
    instance_type = pa.list_(pa.list_(pa.uint16())) if instance_as_array else _STRUCT
    schema = pa.schema(
        [
            ("image", _STRUCT),
            ("label", _STRUCT),
            ("instance_ids", instance_type),
            ("stem", pa.string()),
            ("site", pa.string()),
        ]
    )
    table = pa.table(
        {
            "image": [
                {"bytes": f["image"], "path": f"{f['stem']}_leftImg8bit.png"} for f in frames
            ],
            "label": [
                {"bytes": f["label"], "path": f"{f['stem']}_gtFine_labelIds.png"} for f in frames
            ],
            "instance_ids": [
                f["instance_array"].tolist()
                if instance_as_array
                else {"bytes": f["instance"], "path": f"{f['stem']}_gtFine_instanceIds.png"}
                for f in frames
            ],
            "stem": [f["stem"] for f in frames],
            "site": [f["site"] for f in frames],
        },
        schema=schema,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def _make_shards(tmp_path: Path, *, instance_as_array: bool = False):
    frames = {s: _frame(i, s, site) for i, (s, site, _) in enumerate(FRAMES)}
    by_split: dict[str, list[dict]] = {}
    for stem, _, split in FRAMES:
        by_split.setdefault(split, []).append(frames[stem])
    shards = [
        _write_shard(
            tmp_path / "hf" / "instances" / f"{split}-00000-of-00001.parquet",
            rows,
            instance_as_array=instance_as_array,
        )
        for split, rows in sorted(by_split.items())
    ]
    return shards, frames


def _source() -> Source:
    return Source(
        id="coralscapes-v2",
        name="Coralscapes V2 (fixture)",
        description="Synthetic V2 fixture source.",
        version="2.0",
        licence=Licence(id="Apache-2.0", name="Apache License 2.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 10, 29), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.HUGGINGFACE,
            uri="https://huggingface.co/datasets/josauder/coralscapesV2",
            params={"hf_id": "josauder/coralscapesV2", "config": "instances"},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(
            layout="image-mask-pairs", schema_id="coralscapes-95", crosswalk_id="coralscapes-95"
        ),
        annotations=(
            Annotation(kind=AnnotationKind.DENSE_MASK, classes=96, supervises=("taxon",)),
        ),
    )


def _profile() -> Profile:
    return Profile(
        id="research",
        description="fixture profile",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
        ships=False,
    )


def _stage(tmp_path: Path, shards: list[Path], out: str = "out", **kwargs):
    kwargs.setdefault("expected_rows", EXPECT)
    kwargs.setdefault("progress", lambda done, total: None)
    return cs.stage(_source(), _profile(), shards, out_root=tmp_path / out, **kwargs)


def _files(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_stage_tree_layout(tmp_path: Path) -> None:
    shards, frames = _make_shards(tmp_path)
    result = _stage(tmp_path, shards)
    root = result.root
    assert root == tmp_path / "out" / "sources" / "coralscapes-v2" / "2.0"
    expected = (
        ["ANNOTATIONS.json", "CHECKSUMS.sha256", "LICENSE", "SOURCE.json", "metadata.parquet"]
        + [f"images/default/{s}_leftImg8bit.png" for s in frames]
        + [f"labels/masks/default/{s}_leftImg8bit.png" for s in frames]
        + [f"labels/instances/default/{s}_instanceIds.png" for s in frames]
    )
    assert _files(root) == sorted(expected)
    assert result.images == 3
    ann = json.loads((root / "ANNOTATIONS.json").read_text())
    assert [ann[k] for k in ("source_id", "version", "images", "images_without_annotation")] == [
        "coralscapes-v2",
        "2.0",
        3,
        0,
    ]
    assert [g["path"] for g in ann["geometries"]] == ["labels/masks/", "labels/instances/"]
    assert ann["geometries"][0]["observed_indices"] == [0, 1, 2, 3]
    assert ann["geometries"][1]["label_divisor"] == 2000
    src = json.loads((root / "SOURCE.json").read_text())
    assert src["_ingest"]["revision"] == cs.DEFAULT_REVISION
    assert src["checksums"] is None
    assert "Apache License" in (root / "LICENSE").read_text()


def test_stage_checksum_lines_match_files(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    result = _stage(tmp_path, shards)
    root = result.root
    lines = (root / "CHECKSUMS.sha256").read_text().splitlines()
    files = [f for f in _files(root) if f != "CHECKSUMS.sha256"]
    assert [line.split("  ", 1)[1] for line in lines] == files
    for line in lines:
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == digest
    assert result.manifest.files == len(files)
    assert result.manifest.root_digest == checksums.file_digest(root / "CHECKSUMS.sha256")


def test_stage_root_digest_identical_across_runs(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    first = _stage(tmp_path, shards, out="a")
    second = _stage(tmp_path, shards, out="b")
    again = _stage(tmp_path, shards, out="a")  # finished tree: verify only
    assert first.manifest.root_digest == second.manifest.root_digest
    assert again.manifest.root_digest == first.manifest.root_digest
    assert (first.root / "CHECKSUMS.sha256").read_bytes() == (
        second.root / "CHECKSUMS.sha256"
    ).read_bytes()


def test_stage_metadata_parquet_columns_match_v1_schema(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    root = _stage(tmp_path, shards).root
    table = pq.read_table(root / "metadata.parquet")
    assert len(V2_FIELD_NAMES) == 37
    assert table.column_names == [*V2_FIELD_NAMES, "partition"]
    assert table.schema.metadata[b"marinedata.sample_schema"] == b"2"
    validate_table(table)
    rows = {r["stem"]: r for r in table.to_pylist()}
    assert list(rows) == sorted(rows)
    row = rows[f"{STEM1}_leftImg8bit"]
    assert row["sample_id"] == f"coralscapes-v2/{STEM1}_leftImg8bit"
    assert (row["source_id"], row["source_version"], row["license"]) == (
        "coralscapes-v2",
        "2.0",
        "Apache-2.0",
    )
    assert (row["partition"], row["width"], row["height"]) == ("default", 8, 4)
    assert (row["upstream_id"], row["split_hint"], row["upstream_split"]) == (
        STEM1,
        "train",
        "train",
    )
    assert row["upstream_path"] == f"{STEM1}_leftImg8bit.png"
    assert row["split_group"] == "coralscapes-v2/site1"
    assert rows["site2_000001_000100_leftImg8bit"]["split_group"] == "coralscapes-v2/site2"
    assert rows["site2_000001_000100_leftImg8bit"]["upstream_split"] == "test"


def test_stage_writes_parquet_bytes_verbatim(tmp_path: Path) -> None:
    shards, frames = _make_shards(tmp_path)
    root = _stage(tmp_path, shards).root
    meta = {r["upstream_id"]: r for r in pq.read_table(root / "metadata.parquet").to_pylist()}
    for stem, frame in frames.items():
        paths = {
            "image": root / f"images/default/{stem}_leftImg8bit.png",
            "label": root / f"labels/masks/default/{stem}_leftImg8bit.png",
            "instance": root / f"labels/instances/default/{stem}_instanceIds.png",
        }
        for key, path in paths.items():
            assert path.read_bytes() == frame[key]
            assert checksums.file_digest(path) == hashlib.sha256(frame[key]).hexdigest()
        assert meta[stem]["image_sha256"] == hashlib.sha256(frame["image"]).hexdigest()
        assert meta[stem]["image_bytes"] == len(frame["image"])


def test_stage_instance_array_becomes_uint16_png(tmp_path: Path) -> None:
    shards, frames = _make_shards(tmp_path, instance_as_array=True)
    root = _stage(tmp_path, shards).root
    for stem, frame in frames.items():
        with Image.open(root / f"labels/instances/default/{stem}_instanceIds.png") as im:
            assert im.mode == "I;16"
            assert np.array_equal(np.asarray(im), frame["instance_array"])


def test_stage_resumes_without_rewriting_matching_files(tmp_path: Path) -> None:
    shards, frames = _make_shards(tmp_path)
    first = _stage(tmp_path, shards)
    root = first.root
    keep = root / f"images/default/{STEM1}_leftImg8bit.png"
    bad = root / "images/default/site1_000002_002600_leftImg8bit.png"
    gone = root / "labels/masks/default/site1_000002_002600_leftImg8bit.png"
    kept_ns = keep.stat().st_mtime_ns
    bad.write_bytes(b"x" * bad.stat().st_size)  # same size, wrong bytes
    gone.unlink()
    (root / "CHECKSUMS.sha256").unlink()  # an interrupted run never reached the manifest
    second = _stage(tmp_path, shards)
    assert keep.stat().st_mtime_ns == kept_ns
    assert bad.read_bytes() == frames["site1_000002_002600"]["image"]
    assert gone.is_file()
    assert second.manifest.root_digest == first.manifest.root_digest


def test_stage_limit_and_progress(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    ticks: list[tuple[int, int]] = []
    result = _stage(
        tmp_path, shards, limit=2, progress=lambda d, t: ticks.append((d, t)), progress_every=1
    )
    assert result.images == 2
    assert ticks == [(1, 2), (2, 2)]


def test_stage_refuses_a_short_tree_before_writing(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    with pytest.raises(IngestError, match="short tree"):
        _stage(tmp_path, shards, expected_rows=cs.EXPECTED_SPLIT_ROWS)
    assert not (tmp_path / "out").exists()


def test_stage_refuses_non_png_bytes(tmp_path: Path) -> None:
    frame = _frame(0, STEM1, "site1")
    frame["image"] = b"\xff\xd8\xff" + b"0" * 40
    shard = _write_shard(
        tmp_path / "hf" / "instances" / "train-00000-of-00001.parquet",
        [frame],
        instance_as_array=False,
    )
    with pytest.raises(IngestError, match="not a PNG"):
        _stage(tmp_path, [shard], expected_rows={"train": 1})


def test_stage_find_shards_needs_a_download_first(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="--download first"):
        cs.find_shards(tmp_path / "empty")
    shards, _ = _make_shards(tmp_path)
    assert cs.find_shards(tmp_path / "hf") == sorted(shards, key=lambda p: p.name)


def test_stage_download_pins_revision_and_patterns(tmp_path: Path) -> None:
    calls: list[dict] = []
    cs.download(tmp_path / "hf", snapshot_download=lambda **kw: calls.append(kw))
    assert len(cs.DEFAULT_REVISION) == 40
    assert calls == [
        {
            "repo_id": "josauder/coralscapesV2",
            "repo_type": "dataset",
            "revision": cs.DEFAULT_REVISION,
            "allow_patterns": ["instances/*", "README.md", "LICENSE*", "extended_metadata.json"],
            "local_dir": str(tmp_path / "hf"),
        }
    ]


def test_stage_resolve_revision_prefix_must_match_head() -> None:
    full = "8a4567b2" + "0" * 32
    assert cs.resolve_revision("8a4567b2", fetch_json=lambda url: {"sha": full}) == full
    assert cs.resolve_revision(full) == full
    with pytest.raises(IngestError, match="HEAD is"):
        cs.resolve_revision("deadbeef", fetch_json=lambda url: {"sha": full})


def test_stage_cli_needs_an_action(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cs.main(["--hf-cache", str(tmp_path)])


def test_stage_limit_tree_is_not_reused_for_a_full_run(tmp_path: Path) -> None:
    shards, _ = _make_shards(tmp_path)
    _stage(tmp_path, shards, limit=2)
    with pytest.raises(IngestError, match="fresh --out"):
        _stage(tmp_path, shards)
    assert _stage(tmp_path, shards, limit=2).images == 2
