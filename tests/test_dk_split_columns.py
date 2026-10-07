"""D-AI2: schema-v2 ``split_group``/``upstream_split``/``upstream_path`` on the D-K
``SampleRow``, coralseg's one-group-per-mosaic rule, and StagedTreeLoader reading a flat
D-K tree whose RGB masks sit in ``labels/files/`` with the class in the red channel."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")

import pyarrow.parquet as pq  # noqa: E402
from _wp6_fixtures import png  # noqa: E402
from conftest import make_source  # noqa: E402

from marinedata.adapters import Decoded, RemoteItem  # noqa: E402
from marinedata.loaders import LoaderError, build_loader  # noqa: E402
from marinedata.release import SUPERVISED_DEFAULT_RATIOS  # noqa: E402
from marinedata.sample_schema import (  # noqa: E402
    LEGACY_FIELD_NAMES,
    V2_FIELDS,
    row_from_mapping,
    to_table,
    validate_table,
    write_samples,
)
from marinedata.splitmap import resolve_splits, rows_to_counts  # noqa: E402
from marinedata.staged_writer import StagedWriter, WriterConfig  # noqa: E402
from marinedata.task_layers.sources import coralseg  # noqa: E402

ITEM = RemoteItem("x", "http://example/x")
MOSAICS = [f"M{i}" for i in range(15)] + ["PALWave37"]


def _write_tree(root: Path, pairs: list[tuple[str, str]], rule=None, mask_r=(0, 1)) -> list:
    """Stage ``(split, upstream_id)`` pairs through the real D-K writer; masks are RGB
    PNGs under ``labels/files/`` with the class in red, like coralseg's live tree."""
    cfg = WriterConfig("fixture", "v1", "CC-BY-4.0", "t", dt.date(2026, 9, 25), split_group=rule)
    w = StagedWriter(root, cfg)
    for i, (split, uid) in enumerate(pairs):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        for band, value in enumerate(mask_r):  # later bands overwrite from row band*4//n
            arr[band * 4 // len(mask_r) :, :, 0] = value
        mask = root.parent / f"m{i}.png"
        Image.fromarray(arr, "RGB").save(mask)
        stem_name = Path(uid).stem
        w.add(
            ITEM,
            Decoded(
                uid,
                png(i, 8),
                ".png",
                split_hint=split,
                label_files={f"{stem_name}.png": mask.read_bytes()},
            ),
        )
    w.finish_item("f" * 64)
    w.finalize()
    write_samples(root / "metadata.parquet", w.rows)
    return w.rows


def test_writer_emits_v2_columns_and_null_without_a_rule(tmp_path: Path) -> None:
    rows = _write_tree(tmp_path / "a", [("test", "test/FR3_0_512_0_512.jpg")], coralseg.SPLIT_GROUP)
    assert (rows[0].split_group, rows[0].upstream_split) == ("coralseg/FR3", "test")
    assert rows[0].upstream_path == "test/FR3_0_512_0_512.jpg"
    bare = _write_tree(tmp_path / "b", [("val", "val/FR3_0_512_0_512.jpg")])
    assert bare[0].split_group is None and bare[0].upstream_split == "val"


def test_v1_parquet_without_the_new_columns_still_loads(tmp_path: Path) -> None:
    root = tmp_path / "t"
    rows = _write_tree(root, [("train", "train/FR3_0_0_0_0.jpg")], coralseg.SPLIT_GROUP)
    pq.write_table(to_table(rows).select(list(LEGACY_FIELD_NAMES)), root / "metadata.parquet")
    table = pq.read_table(root / "metadata.parquet")
    validate_table(table)
    back = row_from_mapping(table.to_pylist()[0])
    assert all(getattr(back, n) is None for n in V2_FIELDS)
    sample = next(iter(build_loader(make_source("staged-tree"), root)))
    assert sample.meta["partition"] == "" and sample.meta["split_group"] is None


def test_flat_dk_tree_masks_fall_back_to_labels_files_with_red_decode(tmp_path: Path) -> None:
    root = tmp_path / "t"
    _write_tree(root, [("train", "train/FR3_0_0_0_0.jpg")], coralseg.SPLIT_GROUP, mask_r=(0, 1, 2))
    assert not (root / "labels/masks").exists()
    params = {"mask_channel": "r", "mask_values": "0=Other,1=Hard Coral,2=Soft Coral"}
    samples = list(build_loader(make_source("staged-tree", params), root))
    assert len(samples) == 1 and samples[0].mask is not None
    assert samples[0].meta["split_group"] == "coralseg/FR3"
    with Image.open(samples[0].mask) as im:
        assert sorted(set(np.asarray(im).ravel().tolist())) == [0, 1, 2]
    assert samples[0].meta["mask_values"] == {"0": "Other", "1": "Hard Coral", "2": "Soft Coral"}
    raw = list(build_loader(make_source("staged-tree"), root))
    assert raw[0].mask is not None and raw[0].mask.parent.name == "files"


def test_undeclared_red_value_raises(tmp_path: Path) -> None:
    root = tmp_path / "t"
    _write_tree(root, [("train", "train/FR3_0_0_0_0.jpg")], mask_r=(0, 7))
    params = {"mask_channel": "r", "mask_values": "0=Other,1=Hard Coral,2=Soft Coral"}
    with pytest.raises(LoaderError, match="R=7"):
        list(build_loader(make_source("staged-tree", params), root))


def test_coralseg_rule_matches_registry(registry) -> None:
    assert registry.source("coralseg-ucsd-mosaics").split_group == coralseg.SPLIT_GROUP


def test_no_mosaic_straddles_release_splits(tmp_path: Path) -> None:
    stems = [
        f"{split}_{m}_{i}_512_{j}_1024"
        for m in MOSAICS
        for split in ("train", "val", "test")
        for i in range(2)
        for j in range(2)
    ]
    rows = [
        (
            f"{n:064x}",
            coralseg.SPLIT_GROUP.resolve(
                source_id=coralseg.SOURCE_ID, stem=s, upstream_path=s, partition=""
            ),
            "coralseg",
        )
        for n, s in enumerate(stems)
    ]
    counts, strata, merge = rows_to_counts(rows, stratified=True, links=[])
    assert len(counts) == 16
    got = resolve_splits(
        tmp_path / "map.json",
        counts,
        dict(SUPERVISED_DEFAULT_RATIOS),
        by="group",
        strata=strata,
        stratify="source",
        merge_canonical=merge.canonical,
    )
    by_mosaic: dict[str, set[str]] = {}
    for (_sha, group, _src), stem in zip(rows, stems, strict=True):
        by_mosaic.setdefault(stem.split("_")[1], set()).add(got[merge.canonical.get(group, group)])
    assert len(by_mosaic) == 16 and all(len(s) == 1 for s in by_mosaic.values())
