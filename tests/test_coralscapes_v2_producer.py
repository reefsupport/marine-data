"""v1.2.0 WP3: the Coralscapes V2 semantic and instance producers and the resolver.

A synthetic 3-frame staged tree (8x16 PNGs) stands in for ``sources/coralscapes-v2/2.0/``: the
RGB frame, a uint8 semantic mask and a uint16 cityscapes ``_instanceIds`` map per frame, a
``metadata.parquet`` with the upstream split and a ``CHECKSUMS.sha256``.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.annotation_schema import normalise_split, validate_rows
from marinedata.condition_masks import local_tree
from marinedata.metadata_norm.resolvers import RESOLVERS
from marinedata.metadata_norm.resolvers import coralscapes_v2 as res
from marinedata.metadata_norm.resolvers.staged import ResolveContext, StagedImage
from marinedata.registry import Registry
from marinedata.task_layers.masks_table import MASK_SOURCES, staged_mask_rows
from marinedata.task_layers.points_table import resolver_for
from marinedata.task_layers.producers import coralscapes_semseg as v1
from marinedata.task_layers.producers import coralscapes_v2_instances as inst
from marinedata.task_layers.producers import coralscapes_v2_semseg as sem

H, W = 8, 16
FISH = inst.FISH_CLASS_ID
TREE = "coralscapes-v2/2.0"
FRAMES = {  # stem -> upstream split
    "site1_000001_000100": "train",
    "site1_000002_000200": "validation",
    "site22_000003_000300": "test",
}
INSTANCE_DIR = "labels/instances/default"


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _frame_arrays(index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(RGB image, uint8 semantic mask, uint16 instance map) of synthetic frame ``index``."""
    image = np.full((H, W, 3), 40 * (index + 1), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=np.uint8)  # 0 = unlabelled
    if index == 0:  # a fish band of two instances over a band of class 1
        mask[0:4, :], mask[4:6, :] = FISH, 1
    elif index == 1:  # class 95 everywhere with one 2x4 fish
        mask[:, :], mask[0:2, 0:4] = 95, FISH
    else:  # no fish
        mask[2:4, :] = 1
    ids = mask.astype(np.uint16)  # stuff pixels keep their semantic class id
    if index == 0:
        ids[0:4, 0:8] = inst.encode_instance_id(FISH, 1)
        ids[0:4, 8:16] = inst.encode_instance_id(FISH, 2)
    elif index == 1:
        ids[0:2, 0:4] = inst.encode_instance_id(FISH, 1)
    return image, mask, ids


@pytest.fixture(scope="module")
def tree_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("cs-v2-tree")
    for i, stem in enumerate(FRAMES):
        image, mask, ids = _frame_arrays(i)
        files = {
            f"images/default/{stem}_leftImg8bit.png": _png(image),
            f"labels/masks/default/{stem}_leftImg8bit.png": _png(mask),
            f"{INSTANCE_DIR}/{stem}_instanceIds.png": _png(ids),
        }
        for rel, data in files.items():
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_bytes(data)
    meta = [
        {"partition": "default", "stem": f"{s}_leftImg8bit", "upstream_split": sp}
        for s, sp in FRAMES.items()
    ]
    pq.write_table(pa.Table.from_pylist(meta), root / "metadata.parquet")
    lines = [
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root).as_posix()}"
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    return root


@pytest.fixture(scope="module")
def tree(tree_root):
    return local_tree(tree_root, TREE)


@pytest.fixture(scope="module")
def reg():
    return Registry.load()


def _sha(root: Path, stem: str) -> str:
    return hashlib.sha256(
        (root / f"images/default/{stem}_leftImg8bit.png").read_bytes()
    ).hexdigest()


def test_instance_id_round_trip_and_uint16_bounds():
    assert inst.LABEL_DIVISOR == 2000 and FISH == 25 and sem.ID_TO_LABEL[FISH] == "fish"
    assert inst.encode_instance_id(FISH, 1) == 50001
    for class_id, index in [(25, 0), (25, 1999), (1, 1), (32, 1535)]:
        value = inst.encode_instance_id(class_id, index)
        assert value <= 0xFFFF and inst.decode_instance_id(value) == (class_id, index)
    for class_id, index in [(25, 2000), (33, 0), (32, 1536), (0, 5), (25, -1)]:
        with pytest.raises(ValueError, match="uint16"):
            inst.encode_instance_id(class_id, index)
    assert inst.decode_instance_id(25) == (0, 25) and not inst.is_thing(25)  # a stuff pixel
    assert inst.is_thing(50001) and not inst.is_thing(1999)


def test_decode_instance_png_counts_only_thing_pixels(tree_root):
    two = (tree_root / INSTANCE_DIR / "site1_000001_000100_instanceIds.png").read_bytes()
    assert inst.decode_instance_png(two) == ({50001: 32, 50002: 32}, H, W)
    none = (tree_root / INSTANCE_DIR / "site22_000003_000300_instanceIds.png").read_bytes()
    assert inst.decode_instance_png(none) == ({}, H, W)
    top = np.full((2, 2), 65535, dtype=np.uint16)  # uint16 passes through, nothing truncated
    assert inst.decode_instance_png(_png(top)) == ({65535: 4}, 2, 2)


def test_mask_class_counts_drops_unlabelled(tree_root):
    mask = tree_root / "labels/masks/default/site1_000001_000100_leftImg8bit.png"
    assert sem.IGNORE_INDEX == 0
    assert sem.mask_class_counts(mask) == {"fish": 64, "acanthaster planci": 32}


def test_semantic_rows_one_per_frame(tree, tree_root, reg):
    rows = list(staged_mask_rows(tree, sem.mask_source(), reg))
    assert len(rows) == len(FRAMES)
    assert {r["image_sha256"] for r in rows} == {_sha(tree_root, s) for s in FRAMES}
    assert {(r["source_id"], r["source_version"]) for r in rows} == {("coralscapes-v2", "2.0")}
    assert all(r["ignore_value"] == 0 for r in rows)
    by_sha = {r["image_sha256"]: r for r in rows}
    for stem, split in FRAMES.items():
        assert by_sha[_sha(tree_root, stem)]["upstream_split"] == normalise_split(split)
    validate_rows("masks", rows)


def test_produce_semseg_pairs_every_frame(tree_root, tmp_path):
    out = tmp_path / "semseg.parquet"
    assert sem.produce_semseg(tree_root, out) == len(FRAMES)
    rows = {r["sha256"]: r for r in pq.read_table(out).to_pylist()}
    assert set(rows) == {_sha(tree_root, s) for s in FRAMES}
    assert {r["source_id"] for r in rows.values()} == {"coralscapes-v2"}
    first = rows[_sha(tree_root, "site1_000001_000100")]
    assert json.loads(first["class_counts"]) == {"fish": 64, "acanthaster planci": 32}


def test_instance_rows_one_per_thing_id(tree, tree_root, reg):
    rows = list(inst.staged_instance_rows(tree, reg))
    sha0, sha1 = (_sha(tree_root, s) for s in list(FRAMES)[:2])
    keys = [(r["image_sha256"], r["instance_id"]) for r in rows]
    assert keys == [(sha0, 50001), (sha0, 50002), (sha1, 50001)]  # frame 2 has no fish
    assert [r["ann_id"] for r in rows] == [f"coralscapes-v2:{i}" for i in range(3)]
    assert all(r["mask_kind"] == "instance" and r["class_map"] is None for r in rows)
    assert {(r["label_native"], r["label_native_id"]) for r in rows} == {("fish", "25")}
    assert (
        rows[0]["mask_ref"] == f"sources/{TREE}/{INSTANCE_DIR}/site1_000001_000100_instanceIds.png"
    )
    attrs = [json.loads(r["attrs"]) for r in rows]
    assert [(a["class_id"], a["instance_index"], a["area"]) for a in attrs] == [
        (25, 1, 32),
        (25, 2, 32),
        (25, 1, 8),
    ]
    assert {r["upstream_split"] for r in rows[:2]} == {normalise_split("train")}
    assert rows[2]["upstream_split"] == normalise_split("validation")
    validate_rows("masks", rows)


def test_masks_layer_is_deterministic_and_limited(tree, reg):
    rows = list(inst.staged_masks_rows(tree, reg))
    assert len(rows) == 3 + 3 and len({r["ann_id"] for r in rows}) == 6  # one key space
    assert [r["mask_kind"] for r in rows[3:]] == ["instance"] * 3
    assert rows == list(inst.staged_masks_rows(tree, reg))
    validate_rows("masks", rows)
    assert len(list(inst.staged_instance_rows(tree, reg, limit=1))) == 2  # frame 0 only
    assert len(list(staged_mask_rows(tree, sem.mask_source(), reg, limit=2))) == 2


def test_every_class_resolves_into_the_taxonomy(reg):
    resolve = resolver_for(reg, sem.CROSSWALK_ID)  # the registry crosswalk, not a local table
    schema = reg.label_schema("rs-benthic-v1")
    taxa = {resolve(native)[0] for native in sem.ID_TO_LABEL.values()} - {None}
    assert taxa and all(schema.node(t) is not None for t in taxa)
    assert sorted(sem.ID_TO_LABEL) == list(range(1, 96))


def _staged(stem: str) -> StagedImage:
    return StagedImage("default", stem, f"images/default/{stem}.png", "a" * 64, {})


def test_resolver_site_from_stem_and_partition_kept():
    ctx = ResolveContext(source_id="coralscapes-v2", version="2.0", get=lambda key: b"")
    stems = [
        "site1_000001_000100_leftImg8bit",
        "site22_000003_000300_leftImg8bit",
        "site7_000009_000900",
        "oddname_leftImg8bit",
    ]
    out = res.resolve([_staged(s) for s in stems], ctx)
    assert [r.site_id for r in out.rows] == ["site1", "site22", "site7", None]
    assert {r.partition for r in out.rows} == {"default"}
    assert {r.source_id for r in out.rows} == {"coralscapes-v2"}
    assert [(u.stem, u.field) for u in out.unresolved] == [("oddname_leftImg8bit", "site")]


def test_v1_producer_and_resolver_stay_registered():
    assert v1.SOURCE_ID == "coralscapes" and sem.SOURCE_ID == "coralscapes-v2"
    assert "coralscapes" in MASK_SOURCES and "coralscapes" in RESOLVERS


def test_semseg_and_instance_configs_split_the_shared_masks_parquet_by_mask_kind(
    tree, reg, tmp_path
):
    """REV F2: V2 semantic and fish-instance rows share one masks parquet; no row is in both."""
    from marinedata.task_layers.configs.masks import build_instance_config, build_semseg_config
    from marinedata.task_layers.masks_table import write_masks

    rows = list(inst.staged_masks_rows(tree, reg))
    n_instance = sum(r["mask_kind"] == "instance" for r in rows)
    assert 0 < n_instance < len(rows)  # a mixed table
    write_masks(tmp_path, sem.SOURCE_ID, sem.SOURCE_VERSION, rows)
    assert len(build_semseg_config(reg, tmp_path).rows) == len(rows) - n_instance
    assert len(build_instance_config(reg, tmp_path).rows) == n_instance


def test_masks_table_entry_equals_the_producer_spec():
    assert MASK_SOURCES[sem.SOURCE_ID] == sem.mask_source()
