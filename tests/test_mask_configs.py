"""MV-1: self-contained HF mask configs (image + mask + class_map in one row)."""

from __future__ import annotations

import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from marinedata import hf_card, hf_export
from marinedata.hf_export import SampleRow, build_layout, export
from marinedata.registry import Registry
from marinedata.task_layers import configs
from marinedata.task_layers import mask_encode as me
from marinedata.task_layers.mask_classmap import hf_class_map
from marinedata.task_layers.mask_configs import ImageInfo, mask_layout_entries
from marinedata.task_layers.masks_table import MASK_SOURCES, semantic_row, write_masks
from marinedata.task_layers.points_table import resolver_for

W, H = 24, 16


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _decode(data: bytes):
    return np.asarray(Image.open(io.BytesIO(data)))


# ---- encodings -> index PNG ---------------------------------------------------------------------


def test_indexed_png_remaps_the_source_ignore_value_to_255():
    arr = np.zeros((H, W), np.uint8)
    arr[2:6, 2:6], arr[8:12, 8:12] = 3, 39
    out = _decode(me.semantic_png(_png(arr), encoding="png-indexed", ignore_value=0))
    assert out.dtype == np.uint8 and out.shape == (H, W)
    assert set(np.unique(out)) == {3, 39, 255} and (out == 3).sum() == 16
    assert (out[arr == 0] == 255).all()


def test_palette_png_keeps_pixel_values_when_the_source_has_no_ignore_value():
    arr = np.zeros((H, W), np.uint8)
    arr[:, 12:] = 2  # coralseg: 0 Other, 1 Hard Coral, 2 Soft Coral
    out = _decode(me.semantic_png(_png(arr), encoding="png-indexed", ignore_value=None))
    assert (out == arr).all()  # 0 is a real class here, not unlabelled


def test_rgb_red_channel_encoding_ignores_the_other_channels():
    rgb = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    rgb[..., 0] = (np.arange(H * W).reshape(H, W) % 3).astype(np.uint8)
    out = _decode(me.semantic_png(_png(rgb), encoding="png-rgb-red", ignore_value=None))
    assert (out == rgb[..., 0]).all() and out.ndim == 2


def test_suim_rgb_palette_decodes_to_the_8_class_index_and_grey_indexes_pass_through():
    table = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (0, 255, 255),
             (255, 0, 0), (255, 0, 255), (255, 255, 0), (255, 255, 255)]  # fmt: skip
    rgb = np.array([table], np.uint8).repeat(4, axis=1)  # 1 x 32 x 3
    out = _decode(me.semantic_png(_png(rgb), encoding="rgb-palette", ignore_value=None))
    assert out[0].tolist() == [i for i in range(8) for _ in range(4)]
    grey = np.full((2, 2, 3), 5, np.uint8)  # already an index replicated over RGB
    assert (me.decode_index(_png(grey), "rgb-palette") == 5).all()
    assert me.encoding_for("suim", "png-indexed") == "rgb-palette"


def test_a_native_255_that_is_not_the_ignore_value_raises():
    arr = np.full((2, 2), 255, np.uint8)
    arr[0, 0] = 1
    with pytest.raises(me.MaskEncodingError):
        me.semantic_png(_png(arr), encoding="png-indexed", ignore_value=0)


def test_per_class_binary_pngs_compose_into_one_index_png():
    a = np.zeros((H, W), np.uint8)
    b = np.zeros((H, W), np.uint8)
    a[:8], b[4:12] = 255, 255
    out = _decode(me.compose_parts([(_png(a), 1), (_png(b), 2)], (W, H)))
    assert out[0, 0] == 1 and out[6, 0] == 2 and out[14, 0] == 255 and out.dtype == np.uint8


# ---- instances -> uint16 id PNG -----------------------------------------------------------------


def test_rle_decoding_matches_pycocotools_for_compressed_and_list_counts():
    mask_utils = pytest.importorskip("pycocotools.mask")
    arr = np.zeros((H, W), np.uint8)
    arr[3:9, 5:15] = 1
    rle = mask_utils.encode(np.asfortranarray(arr))
    assert (me.decode_rle({"size": rle["size"], "counts": rle["counts"]}) == arr).all()
    runs = me.rle_counts(rle["counts"])
    assert (me.decode_rle({"size": [H, W], "counts": runs}) == arr).all()


def test_instance_png_paints_ids_polygon_rle_and_per_instance_png():
    rle = {"size": [H, W], "counts": [10 * H, 3 * H, W * H - 13 * H]}  # columns 10..12
    rows = [
        {"polygon": json.dumps([[0, 0, 8, 0, 8, 8, 0, 8]]), "attrs": json.dumps({"area": 64})},
        {"rle": json.dumps(rle), "attrs": json.dumps({"area": 48})},
        {"mask_ref": "k/inst.png", "attrs": "{}"},
    ]
    stamp = np.zeros((H, W), np.uint8)
    stamp[12:16, 18:24] = 255
    out = _decode(
        me.instance_png(rows, [1, 2, 3], canvas=(W, H), size=(W, H), fetch=lambda k: _png(stamp))
    )
    assert out.dtype == np.uint16 and out.shape == (H, W)
    assert out[4, 4] == 1 and out[5, 11] == 2 and out[14, 20] == 3 and out[15, 0] == 0
    assert set(np.unique(out)) == {0, 1, 2, 3}


# ---- class_map ----------------------------------------------------------------------------------


def test_hf_class_map_resolves_taxon_and_coarse_and_blanks_unmapped():
    rec = {
        "class_map": json.dumps({"3": "Hard Coral", "9": "__unknown_index_9"}),
        "class_resolution": {
            "Hard Coral": {"taxon_node_id": "n:hc", "match_type": "exact"},
            "__unknown_index_9": {"taxon_node_id": None, "match_type": "unmapped"},
        },
    }
    got = json.loads(hf_class_map(rec, lambda n: "HC" if n == "n:hc" else None))
    assert got == [
        {"id": 3, "label_native": "Hard Coral", "taxon_node": "n:hc", "coarse": "HC"},
        {"id": 9, "label_native": "__unknown_index_9", "taxon_node": None, "coarse": None},
    ]
    assert hf_class_map({"class_map": None}, lambda n: None) is None


def test_semseg_config_carries_the_export_class_map(tmp_path):
    reg = Registry.load()
    spec = MASK_SOURCES["coralscapes"]
    row = semantic_row(
        spec=spec, resolve=resolver_for(reg, spec.crosswalk_id), ordinal=0,
        image_sha256="a" * 64, mask_ref="sources/coralscapes/1.0/labels/masks/x.png",
        counts={0: 5, 1: 10},
    )  # fmt: skip
    write_masks(tmp_path, "coralscapes", spec.version, [row])
    out = configs.build_semseg_config(reg, tmp_path).rows
    assert len(out) == 1 and out[0]["ignore_value"] == 0
    class_map = json.loads(out[0]["hf_class_map"])
    assert [c["id"] for c in class_map] == [1] and class_map[0]["label_native"]
    assert out[0]["mask_encoding"] == "png-indexed"
    assert configs.MASK_CONFIG_BY_SOURCE["coralscapes"] == "coral-masks"
    assert set(configs.MASK_CONFIG_BY_SOURCE.values()) == set(configs.MASK_EXPORT_CONFIG_IDS)


# ---- layout + datasets round trip ----------------------------------------------------------


def _image(tmp_path: Path, name: str) -> Path:
    path = tmp_path / "img" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(_png(np.full((H, W, 3), 120, np.uint8)))
    return path


def _sem(sha, source, key, encoding, ignore, class_map, annotator="human"):
    return {
        "sha256": sha, "image_sha256": sha, "source_id": source, "mask_kind": "semantic",
        "mask_key": key, "hf_class_map": json.dumps(class_map), "ignore_value": ignore,
        "mask_encoding": encoding, "mask_parts": None, "annotator_type": annotator,
        "ann_license": "CC-BY-4.0", "ann_id": f"{source}:0",
    }  # fmt: skip


def _inst(sha, n, source="uiis"):
    box = [2 + 6 * n, 2, 4, 4]
    ring = [box[0], 2, box[0] + 4, 2, box[0] + 4, 6, box[0], 6]
    return {
        "image_sha256": sha, "source_id": source, "mask_kind": "instance",
        "ann_id": f"{source}:{n}", "instance_id": 100 - n, "label_native": "fish",
        "taxon_node_id": "n:fish", "match_type": "exact", "coarse": None,
        "annotator_type": "human", "ann_license": "Apache-2.0",
        "polygon": json.dumps([ring]), "rle": None, "mask_ref": None,
        "attrs": json.dumps({"bbox_xywh": box, "img_w": W, "img_h": H, "area": 16}),
    }  # fmt: skip


@pytest.fixture
def tree(tmp_path):
    """3 rows per config: coralscapes (indexed), suim (RGB palette), uiis (instances) and one
    machine mask, as the staged-key -> bytes map plus the task-layer rows and release images."""
    store: dict[str, bytes] = {}
    layers: dict[str, list[dict]] = {"semseg": [], "instances": []}
    images: dict[str, ImageInfo] = {}
    cmap = [{"id": 1, "label_native": "HC", "taxon_node": "n:hc", "coarse": "HC"}]
    for i in range(3):
        sha = f"{i:x}" * 64
        images[sha] = ImageInfo(_image(tmp_path, f"cs{i}"), "train", f"site{i}", "Apache-2.0")
        store[f"k/cs{i}.png"] = _png(np.full((H, W), 1 + i, np.uint8))
        layers["semseg"].append(_sem(sha, "coralscapes", f"k/cs{i}.png", "png-indexed", 0, cmap))
    for i in range(3):
        sha = f"{i + 3:x}" * 64
        images[sha] = ImageInfo(_image(tmp_path, f"su{i}"), "validation", None, "MIT")
        store[f"k/su{i}.png"] = _png(np.full((H, W, 3), (0, 255, 255), np.uint8))
        layers["semseg"].append(_sem(sha, "suim", f"k/su{i}.png", "png-indexed", None, cmap))
    for i in range(3):
        sha = f"{i + 6:x}" * 64
        images[sha] = ImageInfo(_image(tmp_path, f"ii{i}"), "train", None, None)
        layers["instances"] += [_inst(sha, 0), _inst(sha, 1)]
    sha = "9" * 64
    images[sha] = ImageInfo(_image(tmp_path, "cop"), "train", None, None)
    store["k/cop.png"] = _png(np.array([[0, 1]], np.uint8).repeat(H, 0).repeat(W // 2, 1))
    layers["semseg"].append(
        _sem(sha, "coralscop-masks-rs", "k/cop.png", "png-indexed", 0, cmap, "pseudo")
    )
    return layers, images, store


def test_layout_has_one_self_contained_config_per_target_and_normalised_masks(tree):
    layers, images, store = tree
    stats = Counter()
    entries = mask_layout_entries(layers, images, fetch=store.__getitem__, stats=stats)
    assert set(entries) == {"coral-masks", "scene-masks", "instance-masks", "coral-masks-machine"}
    spec, splits = entries["coral-masks"]
    assert [c for c, _ in spec.columns][:4] == ["image_sha256", "image", "mask", "class_map"]
    assert spec.image_columns == ("image", "mask") and list(splits) == ["train"]
    row = splits["train"][0]
    assert row.values["width"] == W and row.values["annotator_type"] == "human"
    assert row.values["class_map"][0]["label_native"] == "HC" and row.values["lat"] is None
    machine = entries["coral-masks-machine"][1]["train"][0]
    assert machine.values["annotator_type"] == "machine"
    mask = _decode(machine.blobs[1].read())
    assert set(np.unique(mask)) == {1, 255}  # coralscop: background 0 -> ignore
    inst = entries["instance-masks"][1]["train"][0]
    assert [i["id"] for i in inst.values["instances"]] == [
        1,
        2,
    ]  # upstream id order 99, 100 -> 1, 2
    assert inst.values["instances"][0]["bbox_xyxy_norm"] == pytest.approx(
        [8 / W, 2 / H, 12 / W, 6 / H]
    )
    assert _decode(inst.blobs[1].read()).dtype == np.uint16
    assert not stats


def test_a_row_whose_image_is_not_in_the_release_is_dropped_and_counted(tree):
    layers, images, store = tree
    images.pop("0" * 64)
    stats = Counter()
    entries = mask_layout_entries(layers, images, fetch=store.__getitem__, stats=stats)
    assert len(entries["coral-masks"][1]["train"]) == 2
    assert stats["coral-masks:dropped_no_image"] == 1


def test_build_layout_export_and_load_dataset_round_trip(tree, tmp_path):
    datasets = pytest.importorskip("datasets")
    layers, images, store = tree
    rows = {
        "task": [
            SampleRow("task", "val" if i.split == "validation" else "train", sha, "coralscapes",
                      f"s{n}", i.path, i.split_group)
            for n, (sha, i) in enumerate(images.items())
        ]
    }  # fmt: skip
    layout = build_layout(rows, task_layers=layers, flavour=None, mask_fetch=store.__getitem__)
    assert {"coral-masks", "scene-masks", "instance-masks"} <= set(layout)
    out = tmp_path / "hub"
    summary = export(layout, out)
    configs_yaml = "\n".join(hf_card._yaml_configs(summary))
    assert (
        "config_name: coral-masks" in configs_yaml and "config_name: instance-masks" in configs_yaml
    )
    (out / "README.md").write_text(f"---\n{configs_yaml}\n---\n")
    expected = {"coral-masks": ("train", np.uint8), "scene-masks": ("validation", np.uint8),
                "instance-masks": ("train", np.uint16)}  # fmt: skip
    for config, (split, dtype) in expected.items():
        ds = datasets.load_dataset(str(out), config, split=split)
        assert len(ds) == 3
        row = ds[0]
        assert np.asarray(row["image"]).shape == (H, W, 3)
        mask = np.asarray(row["mask"])
        assert mask.shape == (H, W) and mask.dtype == dtype
        assert row["width"] == W and isinstance(row["image_sha256"], str)
    ds = datasets.load_dataset(str(out), "instance-masks", split="train")
    assert ds[0]["instances"][0]["label_native"] == "fish" and len(ds[0]["instances"]) == 2
    cm = datasets.load_dataset(str(out), "coral-masks", split="train")[0]["class_map"]
    assert cm[0]["id"] == 1 and cm[0]["coarse"] == "HC"
    assert hf_export.IMAGES in layout  # the v1 images config is untouched
