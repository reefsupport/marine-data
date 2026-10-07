"""marinedata.labels: native ids -> fixed scheme ids for every published source (offline)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from marinedata import label_schemes as ls
from marinedata import labels

VOCAB = json.loads((Path(__file__).parent / "data" / "label_vocab.json").read_text())
BENTHIC = ("HC", "MIL", "SC", "ALGAE", "ABIOTIC", "OTHER_FAUNA")


def _vocab_entries():
    """(repo/config, source, entry) of every source in the published-vocabulary snapshot."""
    for cfg, srcs in VOCAB.items():
        for source, entry in srcs.items():
            yield cfg, source, entry


def _name(source: str, scheme: str, pixel: int, **opts) -> str | None:
    value = int(labels.lut(source, scheme, **opts)[pixel])
    return None if value == 255 else labels.class_names(scheme)[value]


# ── schemes and ids ───────────────────────────────────────────────────────────────────────────


def test_schemes_and_class_lists():
    assert set(labels.schemes()) == {"benthic-coarse", "coral-binary", "scene"}
    assert labels.class_names("benthic-coarse") == BENTHIC  # registry/tasks/benthic.yaml order
    assert labels.class_names("coral-binary") == ("NOT_CORAL", "CORAL")
    assert labels.class_names("scene") == ("BW", "HD", "PF", "WR", "RO", "RI", "FV", "SR")
    with pytest.raises(ValueError, match="unknown scheme"):
        labels.class_names("nope")


def test_every_lut_is_uint8_256_and_in_range():
    for scheme in labels.schemes():
        n = len(labels.class_names(scheme))
        for source in ls.scheme_spec(scheme).sources:
            if ls.source_spec(source).kind != "semantic":
                continue
            table = labels.lut(source, scheme)
            assert table.shape == (256,) and table.dtype == np.uint8
            assert set(table.tolist()) <= set(range(n)) | {255}
            assert table[255] == 255


def test_lut_is_a_copy_and_unknown_ids_stay_ignored():
    table = labels.lut("coralscapes", "benthic-coarse")
    table[:] = 0
    assert labels.lut("coralscapes", "benthic-coarse")[1] == 255
    assert labels.lut("coralscapes", "benthic-coarse")[200] == 255  # never guessed


# ── pinned entries ────────────────────────────────────────────────────────────────────────────


def test_pinned_benthic_coarse_entries():
    assert labels.lut("coralscapes", "benthic-coarse")[1] == 255  # seagrass: no home in 6 classes
    assert _name("coralscapes", "benthic-coarse", 3) == "HC"  # other coral dead stays HC
    assert _name("coralscapes", "benthic-coarse", 21) == "MIL"
    assert _name("coralscapes", "benthic-coarse", 5) == "ABIOTIC"  # sand
    assert _name("coralscapes", "benthic-coarse", 9) == "OTHER_FAUNA"  # fish
    assert _name("coralscapes", "benthic-coarse", 10) is None  # algae covered substrate (DCA)
    assert _name("coralscapes", "benthic-coarse", 13) is None  # background (water)
    assert labels.lut("reef-support-benthic-own", "benthic-coarse")[1] == 0  # Hard Coral
    assert _name("reef-support-benthic-own", "benthic-coarse", 2) == "SC"
    assert labels.lut("reef-support-seaview-labels", "benthic-coarse")[2] == 2  # SC id is 2
    assert _name("coralseg-ucsd-mosaics", "benthic-coarse", 0) is None  # Other
    assert _name("coralseg-ucsd-mosaics", "benthic-coarse", 2) == "SC"


def test_pinned_coral_binary_entries():
    assert _name("coralscapes", "coral-binary", 22) == "CORAL"  # branching alive
    assert _name("coralscapes", "coral-binary", 21) == "CORAL"  # millepora
    assert _name("coralscapes", "coral-binary", 5) == "NOT_CORAL"  # sand
    assert _name("coralscapes", "coral-binary", 1) == "NOT_CORAL"  # seagrass: benthic
    for pixel in (7, 8, 13, 14, 15):  # human, tools, background, dark, transect line
        assert _name("coralscapes", "coral-binary", pixel) is None
    assert _name("coralseg-ucsd-mosaics", "coral-binary", 0) == "NOT_CORAL"
    assert _name("coralseg-ucsd-mosaics", "coral-binary", 2) == "CORAL"
    assert _name("coralscop-masks-rs", "coral-binary", 1) == "CORAL"
    assert labels.lut("coralscop-masks-rs", "coral-binary")[0] == 255  # not detected != negative
    assert _name("reef-support-seaview-labels", "coral-binary", 2) == "CORAL"


def test_scene_is_suim_identity_on_suim_masks():
    table = labels.lut("suim", "scene")
    assert table[:8].tolist() == list(range(8)) and set(table[8:].tolist()) == {255}
    assert labels.map_label("uiis10k", "corals", "scene") == 5  # RI
    assert labels.map_label("uiis10k", "reptiles", "scene") == 6  # FV
    assert labels.map_label("usis10k", "sea-floor", "scene") == 7
    assert labels.map_label("uiis", "human divers", "scene") == 1
    assert labels.map_label("roboflow-aquarium", "jellyfish", "scene") == 5
    assert labels.map_label("roboflow-aquarium", "shark", "scene") == 6
    assert labels.map_label("uiis10k", "garbage", "scene") is None  # no SUIM class: not guessed


def test_instance_and_box_sources_have_no_pixel_lut():
    with pytest.raises(ValueError, match="map_label"):
        labels.lut("uiis10k", "scene")
    with pytest.raises(ValueError, match="not defined for scheme"):
        labels.lut("suim", "benthic-coarse")
    with pytest.raises(ValueError, match="not in the vocabulary"):
        labels.map_label("uiis10k", "unicorn", "scene")
    with pytest.raises(ValueError, match="unknown source"):
        labels.is_dense("nope")


def test_dense_and_supervised_classes():
    assert [labels.is_dense(s) for s in ("coralscapes", "coralseg-ucsd-mosaics", "suim")] == [
        True
    ] * 3
    for source in ("reef-support-benthic-own", "reef-support-seaview-labels", "coralscop-masks-rs"):
        assert labels.is_dense(source) is False
    assert labels.supervised_classes("reef-support-benthic-own", "benthic-coarse") == ("HC", "SC")
    assert labels.supervised_classes("coralscapes", "benthic-coarse") == (
        "HC", "MIL", "ABIOTIC", "OTHER_FAUNA")  # fmt: skip
    assert labels.supervised_classes("coralscop-masks-rs", "coral-binary") == ("CORAL",)


# ── vocabulary snapshot (tests/data/label_vocab.json, from scripts/dump_label_vocab.py) ────────


def test_every_published_label_is_mapped_or_explicitly_ignored():
    seen = set()
    for _cfg, source, entry in _vocab_entries():
        spec = ls.source_spec(source)
        seen.add(source)
        published = {item["label_native"] for item in entry["labels"]}
        assert published <= set(spec.native_labels), (source, published - set(spec.native_labels))
        for scheme in (s for s in labels.schemes() if source in ls.scheme_spec(s).sources):
            sch = ls.scheme_spec(scheme)
            for label in published:
                if sch.labels is not None:  # by-name scheme: a table entry or an explicit ignore
                    assert label in sch.labels or label in sch.unmapped_labels, (source, label)
                elif label not in sch.source_overrides.get(source, {}):
                    xw = ls._registry().crosswalk(spec.crosswalk)
                    assert xw.edge(label) is not None, (source, label)  # crosswalk decides
                ls.resolve_class(source, label, scheme)  # never raises
    assert seen == set(ls.sources())


def test_pixel_ids_match_the_published_class_maps():
    for _cfg, source, entry in _vocab_entries():
        spec = ls.source_spec(source)
        if spec.kind != "semantic":
            continue
        assert {i["id"]: i["label_native"] for i in entry["labels"]} == dict(spec.ids), source


def test_benthic_coarse_lut_reproduces_the_published_coarse_column():
    for _cfg, source, entry in _vocab_entries():
        if source not in ls.scheme_spec("benthic-coarse").sources:
            continue
        for item in entry["labels"]:
            assert _name(source, "benthic-coarse", item["id"]) == item["coarse"], (source, item)


def test_id_tables_match_the_ingest_code():
    from marinedata.task_layers.masks_table import MASK_SOURCES

    for source, spec in MASK_SOURCES.items():
        assert dict(ls.source_spec(source).ids) == dict(spec.id_to_label), source


# ── options ───────────────────────────────────────────────────────────────────────────────────


def test_exclude_conditions_reads_the_crosswalk_condition_axis():
    opts = {"exclude_conditions": ("dead", "bleached")}
    for scheme in ("benthic-coarse", "coral-binary"):
        for pixel in (
            3,
            4,
            16,
            19,
            20,
            23,
            32,
            33,
            37,
            39,
        ):  # dead / bleached labels (39: dead clam)
            assert _name("coralscapes", scheme, pixel, **opts) is None, (scheme, pixel)
        assert _name("coralscapes", scheme, 22, **opts) is not None  # branching alive stays
        assert _name("coralscapes", scheme, 3) is not None  # default unchanged
    assert _name("coralscapes", "benthic-coarse", 3, exclude_conditions=("bleached",)) == "HC"
    assert _name("coralscapes", "benthic-coarse", 4, exclude_conditions=("UNHEALTHY",)) is None
    with pytest.raises(ValueError, match="unknown condition"):
        labels.lut("coralscapes", "benthic-coarse", exclude_conditions=("deadd",))
    with pytest.raises(TypeError):
        labels.lut("coralscapes", "benthic-coarse", exclude_conditions="dead")


def test_ignore_takes_labels_and_nodes():
    assert _name("coralscapes", "benthic-coarse", 5, ignore=("sand",)) is None
    assert _name("coralscapes", "benthic-coarse", 12, ignore=("sand",)) == "ABIOTIC"
    for pixel in (2, 5, 12, 18):  # node ignore covers the subtree
        assert _name("coralscapes", "benthic-coarse", pixel, ignore=("ABIOTIC",)) is None
    assert labels.map_label("uiis10k", "fish", "scene", ignore=("fish",)) is None


# ── remap_mask / remap_row ────────────────────────────────────────────────────────────────────


def _png(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _row(**kwargs):
    mask = np.array([[1, 3, 5], [255, 13, 21]], dtype=np.uint8)
    class_map = [
        {"id": 1, "label_native": "seagrass", "taxon_node": "SG", "coarse": None},
        {"id": 3, "label_native": "other coral dead", "taxon_node": "HC", "coarse": "HC"},
        {"id": 5, "label_native": "sand", "taxon_node": "SD", "coarse": "ABIOTIC"},
        {"id": 13, "label_native": "background", "taxon_node": "WC", "coarse": None},
        {"id": 21, "label_native": "millepora", "taxon_node": "MIL", "coarse": "MIL"},
    ]
    return {"source": "coralscapes", "mask": mask, "class_map": class_map, **kwargs}


def test_remap_mask_and_row():
    expected = np.array([[255, 0, 4], [255, 255, 1]], dtype=np.uint8)
    out = labels.remap_row(_row(), "benthic-coarse")
    assert out["label"].dtype == np.uint8 and np.array_equal(out["label"], expected)
    assert "label" not in _row()  # input row untouched (new dict returned)
    raw = labels.remap_row(
        _row(mask={"bytes": _png(_row()["mask"]), "path": "m.png"}), "benthic-coarse"
    )
    assert np.array_equal(raw["label"], expected)
    as_pil = labels.remap_row(_row(mask=Image.fromarray(_row()["mask"])), "benthic-coarse")
    assert np.array_equal(as_pil["label"], expected)
    as_json = labels.remap_row(_row(class_map=json.dumps(_row()["class_map"])), "benthic-coarse")
    assert np.array_equal(as_json["label"], expected)
    assert np.array_equal(
        labels.remap_mask(_row()["mask"].astype("int64"), "coralscapes", "benthic-coarse"), expected
    )
    with pytest.raises(ValueError, match=r"outside 0\.\.255"):
        labels.remap_mask(np.array([[300]]), "coralscapes", "benthic-coarse")


def test_remap_row_raises_when_class_map_disagrees():
    bad_label = _row()
    bad_label["class_map"] = [
        {**e, "label_native": "sand"} if e["id"] == 3 else e for e in bad_label["class_map"]
    ]
    with pytest.raises(ValueError, match="class_map says id 3"):
        labels.remap_row(bad_label, "benthic-coarse")
    bad_coarse = _row()
    bad_coarse["class_map"] = [
        {**e, "coarse": "SC"} if e["id"] == 3 else e for e in bad_coarse["class_map"]
    ]
    with pytest.raises(ValueError, match="disagrees with the benthic-coarse LUT"):
        labels.remap_row(bad_coarse, "benthic-coarse")
    missing = _row(mask=np.array([[7]], dtype=np.uint8))
    with pytest.raises(ValueError, match="missing from its class_map"):
        labels.remap_row(missing, "benthic-coarse")
    with pytest.raises(ValueError, match="not defined for scheme"):
        labels.remap_row(_row(), "scene")


# ── torch wiring (default behaviour unchanged) ────────────────────────────────────────────────


def test_torch_dataset_scheme_option(tmp_path):
    torch = pytest.importorskip("torch")
    from marinedata.integrations.torch import to_torch_dataset

    path = tmp_path / "m.png"
    path.write_bytes(_png(np.array([[1, 3], [5, 255]], dtype=np.uint8)))
    sample = SimpleNamespace(
        source_id="coralscapes", key="k", image=None, mask=path, boxes=None, points=None,
        supervised=frozenset(), labels={},
    )  # fmt: skip
    dataset = SimpleNamespace(
        samples=[sample], projector=None,
        label_index=SimpleNamespace(encode=lambda s, projector=None: {}, num_classes=dict),
    )  # fmt: skip
    plain = to_torch_dataset(dataset, decode_images=False, decode_masks=True)[0]["mask"]
    assert plain.tolist() == [[1, 3], [5, 255]]  # raw ids by default
    mapped = to_torch_dataset(
        dataset, decode_images=False, decode_masks=True, scheme="benthic-coarse"
    )
    assert mapped[0]["mask"].tolist() == [[255, 0], [4, 255]]
    assert isinstance(mapped[0]["mask"], torch.Tensor)
    both = to_torch_dataset(
        dataset, decode_images=False, decode_masks=True, scheme="coral-binary",
        scheme_options={"exclude_conditions": ("dead",)},
    )  # fmt: skip
    assert both[0]["mask"].tolist() == [[0, 255], [0, 255]]


# ── packaging and network ─────────────────────────────────────────────────────────────────────


def test_scheme_files_live_under_the_registry_root():
    from marinedata.registry import _default_root

    for name in ("schemes.yaml", "sources.yaml"):
        assert (_default_root() / "label-schemes" / name).is_file()


def _stream(spec: ls.SourceSpec, source: str, n: int = 2) -> list[dict]:
    """First ``n`` train rows of one source (all columns, image bytes included)."""
    from datasets import load_dataset

    files = f"data/{spec.config}/train-{source}-*.parquet"
    stream = load_dataset(spec.repo, data_files={"train": files}, split="train", streaming=True)
    return list(stream.take(n))


def _stream_box_labels(spec: ls.SourceSpec, source: str, n: int = 2) -> list[list[str]]:
    """Box labels of ``n`` rows of ``source`` (column projection: no image bytes are fetched)."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem

    shards = sorted(
        f for f in HfApi().list_repo_files(spec.repo, repo_type="dataset")
        if f.startswith(f"data/{spec.config}/train-")
    )  # fmt: skip
    out: list[list[str]] = []
    for shard in shards:
        with HfFileSystem().open(f"datasets/{spec.repo}/{shard}") as handle:
            table = pq.ParquetFile(handle).read(columns=["source_id", "boxes"]).to_pylist()
        out += [[b["label"] for b in r["boxes"]] for r in table if r["source_id"] == source]
        if len(out) >= n:
            break
    return out[:n]


@pytest.mark.network
def test_streamed_rows_remap_into_scheme_ids(capsys):
    """2 rows per source/config from both HF repos (the private one needs a login).

    Run with ``pytest -m network -s``; skipped by default.
    """
    checked = 0
    for source, spec in ls.sources().items():
        for scheme in (s for s in labels.schemes() if source in ls.scheme_spec(s).sources):
            allowed = set(range(len(labels.class_names(scheme)))) | {255}
            if spec.kind == "semantic":
                for row in _stream(spec, source):
                    out = labels.remap_row(row, scheme)["label"]
                    raw = sorted(set(np.unique(np.asarray(row["mask"])).tolist()))
                    got = sorted(set(np.unique(out).tolist()))
                    with capsys.disabled():
                        print(f"{spec.repo}/{spec.config} {source} {scheme}: raw {raw} -> {got}")
                    assert set(got) <= allowed, (source, scheme)
                    assert out.shape == (row["height"], row["width"]), source
                    checked += 1
            else:
                if spec.kind == "instances":
                    rows = [
                        [i["label_native"] for i in r["instances"]] for r in _stream(spec, source)
                    ]
                else:
                    rows = _stream_box_labels(spec, source)
                assert rows, source
                for names in rows:
                    ids = {labels.map_label(source, name, scheme) for name in names}
                    assert ids <= allowed | {None}, (source, scheme)
                    checked += 1
    assert checked >= 20
