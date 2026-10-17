"""PR-2: a v1.1.0 image keeps a frozen v1.0 ``bytes_config`` only when the v1.0 tree holds its sha.

``bytes_config`` used to key on the source alone. A new image of a v1.0 source (benthic-own +24,
seaview-labels +11, aquarium +1 in run 3) was stamped ``coral-masks`` / ``fish-boxes``, configs with
no row for it, and the export shipped no bytes for it (G1 join: 36 unresolved).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.checksums import file_digest
from marinedata.registry import Registry
from marinedata.task_layers.configs.common import ConfigResult
from marinedata.task_layers.v11_tables import (
    MissingV1TreeError,
    V11Inputs,
    build_v11_tables,
    bytes_config_of,
    inputs_from_build,
    load_v1_config_shas,
    source_bytes_config,
)

OWN, AQUARIUM, UIIS = "reef-support-benthic-own", "roboflow-aquarium", "uiis"
CORAL, OPT = "coralscapes", "reefolution"
OLD, NEW, OTHER, FISH, BOTH = (f"{i:064x}" for i in range(1, 6))
SPLIT_MAP = SimpleNamespace(assignments={}, withheld=())


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _v1_tree(root: Path, configs: dict[str, list[str]]) -> Path:
    """A tiny frozen v1.0 tree: ``data/<config>/train-00000-of-00001.parquet`` per config."""
    for config, shas in configs.items():
        path = root / "data" / config / "train-00000-of-00001.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.table({"image_sha256": shas, "image": [b"v1"] * len(shas)}), path)
    return root


@pytest.mark.parametrize(
    ("source_id", "sha", "want"),
    [
        (OWN, OLD, "coral-masks"),  # a v1.0 image keeps the frozen config
        (OWN, NEW, "images"),  # a new image of a v1.0 source goes to images
        (AQUARIUM, FISH, "fish-boxes"),
        (AQUARIUM, NEW, "images"),
        ("noaa-pifsc-bleaching", NEW, "images"),  # not a v1.0 source
        ("noaa-pifsc-bleaching", OLD, "coral-masks"),  # shared with a v1.0 image: bytes are there
        (UIIS, BOTH, "instance-masks"),  # grandfathered double embed: the mask config wins
        (UIIS, FISH, "fish-boxes"),  # only the other config holds it
        (OWN, None, "images"),
        (None, NEW, "images"),
    ],
)
def test_bytes_config_follows_the_v1_tree_not_the_source(source_id, sha, want) -> None:
    v1 = {"coral-masks": {OLD}, "fish-boxes": {FISH, BOTH}, "instance-masks": {BOTH}}
    assert bytes_config_of(source_id, sha, v1) == want
    assert bytes_config_of(source_id, sha, {}) == "images"  # no tree: nothing is a v1.0 image


def test_sources_are_ordered_by_the_config_the_v1_tree_gives_them() -> None:
    v1 = {"coral-masks": {OLD}, "fish-boxes": set()}
    assert source_bytes_config(OWN, v1) == "coral-masks"
    assert source_bytes_config(AQUARIUM, v1) == "images"  # fish-boxes holds nothing in this tree
    assert source_bytes_config("noaa-pifsc-bleaching", v1) == "images"


def _stage(tmp_path: Path, staged: dict[str, list[tuple[str, str]]]) -> dict[str, Path]:
    roots = {}
    for source_id, items in staged.items():
        root = tmp_path / "roots" / source_id
        root.mkdir(parents=True)
        rows = [{"image_sha256": s, "split_group": g, "license": "CC BY 4.0"} for s, g in items]
        pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
        roots[source_id] = root
    return roots


def test_metadata_and_task_rows_take_the_config_per_image(registry, tmp_path) -> None:
    roots = _stage(tmp_path, {OPT: [(OLD, "g1"), (OTHER, "g3")], CORAL: [(OLD, "g1"), (NEW, "g2")]})
    split_of = {OLD: "train", NEW: "val", OTHER: "test"}
    inputs = V11Inputs(
        taxonomy_version="3.0.0", roots=roots, split_of=split_of,
        v1_config_shas={"coral-masks": {OLD}},
    )  # fmt: skip
    mask = {
        "source_id": CORAL, "label_origin": "human", "class_counts": "{}", "label_status": "ok",
        "canonical_class_counts": "{}", "hf_class_map": "[]",
    }  # fmt: skip
    semseg = ConfigResult(
        "semseg", tuple({**mask, "sha256": s, "mask_key": f"m/{s[-1]}.png"} for s in (OLD, NEW)), {}
    )
    out = build_v11_tables(registry, {"semseg": semseg}, inputs)
    meta = {r["image_sha256"]: r for r in out["metadata"].rows}
    assert {s: r["bytes_config"] for s, r in meta.items()} == {
        OLD: "coral-masks", NEW: "images", OTHER: "images",
    }  # fmt: skip
    assert meta[OLD]["source_id"] == CORAL  # the v1.0 source describes a shared image, as before
    by_source = out["metadata"].notes["report"]["metadata_by_source"]
    assert (by_source[CORAL]["rows"], by_source[OPT]["shared"]) == (2, 1)
    assert {r["image_sha256"]: r["bytes_config"] for r in out["semseg"].rows} == {
        OLD: "coral-masks", NEW: "images",
    }  # fmt: skip
    old = V11Inputs(taxonomy_version="3.0.0", roots=roots, split_of=split_of)  # no v1.0 tree
    rows = build_v11_tables(registry, {}, old)["metadata"].rows
    assert {r["bytes_config"] for r in rows} == {"images"}  # never a v1.0 config unasked


def test_a_wide_tree_is_stamped_the_same_way(registry, tmp_path) -> None:
    root, rows, shas = tmp_path / OWN, [], {}
    for stem in ("w1", "w2"):
        path = root / "images" / "SEAFLOWER_BOLIVAR" / f"{stem}.JPG"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stem.encode())
        shas[stem] = file_digest(path)
        rows.append(
            {
                "sample_id": f"{OWN}/{stem}", "source_id": OWN, "source_version": "2026-10-06",
                "stem": stem, "image_path": f"images/SEAFLOWER_BOLIVAR/{stem}.JPG",
                "image_sha256": shas[stem], "license": "CC-BY-4.0",
                "partition": "SEAFLOWER_BOLIVAR", "split_group": f"{OWN}/SEAFLOWER_BOLIVAR",
            }
        )  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
    inputs = V11Inputs(
        taxonomy_version="3.0.0", roots={OWN: root}, split_of=dict.fromkeys(shas.values(), "train"),
        v1_config_shas={"coral-masks": {shas["w1"]}},
    )  # fmt: skip
    meta = {r["image_sha256"]: r for r in build_v11_tables(registry, {}, inputs)["metadata"].rows}
    assert meta[shas["w1"]]["bytes_config"] == "coral-masks"  # in the frozen v1.0 tree
    assert meta[shas["w2"]]["bytes_config"] == "images"  # a new image of the same v1.0 source


def test_the_v1_tree_reader_keeps_the_pixel_configs_only(tmp_path) -> None:
    v1 = _v1_tree(tmp_path / "v1", {"coral-masks": [OLD], "fish-boxes": [FISH], "metadata": [NEW]})
    found = load_v1_config_shas(v1, [OWN])
    assert found == {"coral-masks": frozenset({OLD}), "fish-boxes": frozenset({FISH})}
    inputs = inputs_from_build(
        taxonomy_version="3.0.0", roots={OWN: tmp_path}, manifests=[], split_map=SPLIT_MAP,
        v1_config_shas=found,
    )  # fmt: skip
    assert inputs.v1_config_shas == found


def test_a_v1_source_without_a_v1_tree_fails_naming_the_flag(tmp_path) -> None:
    with pytest.raises(MissingV1TreeError, match=rf"{OWN}.*--v1-dir DIR"):
        load_v1_config_shas(None, ["noaa-pifsc-bleaching", OWN])
    with pytest.raises(MissingV1TreeError, match="--v1-dir DIR"):
        inputs_from_build(
            taxonomy_version="3.0.0", roots={AQUARIUM: tmp_path}, manifests=[], split_map=SPLIT_MAP
        )
    with pytest.raises(MissingV1TreeError, match=rf"--v1-dir {tmp_path / 'nope'}"):
        load_v1_config_shas(tmp_path / "nope", [OWN])  # a wrong path is not a tree either


def test_no_v1_source_needs_no_v1_tree(tmp_path) -> None:
    assert load_v1_config_shas(None, ["noaa-pifsc-bleaching", OPT]) == {}
    inputs = inputs_from_build(
        taxonomy_version="3.0.0", roots={OPT: tmp_path}, manifests=[], split_map=SPLIT_MAP
    )  # fmt: skip
    assert dict(inputs.v1_config_shas) == {}
