"""WP-U4: coral and benthic masks into the unified ``masks`` table (offline, synthetic bytes)."""

from __future__ import annotations

import io
import json

import pytest

from marinedata.annotation_schema import validate_row
from marinedata.registry import Registry, _default_root
from marinedata.task_layers import masks_table as mt
from marinedata.task_layers.configs import SEMSEG_SOURCES, build_semseg_config
from marinedata.task_layers.producers.coralscapes_semseg import _ID_TO_LABEL
from marinedata.task_layers.s3_keyed import StagedTree

SHA = "a" * 64


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load(_default_root())


def _png(values: list[list[int]], rgb: bool = False) -> bytes:
    import numpy as np
    from PIL import Image

    arr = np.array(values, dtype=np.uint8)
    im = Image.fromarray(np.stack([arr, arr * 0, arr * 0], -1) if rgb else arr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _row(reg, sid, counts):
    spec = mt.MASK_SOURCES[sid]
    return mt.semantic_row(
        spec=spec, resolve=mt.resolver_for(reg, spec.crosswalk_id), ordinal=0,
        image_sha256=SHA, mask_ref=f"sources/{spec.tree}/labels/masks/x.png", counts=counts,
    )  # fmt: skip


def test_five_sources_are_registered_and_wired_into_the_semseg_config():
    assert set(mt.MASK_SOURCES) == {
        "coralscapes", "coralseg-ucsd-mosaics", "coralscop-masks-rs",
        "reef-support-benthic-own", "suim",
    }  # fmt: skip
    assert set(SEMSEG_SOURCES) == set(mt.MASK_SOURCES)


def test_coralseg_row_maps_exact_and_keeps_other_explicitly_unmapped(reg):
    row = _row(reg, "coralseg-ucsd-mosaics", {0: 70, 1: 20, 2: 10})
    assert validate_row("masks", row) == []
    assert json.loads(row["class_map"]) == {"0": "Other", "1": "Hard Coral", "2": "Soft Coral"}
    attrs = json.loads(row["attrs"])
    assert attrs["class_resolution"]["Other"]["match_type"] == "unmapped"
    assert attrs["class_resolution"]["Hard Coral"]["taxon_node_id"] == "HC"
    assert json.loads(row["canonical_class_counts"]) == {"HC": 20, "SC": 10}
    assert (attrs["mapped_pixels"], attrs["labelled_pixels"]) == (30, 100)
    assert row["match_type"] == "exact" and row["taxon_node_id"] is None


def test_unknown_index_is_unmapped_never_guessed(reg):
    row = _row(reg, "reef-support-benthic-own", {0: 5, 1: 3, 9: 2})
    attrs = json.loads(row["attrs"])
    assert json.loads(row["class_map"])["9"] == f"{mt.UNKNOWN_PREFIX}9"
    assert attrs["class_resolution"][f"{mt.UNKNOWN_PREFIX}9"]["match_type"] == "unmapped"
    assert "0" not in json.loads(row["class_map"])  # the ignore value is not a class
    assert validate_row("masks", row) == []


def test_all_unknown_or_empty_row_is_unmapped(reg):
    assert _row(reg, "suim", {42: 9})["match_type"] == "unmapped"
    assert _row(reg, "coralscop-masks-rs", {0: 100})["match_type"] == "unmapped"


@pytest.mark.parametrize("sid", sorted(mt.MASK_SOURCES))
def test_every_declared_class_resolves_through_its_crosswalk(reg, sid):
    spec = mt.MASK_SOURCES[sid]
    resolve = mt.resolver_for(reg, spec.crosswalk_id)
    mapped = [lbl for lbl in spec.id_to_label.values() if resolve(lbl)[1] != "unmapped"]
    assert mapped, sid
    if sid != "coralseg-ucsd-mosaics":
        assert len(mapped) == len(spec.id_to_label)
    assert len(_ID_TO_LABEL) == 39


def test_coralscop_rows_are_pseudo_internal_only_and_droppable(reg):
    row = _row(reg, "coralscop-masks-rs", {0: 60, 1: 40})
    assert (row["annotator_type"], row["ann_license"]) == ("pseudo", "CC-BY-NC-SA-4.0")
    assert json.loads(row["attrs"])["licence_class"] == mt.INTERNAL_ONLY
    own = _row(reg, "reef-support-benthic-own", {1: 1})
    assert mt.is_internal_only(row) and not mt.is_internal_only(own)
    assert mt.drop_internal_only([row, own]) == [own]


def test_staged_mask_rows_join_sha_through_checksums(reg):
    files = {
        "sources/coralseg-ucsd-mosaics/unversioned/CHECKSUMS.sha256": (
            f"{SHA}  images/test_A_1.jpg\n{'b' * 64}  labels/files/test_A_1.png\n"
        ).encode(),
        "sources/coralseg-ucsd-mosaics/unversioned/labels/files/test_A_1.png": _png(
            [[0, 1], [2, 1]], rgb=True
        ),
    }
    tree = StagedTree("coralseg-ucsd-mosaics/unversioned", fetch=lambda k, **_: files[k])
    spec = mt.MASK_SOURCES["coralseg-ucsd-mosaics"]
    (row,) = list(mt.staged_mask_rows(tree, spec, reg, workers=1))
    assert row["image_sha256"] == SHA and row["ann_id"] == "coralseg-ucsd-mosaics:0"
    assert json.loads(row["class_counts"]) == {"Other": 1, "Hard Coral": 2, "Soft Coral": 1}
    assert row["mask_ref"].endswith("labels/files/test_A_1.png")
    assert validate_row("masks", row) == []


def test_seaview_labels_pending_rows_follow_the_staged_tree():
    base = mt.SEAVIEW_LABELS_PREFIX
    assert base == "sources/reef-support-seaview-labels/2026-10-06/"
    keys = [
        base + "images/SEAVIEW_ATL/a (1).JPG",
        base + "labels/instance_masks/SEAVIEW_ATL/masks/a (1)_mask_0.png",
        base + "labels/instance_masks/SEAVIEW_ATL/masks/a (1)_mask_1.png",
        base + "labels/masks/SEAVIEW_ATL/a (1).png",  # stitched mask: not an instance row
    ]
    rows = mt.pending_rows(keys, images=mt.image_index(keys))
    assert len(rows) == 2
    assert {r["image_key"] for r in rows} == {base + "images/SEAVIEW_ATL/a (1).JPG"}
    assert all(r["source_id"] == "reef-support-seaview-labels" for r in rows)
    assert mt.validate_pending(rows) == []
    assert all(
        r["match_type"] == "unmapped" and r["label_native"] == mt.PENDING_LABEL for r in rows
    )


def test_semseg_config_reads_unified_rows_and_keeps_the_filter_columns(reg, tmp_path):
    for sid in ("coralscop-masks-rs", "reef-support-benthic-own"):
        mt.write_masks(tmp_path, sid, mt.MASK_SOURCES[sid].version, [_row(reg, sid, {0: 3, 1: 7})])
    result = build_semseg_config(reg, tmp_path)
    by_src = {r["source_id"]: r for r in result.rows}
    assert set(by_src) == {"coralscop-masks-rs", "reef-support-benthic-own"}
    assert by_src["coralscop-masks-rs"]["licence_class"] == mt.INTERNAL_ONLY
    assert by_src["coralscop-masks-rs"]["annotator_type"] == "pseudo"
    assert json.loads(by_src["coralscop-masks-rs"]["canonical_class_counts"]) == {"CNIDARIA": 7}
    assert json.loads(by_src["reef-support-benthic-own"]["canonical_class_counts"]) == {"HC": 7}
    assert by_src["reef-support-benthic-own"]["sha256"] == SHA
