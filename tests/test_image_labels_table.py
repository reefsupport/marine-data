"""WP-U5: bleaching and condition labels into the unified ``image_labels`` table (offline, stubs)."""  # noqa: E501

from __future__ import annotations

import io
import json

import pytest

from marinedata.annotation_schema import validate_row, validate_rows
from marinedata.registry import Registry, _default_root
from marinedata.task_layers import image_labels_table as it
from marinedata.task_layers.configs import BLEACHING_SOURCES, build_bleaching_config

SHA = "ab" * 32


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load(_default_root())


class StubTree:
    """The slice of ``StagedTree`` the readers use: ``table``, ``shas``, ``masks``, ``key``."""

    def __init__(self, tables, shas, masks=(), blobs=None):
        self.tables, self.shas, self._masks, self.blobs = tables, shas, list(masks), blobs or {}

    def table(self, rel):
        return self.tables[rel]

    def masks(self):
        return list(self._masks)

    def get_or_skip(self, rel):
        return self.blobs.get(rel)

    def key(self, rel):
        return f"sources/stub/{rel}"


def _rows(reg, sid, tree, **kw):
    spec = it.LABEL_SOURCES[sid]
    rows = list(it.staged_rows(tree, spec, reg, **kw))
    validate_rows("image_labels", rows)
    return rows


def _labels_tree(labels, n_images=None):
    stems = sorted({r["stem"] for r in labels})
    return StubTree(
        {"labels/image_labels.parquet": labels},
        {("default", s): f"{i:064x}" for i, s in enumerate(stems)},
    )


def test_nine_sources_every_row_carries_a_licence_class(reg):
    assert len(it.LABEL_SOURCES) == 9
    assert set(BLEACHING_SOURCES) <= set(it.LABEL_SOURCES)
    registry_class = {"PERMISSIVE": "permissive"}
    for sid, spec in it.LABEL_SOURCES.items():
        if sid in BLEACHING_SOURCES:  # in neither lic TSV: the registry tier decides
            tier = reg.source(sid).licence.tier.name
            assert registry_class[tier] == spec.licence_class, sid
    assert it.LABEL_SOURCES["kaggle-healthy-bleached-corals"].licence_class == "internal-only"
    assert it.LABEL_SOURCES["noaa-pifsc-esa-coral-icra"].licence_class == "open"


def test_roboflow_rows_resolve_condition_and_keep_non_corals_as_a_class_row(reg):
    labels = [
        {"stem": "a", "partition": "default", "label": "Healthy", "schema_id": "s", "confidence": None},  # noqa: E501
        {"stem": "a", "partition": "default", "label": "Non-Corals", "schema_id": "s", "confidence": None},  # noqa: E501
        {"stem": "b", "partition": "default", "label": "Bleached", "schema_id": "s", "confidence": None},  # noqa: E501
        {"stem": "c", "partition": "default", "label": "Made Up", "schema_id": "s", "confidence": None},  # noqa: E501
    ]  # fmt: skip
    rows = _rows(reg, "roboflow-coral-classification-copy-changed-v13i", _labels_tree(labels))
    by = {r["label_native"]: r for r in rows}
    assert (by["Healthy"]["condition_node_id"], by["Healthy"]["match_type"]) == (
        "HEALTHY",
        "broader",
    )
    assert (by["Bleached"]["condition_node_id"], by["Bleached"]["match_type"]) == (
        "BLEACHED",
        "exact",
    )
    assert by["Healthy"]["label_key"] == by["Bleached"]["label_key"] == "condition"
    nc = by["Non-Corals"]
    assert (nc["taxon_node_id"], nc["condition_node_id"], nc["label_key"]) == (
        "UNKNOWN",
        None,
        "class",
    )
    bad = by["Made Up"]  # no edge: explicit unmapped, never guessed
    assert bad["match_type"] == "unmapped"
    assert bad["taxon_node_id"] is bad["condition_node_id"] is None
    assert {r["annotator_type"] for r in rows} == {"crowd"}
    assert [r["ann_id"] for r in rows] == [f"{rows[0]['source_id']}:{i}" for i in range(4)]
    assert all(json.loads(r["attrs"])["licence_class"] == "permissive" for r in rows)


def test_limit_takes_evenly_spaced_images_not_the_first_block(reg):
    labels = [
        {"stem": f"{'b' if i < 8 else 'h'}{i:02d}", "partition": "default",
         "label": "CORAL_BL" if i < 8 else "CORAL", "schema_id": "s", "confidence": None}
        for i in range(10)
    ]  # fmt: skip
    rows = _rows(reg, "noaa-pifsc-bleaching", _labels_tree(labels), limit=5)
    assert len({r["image_sha256"] for r in rows}) == 5
    assert {r["condition_node_id"] for r in rows} == {"BLEACHED", "HEALTHY"}


def test_kaggle_kv_rows_are_internal_only_and_dropped_by_the_release_filter(reg):
    labels = [
        {"stem": "x", "key": "label", "value": "bleached_corals"},
        {"stem": "y", "key": "label", "value": "healthy_corals"},
    ]
    rows = _rows(reg, "kaggle-healthy-bleached-corals", _labels_tree(labels))
    assert [(r["condition_node_id"], r["match_type"]) for r in rows] == [
        ("BLEACHED", "exact"), ("HEALTHY", "broader")]  # fmt: skip
    assert all(it.is_internal_only(r) and r["ann_license"] is None for r in rows)
    assert it.drop_internal_only(rows) == []
    assert it.drop_internal_only([{"attrs": None}, {"attrs": '{"licence_class": "open"}'}]) != []


def _png(values):
    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.array(values, dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


def test_mask_presence_rows_one_per_class_with_pixels(reg):
    rel = "labels/masks/SITE/img1.png"
    tree = StubTree(
        {}, {("SITE", "img1"): SHA}, masks=[(("SITE", "img1"), rel)],
        blobs={rel: _png([[0, 1, 1], [2, 2, 2]])},
    )  # fmt: skip
    rows = _rows(reg, "reef-support-bleaching", tree)
    got = {
        r["label_native"]: (r["condition_node_id"], json.loads(r["attrs"])["pixel_count"])
        for r in rows
    }
    assert got == {"bleached": ("BLEACHED", 2), "non_bleached": ("HEALTHY", 3)}
    assert {r["annotator_type"] for r in rows} == {"expert"}


def test_yolo_presence_rows_resolve_the_species_to_the_genus_node(reg):
    assert it.parse_yolo_classes("0 .1 .2 .3 .4\n\n0 .5 .5 .1 .1\n") == {0: 2}
    assert (
        it.yolo_label_path("dataset/images/test/a__IMG_1.JPG") == "dataset/labels/test/a__IMG_1.txt"
    )
    meta = [
        {"stem": s, "upstream_id": f"dataset/images/val/{s}.JPG", "split_hint": "val"}
        for s in "abc"
    ]
    tree = StubTree(
        {"metadata.parquet": meta}, {("default", s): f"{i:064x}" for i, s in enumerate("abc")}
    )
    files = {
        "dataset/labels/val/a.txt": b"0 .1 .1 .2 .2\n0 .5 .5 .1 .1\n",
        "dataset/labels/val/b.txt": b"\n",
    }
    rows = _rows(reg, "noaa-pifsc-esa-coral-icra", tree, fetch=files.get)  # c: missing, b: empty
    assert len(rows) == 1
    row = rows[0]
    assert (row["label_native"], row["label_native_id"], row["label_key"]) == (
        "ICRA",
        "0",
        "species",
    )
    assert (row["taxon_node_id"], row["taxon_rank"], row["match_type"]) == (
        "HC_ISOPORA",
        "Genus",
        "broader",
    )
    assert row["worms_aphia_id"] == 730685 and row["upstream_split"] == "val"
    assert json.loads(row["attrs"])["n_boxes"] == 2 and row["ann_license"] == "US-GOV-PD"


def test_bleaching_config_reads_unified_rows_and_keeps_the_legacy_columns(reg, tmp_path):
    spec = it.LABEL_SOURCES["noaa-pifsc-bleaching"]
    labels = [
        {"stem": "a", "partition": "default", "label": "CORAL_BL", "schema_id": "d", "confidence": None},  # noqa: E501
        {"stem": "b", "partition": "default", "label": "CORAL", "schema_id": "d", "confidence": None},  # noqa: E501
    ]  # fmt: skip
    rows = _rows(reg, spec.source_id, _labels_tree(labels))
    it.write_image_labels(tmp_path, spec.source_id, spec.version, rows)
    result = build_bleaching_config(reg, tmp_path)
    got = {r["native_label"]: r for r in result.rows}
    assert set(got) == {"CORAL_BL", "CORAL"}
    assert (got["CORAL_BL"]["canonical_condition"], got["CORAL_BL"]["coral_health"]) == (
        "BLEACHED",
        "UNHEALTHY",
    )
    assert (got["CORAL"]["canonical_condition"], got["CORAL"]["coral_health"]) == (
        "HEALTHY",
        "HEALTHY",
    )
    assert (
        got["CORAL"]["sha256"] == got["CORAL"]["image_sha256"]
        and got["CORAL"]["licence_class"] == "permissive"
    )
    assert result.unmapped_by_source[spec.source_id] == 0.0


def test_a_mapped_row_without_a_node_fails_validation(reg):
    row = _rows(
        reg,
        "noaa-pifsc-bleaching",
        _labels_tree(
            [
                {
                    "stem": "a",
                    "partition": "default",
                    "label": "CORAL",
                    "schema_id": "d",
                    "confidence": None,
                }
            ]
        ),
    )[0]
    assert validate_row("image_labels", {**row, "condition_node_id": None})  # needs a node
