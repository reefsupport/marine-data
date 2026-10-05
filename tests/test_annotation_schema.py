"""The unified annotation schema (WP-U1): tables, validators, enums, parquet round-trip."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("pyarrow")

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata import annotation_schema as an
from marinedata import sample_schema as ss

SHA = "a" * 64
SHA2 = "b" * 64

COLUMN_COUNTS = {
    "boxes": 33,
    "masks": 30,
    "points": 26,
    "image_labels": 23,
    "tracks": 36,
    "captions": 14,
    "vqa": 15,
    "depth": 15,
    "pairs": 13,
    "identities": 35,
}
TAXON_FREE = {"captions", "vqa", "depth", "pairs"}
COMMON_TAXON = [
    "image_sha256",
    "source_id",
    "source_version",
    "ann_id",
    "label_native",
    "label_native_id",
    "label_set",
    "taxon_node_id",
    "form_node_id",
    "condition_node_id",
    "taxon_rank",
    "worms_aphia_id",
    "rs_benthic_code",
    "match_type",
    "annotator_type",
    "annotator_detail",
    "ann_license",
    "ann_attribution",
    "confidence",
    "upstream_split",
    "label_status",
    "attrs",
]
COMMON_FREE = [
    "image_sha256",
    "source_id",
    "source_version",
    "ann_id",
    "annotator_type",
    "annotator_detail",
    "ann_license",
    "ann_attribution",
    "confidence",
    "upstream_split",
    "label_status",
]

_LABELLED = dict(
    label_native="Acropora sp.",
    label_native_id="12",
    label_set="coralnet-native",
    taxon_node_id="taxon:acropora",
    taxon_rank="genus",
    worms_aphia_id=206988,
    rs_benthic_code="HC",
    match_type="exact",
    attrs='{"form": "branching"}',
)
_PROV = dict(
    annotator_type="expert",
    annotator_detail=None,
    ann_license="CC-BY-4.0",
    ann_attribution="Someone et al.",
    confidence=0.75,
    upstream_split="train",
    label_status="ok",
)
_BOX = dict(
    x_min=0.25,
    y_min=0.25,
    x_max=0.75,
    y_max=0.5,
    x_min_px=25,
    y_min_px=25,
    x_max_px=75,
    y_max_px=50,
    img_w=100,
    img_h=100,
)
_PAYLOAD = {
    "boxes": dict(**_BOX, is_crowd=False),
    "masks": dict(
        mask_kind="semantic",
        mask_ref="sources/s/v1/labels/masks/default/a.png",
        class_map='{"0": "background", "1": "Acropora sp."}',
        ignore_value=255,
        class_counts='{"1": 12}',
        canonical_class_counts='{"taxon:acropora": 12}',
    ),
    "points": dict(x=0.5, y=0.25, x_px=50, y_px=25),
    "image_labels": dict(label_key="class"),
    "tracks": dict(video_id="vid-1", frame_idx=7, track_id="t3", **_BOX),
    "captions": dict(text="A reef.", lang="en", caption_type="caption"),
    "vqa": dict(question="How many?", answer="Two.", qa_type="count", lang="en-GB"),
    "depth": dict(depth_ref="sources/s/v1/depth/a.npy", units="m", gt_type="sensor"),
    "pairs": dict(pair_role="enhanced", ref_image_sha256=SHA2),
    "identities": dict(individual_id="turtle-9", individual_scope="dataset", **_BOX),
}


def make_row(table: str, ordinal: int = 0, **kw) -> dict:
    """A valid row for ``table`` (every column present), overridden by ``kw``."""
    base = {
        "image_sha256": SHA,
        "source_id": "src",
        "source_version": "v1",
        "ann_id": f"src:{ordinal}",
    }
    if table in TAXON_FREE:
        base.update(_PROV)
    else:
        base.update(_LABELLED)
        base.update(_PROV)
    base.update(_PAYLOAD[table])
    base.update(kw)
    spec = an.table_spec(table)
    return {n: base.get(n) for n in spec.names} | {
        k: v for k, v in kw.items() if k not in spec.names
    }


def errors(table: str, **kw) -> list[str]:
    return an.validate_row(table, make_row(table, **kw))


# --- schemas ------------------------------------------------------------------------------


def test_ten_tables_nine_tasks_plus_identities():
    assert set(an.TABLES) == set(COLUMN_COUNTS)
    assert len(an.TABLES) == 10


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_column_count_and_common_prefix(table):
    spec = an.table_spec(table)
    assert len(spec.names) == COLUMN_COUNTS[table]
    assert len(set(spec.names)) == len(spec.names)
    common = COMMON_FREE if table in TAXON_FREE else COMMON_TAXON
    assert list(spec.names[: len(common)]) == common
    assert spec.has_taxon is (table not in TAXON_FREE)


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_arrow_schema_matches_spec(table):
    spec = an.table_spec(table)
    schema = an.arrow_schema(table)
    assert schema.names == list(spec.names)
    for col in spec.columns:
        field = schema.field(col.name)
        assert field.nullable is col.nullable
    assert schema.metadata[b"marinedata.annotation_schema"] == b"1"
    assert schema.metadata[b"marinedata.annotation_table"] == table.encode()


def test_key_and_common_column_types():
    s = an.arrow_schema("boxes")
    assert s.field("worms_aphia_id").type == pa.int64()
    assert s.field("confidence").type == pa.float32()
    for name in ("x_min", "y_min", "x_max", "y_max"):
        assert s.field(name).type == pa.float32()
    for name in ("x_min_px", "y_max_px", "img_w", "img_h"):
        assert s.field(name).type == pa.int32()
    assert s.field("is_crowd").type == pa.bool_()
    assert an.arrow_schema("points").field("x").type == pa.float32()
    assert an.arrow_schema("tracks").field("frame_idx").type == pa.int32()
    assert an.arrow_schema("masks").field("ignore_value").type == pa.int32()


def test_required_columns():
    for table in COLUMN_COUNTS:
        s = an.arrow_schema(table)
        for name in ("source_id", "source_version", "ann_id", "annotator_type", "label_status"):
            assert not s.field(name).nullable
    for table in set(COLUMN_COUNTS) - {"tracks"}:
        assert not an.arrow_schema(table).field("image_sha256").nullable
    assert an.arrow_schema("tracks").field("image_sha256").nullable
    for table in set(COLUMN_COUNTS) - TAXON_FREE:
        s = an.arrow_schema(table)
        assert not s.field("match_type").nullable
        assert not s.field("label_set").nullable
        assert s.field("label_native").nullable
    for table in TAXON_FREE:
        names = an.table_spec(table).names
        assert not {"match_type", "taxon_node_id", "label_native"} & set(names)


def test_manager_columns_are_nullable_strings():
    for table in COLUMN_COUNTS:
        s = an.arrow_schema(table)
        for name in ("ann_license", "ann_attribution", "annotator_detail"):
            assert s.field(name).nullable and s.field(name).type == pa.string()


def test_join_key_and_unknown_table():
    assert an.JOIN_KEY == ("image_sha256", "source_id", "source_version", "ann_id")
    with pytest.raises(an.AnnotationSchemaError):
        an.arrow_schema("nope")
    with pytest.raises(an.AnnotationSchemaError):
        an.table_spec("nope")


def test_annotation_path(tmp_path):
    p = an.annotation_path(tmp_path, "boxes", "fathomnet", "v2")
    assert p == tmp_path / "_annotations" / "boxes" / "fathomnet" / "v2.parquet"
    with pytest.raises(an.AnnotationSchemaError):
        an.annotation_path(tmp_path, "nope", "s", "v")


# --- enums --------------------------------------------------------------------------------


def test_enum_values():
    assert {"exact", "broader", "narrower", "related", "unmapped"} == an.MATCH_TYPES
    assert {"expert", "human", "crowd", "model", "pseudo"} == an.ANNOTATOR_TYPES
    assert an.MatchType("narrower") is an.MatchType.NARROWER
    assert an.AnnotatorType.PSEUDO.value == "pseudo"
    assert an.MatchType.EXACT == "exact"


@pytest.mark.parametrize(
    ("origin", "expected"),
    [
        ("human_expert", ("expert", None)),
        ("human_crowd", ("crowd", None)),
        ("human", ("human", None)),
        ("pseudo_model", ("pseudo", None)),
        ("model", ("model", None)),
        ("derived_rule", ("pseudo", "rule")),
    ],
)
def test_annotator_from_origin(origin, expected):
    assert an.annotator_from_origin(origin) == expected
    assert expected[0] in an.ANNOTATOR_TYPES


def test_annotator_from_origin_rejects_unknown():
    with pytest.raises(an.AnnotationSchemaError):
        an.annotator_from_origin("unknown")


def test_normalise_split_is_the_sample_schema_one():
    assert an.normalise_split is ss.normalise_split
    assert an.normalise_split("Validation") == "val"


# --- validators: every table accepts its factory row --------------------------------------


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_factory_row_is_valid(table):
    assert errors(table) == []


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_unknown_column_and_missing_required(table):
    assert any("unknown column" in e for e in errors(table, bogus=1))
    row = make_row(table)
    del row["ann_id"]
    assert "ann_id is required" in an.validate_row(table, row)


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_common_value_checks(table):
    assert errors(table, image_sha256="ABC")
    assert errors(table, ann_id="other:1")
    assert errors(table, ann_id="src:x")
    assert errors(table, annotator_type="robot")
    assert errors(table, label_status="great")
    assert errors(table, confidence=1.5)
    assert errors(table, confidence=-0.1)
    assert errors(table, upstream_split="validation")
    assert errors(table, ann_license="not a licence!")
    assert errors(table, source_id="")
    assert errors(table, confidence="high")
    assert errors(table, confidence=float("nan"))
    assert errors(table, confidence=True)


def test_nullable_provenance_and_spdx_expressions():
    for table in COLUMN_COUNTS:
        ok = errors(
            table,
            confidence=None,
            upstream_split=None,
            ann_license="CC-BY-4.0 AND CC0-1.0",
            ann_attribution=None,
            annotator_detail="rule",
        )
        assert ok == []
        assert errors(table, ann_license=None) == []
        assert errors(table, ann_license="LicenseRef-CoralSCOP-1.0") == []


def test_confidence_tolerance():
    assert errors("points", confidence=1.0 + 5e-7) == []
    assert errors("points", confidence=1.0 + 5e-6)


def test_ann_id_ordinal_and_source_with_hyphen():
    assert errors("points", source_id="mermaid-aws", ann_id="mermaid-aws:464024") == []
    assert errors("points", source_id="mermaid-aws", ann_id="mermaid-aws:") != []


# --- taxon rules ---------------------------------------------------------------------------


def test_unmapped_needs_null_nodes():
    unmapped = dict(
        match_type="unmapped",
        taxon_node_id=None,
        form_node_id=None,
        condition_node_id=None,
        taxon_rank=None,
        worms_aphia_id=None,
        rs_benthic_code=None,
    )
    assert errors("image_labels", **unmapped) == []
    assert errors("image_labels", **{**unmapped, "taxon_node_id": "taxon:x"})
    assert errors("image_labels", **{**unmapped, "condition_node_id": "cond:bleached"})
    assert errors("image_labels", **{**unmapped, "rs_benthic_code": "HC"})


def test_mapped_needs_a_node_but_any_axis_will_do():
    none = dict(taxon_node_id=None, taxon_rank=None, worms_aphia_id=None, rs_benthic_code=None)
    assert errors("image_labels", **none)
    assert errors("image_labels", **none, condition_node_id="cond:bleached") == []
    assert errors("image_labels", **none, form_node_id="form:massive") == []


def test_derived_columns_need_a_taxon_node():
    assert errors("points", taxon_node_id=None, condition_node_id="cond:x")
    assert errors("points", worms_aphia_id=0)
    assert errors("points", worms_aphia_id=-5)


def test_match_type_values_and_narrower_is_stored():
    for mt in an.MATCH_TYPES - {"unmapped"}:
        assert errors("points", match_type=mt) == []
    assert errors("points", match_type="coarsened")
    assert errors("points", match_type=None)


def test_label_native_required_except_semantic_masks_and_identities():
    assert errors("points", label_native=None)
    assert errors("boxes", label_native=None)
    assert errors("masks", label_native=None) == []  # semantic factory row
    inst = dict(mask_kind="instance", class_map=None, rle="abc")
    assert errors("masks", label_native=None, **inst)
    assert errors("masks", **inst) == []
    assert errors("identities", label_native=None) == []


def test_attrs_must_be_a_json_object():
    assert errors("points", attrs="[1]")
    assert errors("points", attrs="{nope")
    assert errors("points", attrs=None) == []


# --- boxes ---------------------------------------------------------------------------------


def test_box_order_and_range():
    assert errors("boxes", x_min=0.8)  # x_min > x_max
    assert errors("boxes", x_min=0.75)  # x_min == x_max
    assert errors("boxes", y_max=0.25)
    assert errors("boxes", x_max=1.2, x_max_px=120)
    assert errors("boxes", x_min=-0.1, x_min_px=-10)


def test_box_tolerance_edges():
    edge = dict(x_min=-5e-7, x_min_px=0, x_max=1.0 + 5e-7, x_max_px=100)
    assert errors("boxes", **edge) == []
    assert errors("boxes", x_min=-5e-6)
    assert errors("boxes", x_max=1.0 + 5e-6)


def test_box_pixel_consistency():
    assert errors("boxes", x_min_px=40)  # 0.40 vs 0.25
    assert errors("boxes", x_max_px=75, img_w=200)  # px disagrees with img size
    assert errors("boxes", x_min_px=None)  # all four or none
    assert errors("boxes", img_w=None)  # both dims or neither
    assert errors("boxes", img_w=0, img_h=0)
    nopx = dict(x_min_px=None, y_min_px=None, x_max_px=None, y_max_px=None, img_w=None, img_h=None)
    assert errors("boxes", **nopx) == []
    assert errors("boxes", x_min_px=75, x_max_px=75, x_min=0.75)


def test_box_requires_is_crowd_and_normalised_coordinates():
    assert errors("boxes", is_crowd=None)
    assert errors("boxes", x_min=None)
    assert errors("boxes", is_crowd="no")


# --- masks ---------------------------------------------------------------------------------


def test_semantic_mask_rules():
    assert errors("masks", mask_ref=None)
    assert errors("masks", rle="x")
    assert errors("masks", polygon="[[0,0]]")
    assert errors("masks", class_map=None)
    assert errors("masks", class_map="[1]")
    assert errors("masks", class_map='{"a": "b"}')
    assert errors("masks", class_map='{"1": 5}')
    assert errors("masks", class_counts="[]")
    assert errors("masks", mask_kind="panoptic")
    assert errors("masks", ignore_value=None) == []


def test_semantic_mask_needs_no_label_or_node():
    bare = dict(
        label_native=None,
        taxon_node_id=None,
        taxon_rank=None,
        worms_aphia_id=None,
        rs_benthic_code=None,
    )
    assert errors("masks", **bare) == []


def test_instance_mask_rules():
    inst = dict(mask_kind="instance", class_map=None, mask_ref=None, rle="5 3 2")
    assert errors("masks", **inst) == []
    assert errors("masks", **{**inst, "rle": None})  # no geometry at all
    assert errors("masks", **{**inst, "rle": None, "polygon": "[[0.1, 0.2]]"}) == []
    assert errors("masks", **{**inst, "rle": None, "polygon": "{}"})
    assert errors("masks", **{**inst, "rle": None, "mask_ref": "k.png"}) == []
    assert errors("masks", **{**inst, "class_map": '{"1": "x"}'})


# --- points, image_labels, tracks, identities ----------------------------------------------


def test_points_rules():
    assert errors("points", x=1.5, x_px=150)
    assert errors("points", y=-0.01)
    assert errors("points", x_px=None)  # both px or none
    assert errors("points", x_px=None, y_px=None) == []
    assert errors("points", x_px=-1)
    assert errors("points", x=None)
    assert errors("points", x=1.0 + 5e-7) == []


def test_image_labels_need_a_key():
    assert errors("image_labels", label_key=None)
    assert errors("image_labels", label_key=" ")
    assert errors("image_labels", label_key="condition") == []


def test_tracks_rules():
    assert errors("tracks", image_sha256=None) == []  # no frame staged
    assert errors("tracks", frame_idx=-1)
    assert errors("tracks", track_id=None)
    assert errors("tracks", video_id=None)
    assert errors("tracks", x_min=0.9)
    assert errors("tracks", is_crowd=None) == []
    assert errors("tracks", image_sha256="short")


def test_identities_rules():
    assert errors("identities", individual_scope="planet")
    assert errors("identities", individual_id=None)
    no_box = dict(
        x_min=None,
        y_min=None,
        x_max=None,
        y_max=None,
        x_min_px=None,
        y_min_px=None,
        x_max_px=None,
        y_max_px=None,
        img_w=None,
        img_h=None,
    )
    assert errors("identities", **no_box) == []
    assert errors("identities", **{**no_box, "x_min": 0.1})  # half a box
    assert errors("identities", **{**no_box, "x_min_px": 4})  # px without a box
    assert errors("identities", x_min=0.9)  # inverted


# --- taxon-free tables ---------------------------------------------------------------------


def test_captions_and_vqa_rules():
    assert errors("captions", caption_type="essay")
    assert errors("captions", lang="english")
    assert errors("captions", lang="pt-BR") == []
    assert errors("captions", text="  ")
    assert errors("vqa", lang="x")
    assert errors("vqa", answer=None)
    assert errors("vqa", qa_type=None)


def test_depth_and_pairs_rules():
    assert errors("depth", units="feet")
    assert errors("depth", gt_type="guess")
    assert errors("depth", valid_mask_ref="k.png") == []
    assert errors("depth", depth_ref=None)
    assert errors("pairs", pair_role="twin")
    assert errors("pairs", ref_image_sha256="nothex")
    assert errors("pairs", ref_image_sha256=None)


def test_taxon_free_tables_reject_taxon_columns():
    for table in TAXON_FREE:
        assert any("unknown column" in e for e in errors(table, taxon_node_id="taxon:x"))
        assert any("unknown column" in e for e in errors(table, match_type="exact"))


# --- tables --------------------------------------------------------------------------------


def test_validate_rows_flags_duplicates_and_collects():
    rows = [make_row("points", 1), make_row("points", 1)]
    with pytest.raises(an.AnnotationSchemaError, match="duplicate ann_id"):
        an.validate_rows("points", rows)
    with pytest.raises(an.AnnotationSchemaError, match="x_px"):
        an.validate_rows("points", [make_row("points", 1, x_px=None)])
    an.validate_rows("points", [make_row("points", 1), make_row("points", 2)])


def test_same_ann_id_in_another_version_is_not_a_duplicate():
    rows = [make_row("points", 1), make_row("points", 1, source_version="v2")]
    an.validate_rows("points", rows)


def test_to_table_clips_tolerated_coordinates_and_sorts_by_ordinal():
    rows = [
        make_row("points", 10, x=1.0 + 5e-7, y=-5e-7, confidence=1.0 + 5e-7),
        make_row("points", 2),
    ]
    t = an.to_table("points", rows)
    assert t["ann_id"].to_pylist() == ["src:2", "src:10"]
    assert t["x"].to_pylist()[1] == 1.0
    assert t["y"].to_pylist()[1] == 0.0
    assert t["confidence"].to_pylist()[1] == 1.0
    assert t.schema.equals(an.arrow_schema("points"))


def test_to_table_rejects_invalid_rows():
    with pytest.raises(an.AnnotationSchemaError):
        an.to_table("boxes", [make_row("boxes", x_min=0.9)])


@pytest.mark.parametrize("table", sorted(COLUMN_COUNTS))
def test_parquet_round_trip(table, tmp_path):
    rows = [make_row(table, 1), make_row(table, 0, image_sha256=SHA2)]
    path = an.annotation_path(tmp_path, table, "src", "v1")
    an.write_annotations(path, table, rows)
    assert path.is_file()
    back = an.read_annotations(path, table)
    assert back.schema.names == list(an.table_spec(table).names)
    got = back.to_pylist()
    assert [r["ann_id"] for r in got] == ["src:0", "src:1"]
    want = {r["ann_id"]: r for r in rows}
    for r in got:
        for name, value in want[r["ann_id"]].items():
            if isinstance(value, float):
                assert r[name] == pytest.approx(value)
            else:
                assert r[name] == value, name
    assert pq.read_schema(path).metadata[b"marinedata.annotation_table"] == table.encode()


def test_round_trip_keeps_nulls_and_json_strings(tmp_path):
    row = make_row(
        "masks",
        0,
        label_native=None,
        taxon_node_id=None,
        taxon_rank=None,
        worms_aphia_id=None,
        rs_benthic_code=None,
        confidence=None,
        ann_license=None,
        ignore_value=None,
    )
    path = tmp_path / "m.parquet"
    an.write_annotations(path, "masks", [row])
    got = an.read_annotations(path, "masks").to_pylist()[0]
    assert got["label_native"] is None and got["confidence"] is None
    assert json.loads(got["class_map"]) == {"0": "background", "1": "Acropora sp."}


def test_track_without_frame_round_trips(tmp_path):
    row = make_row("tracks", 0, image_sha256=None)
    path = tmp_path / "t.parquet"
    an.write_annotations(path, "tracks", [row])
    assert an.read_annotations(path, "tracks").to_pylist()[0]["image_sha256"] is None


def test_parquet_carries_column_statistics(tmp_path):
    path = tmp_path / "b.parquet"
    an.write_annotations(path, "boxes", [make_row("boxes", 0), make_row("boxes", 1)])
    meta = pq.ParquetFile(path).metadata
    cols = {meta.schema.column(i).name: i for i in range(meta.num_columns)}
    stats = meta.row_group(0).column(cols["x_min"]).statistics
    assert stats is not None and stats.has_min_max
    assert meta.row_group(0).column(cols["ann_license"]).statistics.null_count == 0


def test_write_is_deterministic(tmp_path):
    rows = [make_row("points", 3), make_row("points", 1), make_row("points", 2)]
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    an.write_annotations(a, "points", rows)
    an.write_annotations(b, "points", list(reversed(rows)))
    assert a.read_bytes() == b.read_bytes()


def test_validate_table_rejects_wrong_shape(tmp_path):
    good = an.to_table("points", [make_row("points", 0)])
    an.validate_table("points", good)
    with pytest.raises(an.AnnotationSchemaError, match="columns differ"):
        an.validate_table("points", good.drop_columns(["attrs"]))
    with pytest.raises(an.AnnotationSchemaError, match="columns differ"):
        an.validate_table("boxes", good)
    bad_type = good.set_column(
        good.schema.get_field_index("x"), pa.field("x", pa.float64()), good["x"].cast(pa.float64())
    )
    with pytest.raises(an.AnnotationSchemaError, match="type"):
        an.validate_table("points", bad_type)


def test_validate_table_checks_rows():
    t = an.to_table("points", [make_row("points", 0)])
    bad = t.set_column(
        t.schema.get_field_index("x"), t.schema.field("x"), pa.array([2.0], type=pa.float32())
    )
    with pytest.raises(an.AnnotationSchemaError, match="outside"):
        an.validate_table("points", bad)
