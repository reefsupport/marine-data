"""WP-U3: benthic points in the unified ``points`` table (``task_layers.points_table``)."""

from __future__ import annotations

from pathlib import Path

import pytest

from marinedata.registry import Registry
from marinedata.task_layers import points_table as P
from marinedata.task_layers.configs import _point_records, _row_sha

SHA = "a" * 64
SEAVIEW_CSV = (
    "quadratid,y,x,label_name,label,func_group,method,data_set\n"
    "q1,100,200,Turf algae,Turf,Algae,random,train\n"
    "q1,5000,-3,Unknown thing,ZZZ_not_a_label,Other,random,test\n"
    "q2,10,10,Sand,Sand,Other,random,train\n"
)


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _info(qid: str):
    return None if qid == "q2" else (f"benthic_datasets/point_labels/SEAVIEW/R/{qid}.jpg", 400, 300)


def test_point_row_normalises_clamps_and_validates(registry):
    resolve = P.resolver_for(registry, "seaview-point-labels")
    row = P.point_row(
        source_id="seaview-survey-imagery",
        source_version="v1",
        ordinal=7,
        image_sha256=SHA,
        native="Turf",
        resolve=resolve,
        registry=registry,
        x_px=500,
        y_px=150,
        width=400,
        height=300,
        annotator="human_expert",
        ann_license="CC-BY-4.0",
    )
    assert row["ann_id"] == "seaview-survey-imagery:7"
    assert 0.0 <= row["x"] <= 1.0 and row["y"] == 0.5  # x clamped to the image edge
    assert row["annotator_type"] == "expert"
    assert row["match_type"] in {"exact", "broader", "narrower", "related"}
    assert row["taxon_node_id"] is not None


def test_seaview_rows_are_pending_with_image_key_and_unmapped_is_explicit(registry):
    rows = P.seaview_rows(SEAVIEW_CSV, "R", registry, _info, source_version="v1")
    assert len(rows) == 2  # q2 has no image info -> skipped, never guessed
    assert all(r["image_key"].endswith("/q1.jpg") for r in rows)
    assert all(r["image_sha256_todo"] == P.TODO_TEXT for r in rows)
    unmapped = next(r for r in rows if r["label_native"] == "ZZZ_not_a_label")
    assert unmapped["match_type"] == "unmapped" and unmapped["taxon_node_id"] is None
    assert unmapped["upstream_split"] == "test"
    assert 0.0 <= unmapped["x"] <= 1.0 and 0.0 <= unmapped["y"] <= 1.0
    assert P.validate_pending(rows) == []


def test_pending_parquet_has_image_key_not_sha(registry, tmp_path):
    import pyarrow.parquet as pq

    rows = P.seaview_rows(SEAVIEW_CSV, "R", registry, _info, source_version="v1")
    out = tmp_path / "p.parquet"
    assert P.write_pending_points(out, rows) == len(rows)
    cols = set(pq.read_table(out).column_names)
    assert {"image_key", "image_sha256_todo"} <= cols and "image_sha256" not in cols


def test_parse_cpc_twips_to_pixels():
    text = "\n".join(
        [
            '"C:\\codes.txt","C:\\imgs\\IMG_001.JPG",6000,4500,15000,12000',
            "0,0",
            "6000,0",
            "6000,4500",
            "0,4500",
            "2",
            "1500,3000",
            "300,150",
            '1,"HC","",""',
            '2,"TA","",""',
        ]
    )
    name, w, h, pts = P.parse_cpc(text)
    assert name == "IMG_001.JPG" and (w, h) == (400, 300)
    assert pts == [(100.0, 200.0, "HC"), (20.0, 10.0, "TA")]


def test_ibf_rows_resolve_codes_and_stay_pending(registry):
    text = "\n".join(
        [
            '"c","IMG_1.JPG",6000,4500,1,1',
            "0,0",
            "0,0",
            "0,0",
            "0,0",
            "1",
            "1500,3000",
            '1,"HC","",""',
        ]
    )
    vocab = Path(__file__).parents[1] / "registry/taxonomy/vocab/ibf-cpce-codes.tsv"
    code = next(
        ln.split("\t")[0] for ln in vocab.read_text().splitlines() if ln and not ln.startswith("#")
    )
    text = text.replace('"HC"', f'"{code}"')
    rows = P.ibf_rows(
        {"d/x.cpc": text}, registry, source_version="v1", image_keys={"d/IMG_1.JPG": "k/IMG_1.JPG"}
    )
    assert len(rows) == 1 and rows[0]["image_key"] == "k/IMG_1.JPG"
    assert rows[0]["annotator_type"] == "human"
    assert rows[0]["match_type"] in {"exact", "broader", "narrower", "related", "unmapped"}


def test_unified_points_win_and_legacy_are_renamed_on_read(registry, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    legacy = tmp_path / "_tasklabels" / "reefolution"
    legacy.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sha256": SHA,
                    "source_id": "reefolution",
                    "label_origin": "human",
                    "native_label": "Acr",
                    "x": 0.1,
                    "y": 0.2,
                }
            ]
        ),
        legacy / "points.parquet",
    )
    (rec,) = _point_records(tmp_path, "reefolution")
    assert (rec["image_sha256"], rec["label_native"], rec["annotator_type"]) == (
        SHA,
        "Acr",
        "human",
    )
    assert _row_sha({"sha256": SHA}) == _row_sha({"image_sha256": SHA}) == SHA

    new = tmp_path / "_annotations" / "points" / "reefolution"
    new.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_sha256": SHA,
                    "source_id": "reefolution",
                    "ann_id": "reefolution:0",
                    "label_native": "Zed",
                    "annotator_type": "expert",
                    "x": 0.3,
                    "y": 0.4,
                }
            ]
        ),
        new / "v1.parquet",
    )
    (rec,) = _point_records(tmp_path, "reefolution")
    assert rec["label_native"] == "Zed"


@pytest.mark.parametrize("crosswalk", ["seaview-point-labels", "ibf-cpce-codes"])
def test_new_crosswalks_load_and_cover_vocab(registry, crosswalk):
    walk = registry.crosswalk(crosswalk)
    assert walk.target_schema and len(walk.edges) > 5
