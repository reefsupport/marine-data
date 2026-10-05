"""WP-U13: the unified ``identities`` table, seaturtleid2022 first (offline, stubs)."""

# ruff: noqa: E501

from __future__ import annotations

import json

import pytest

from marinedata.annotation_schema import validate_rows
from marinedata.registry import Registry, _default_root
from marinedata.task_layers.sources import identities as idm

SPEC = idm.IDENTITY_SOURCES["seaturtleid2022"]


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load(_default_root())


def _meta(shard: int, row: int, w: int = 100, h: int = 50) -> dict:
    sha = f"{shard}{row}".ljust(64, "e")
    return {"upstream_id": f"data/train-0000{shard}-of-00002.parquet#{row}", "image_sha256": sha, "width": w, "height": h}  # fmt: skip


def test_rows_follow_the_upstream_shard_row_order_not_the_staged_order(reg):
    meta = [_meta(1, 0), _meta(0, 1), _meta(0, 10), _meta(0, 2)]  # shuffled; 10 sorts after 2
    assert [m["upstream_id"][-12:] for m in idm.order_meta(meta)] == [
        "00002.parquet#1"[-12:], "00002.parquet#2"[-12:], "00002.parquet#10"[-12:], "00002.parquet#0"[-12:],
    ]  # fmt: skip
    recs = [
        {"identity": i, "width": 100, "height": 50, "year": 2020, "split_open": "valid"}
        for i in (11, 22, 33, 44)
    ]
    res = idm.staged_identities(SPEC, reg, meta, recs)
    rows = list(res.rows)
    validate_rows("identities", rows)
    by_sha = {r["image_sha256"]: r["individual_id"] for r in rows}
    assert (
        by_sha[_meta(0, 1)["image_sha256"]] == "11" and by_sha[_meta(1, 0)["image_sha256"]] == "44"
    )
    assert res.individuals == 4 and rows[0]["ann_id"] == "seaturtleid2022:0"


def test_species_resolves_through_the_crosswalk_and_the_licence_class_is_unknown(reg):
    res = idm.staged_identities(
        SPEC,
        reg,
        [_meta(0, 0)],
        [{"identity": 5, "width": 100, "height": 50, "split_open": "test"}],
    )
    (row,) = res.rows
    assert row["label_native"] == "Caretta caretta" and row["taxon_node_id"] == "A137205"
    assert row["match_type"] != "unmapped" and row["worms_aphia_id"] == 137205
    assert (row["individual_id"], row["individual_scope"], row["upstream_split"]) == (
        "5",
        "dataset",
        "test",
    )
    assert row["x_min"] is None and row["is_crowd"] is None  # no staged box
    attrs = json.loads(row["attrs"])
    assert attrs["licence_class"] == "unknown" and attrs["species_origin"] == "dataset-level"


def test_size_mismatch_and_missing_identity_are_dropped_and_counted(reg):
    meta = [_meta(0, 0), _meta(0, 1), _meta(0, 2)]
    recs = [
        {"identity": 1, "width": 100, "height": 50},
        {"identity": 2, "width": 999, "height": 50},  # not this image
        {"identity": None, "width": 100, "height": 50},
        {"identity": 4, "width": 100, "height": 50},  # no staged image for this record
    ]
    res = idm.staged_identities(SPEC, reg, meta, recs)
    assert (len(res.rows), res.mismatched, res.unlabelled, res.unstaged) == (1, 1, 1, 1)
    assert idm.staged_identities(SPEC, reg, meta, recs, limit=1).rows[0]["individual_id"] == "1"


def test_bad_upstream_id_is_an_error_and_hf_rows_pages(reg):
    with pytest.raises(ValueError, match="upstream_id"):
        idm.order_meta([{"upstream_id": "nope"}])
    seen: list[str] = []

    def get(url: str) -> bytes:
        seen.append(url)
        n = int(url.split("length=")[1])
        return json.dumps({"rows": [{"row": {"identity": i}} for i in range(n)]}).encode()

    out = idm.hf_rows("o/d", 250, get=get, page=100)
    assert (
        len(out) == 250
        and len(seen) == 3
        and "offset=200" in seen[2]
        and seen[2].endswith("length=50")
    )
