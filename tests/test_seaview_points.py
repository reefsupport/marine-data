"""P3.1: the Seaview benthic points table and ``marinedata tasks build-points``."""

from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata import cli
from marinedata.annotation_schema import read_annotations
from marinedata.metadata_norm.resolvers.staged import staged_images
from marinedata.registry import Registry
from marinedata.task_layers import configs as C
from marinedata.task_layers import points_table as P
from marinedata.task_layers import seaview_points as S

VERSION = "v-test"
TREE = f"sources/{S.SOURCE_ID}/{VERSION}/"
REGION = "TESTREG"
Q1, Q2, Q3, Q4, ORPHAN = "11111111111", "22222222222", "33333333333", "44444444444", "55555555555"
SIZES = {Q1: (40, 30), Q2: (64, 48), Q3: (32, 32), Q4: (16, 16)}  # Q4: an image with no rows
HEADER = "quadratid,y,x,label_name,label,func_group,method,data_set\n"
POINTS = [  # (quadrat, y, x, label_name, label, func_group, data_set)
    (Q1, 10, 20, "Hard coral", "EAM_DHC", "Hard coral", "train"),
    (Q1, 15, 5, "Sand", "LSUB_SAND", "Substrate", "train"),
    (Q1, 500, -3, "Mystery", "ZZZ_not_a_label", "Other", "train"),  # outside the frame, unmapped
    (Q2, 3, 4, "Dead substrate", "DSUB", "Substrate", "test"),
    (Q2, 7, 8, "Algal mat", "AMAT", "Algae", "test"),
    (Q3, 9, 9, "Recently bleached", "EAM_RB", "Hard coral", "train"),
    (ORPHAN, 1, 1, "Sand", "LSUB_SAND", "Substrate", "train"),  # an id with no image
    (ORPHAN, 2, 2, "Sand", "LSUB_SAND", "Substrate", "train"),
]
MASTER_SAMPLE = (  # Q2's second label differs; the last id and the cut line are dropped
    HEADER
    + f"{Q1},10,20,x,EAM-DHC,x,random,train\n{Q1},15,5,x,LSUB_SAND,x,random,train\n"
    + f"{Q1},500,-3,x,ZZZ_not_a_label,x,random,train\n"
    + f"{Q2},3,4,x,DSUB,x,random,test\n{Q2},7,8,x,DSUB,x,random,test\n"
    + f"{Q3},9,9,x,EAM_RB,x,random,train\n99999999999,5,"
).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jpeg(quadrat: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", SIZES[quadrat], (int(quadrat[0]) * 20, 90, 140)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


@pytest.fixture(scope="module")
def tree() -> SimpleNamespace:
    jpgs = {q: _jpeg(q) for q in SIZES}
    shas = {q: _sha(b) for q, b in jpgs.items()}
    meta = [
        {
            "partition": REGION,
            "stem": q,
            "image_path": f"images/{REGION}/{q}.jpg",
            "image_sha256": shas[q],
            "width": SIZES[q][0],
            "height": SIZES[q][1],
        }
        for q in SIZES
    ]
    twin = {  # the labelled_data copy of Q1: same stem, other bytes
        "partition": S.LABELLED_PARTITION,
        "stem": Q1,
        "image_path": f"images/{S.LABELLED_PARTITION}/{Q1}.jpg",
        "image_sha256": "f" * 64,
        "width": 40,
        "height": 30,
    }
    meta.append(twin)
    csv_text = HEADER + "".join(
        f"{q},{y},{x},{name},{code},{group},random,{split}\n"
        for q, y, x, name, code, group, split in POINTS
    )
    return SimpleNamespace(
        jpgs=jpgs,
        shas=shas,
        meta=meta,
        csv=csv_text,
        checksums={m["image_path"]: m["image_sha256"] for m in meta},
    )


def _build(registry, tree, **kw):
    return S.build_seaview_points(
        registry,
        version=VERSION,
        meta_rows=tree.meta,
        checksums=tree.checksums,
        csv_texts={REGION: tree.csv},
        **kw,
    )


def test_quadrat_index_uses_region_images_only(tree):
    index, notes = S.quadrat_index(staged_images(tree.meta, tree.checksums))
    assert set(index) == set(SIZES)
    assert index[Q1].image_sha256 == tree.shas[Q1] and index[Q1].partition == REGION
    assert (index[Q2].width, index[Q2].height) == SIZES[Q2]
    assert notes["labelled_data_rows_excluded"] == 1
    assert notes["labelled_data_rows_with_other_sha256_than_region_twin"] == 1


def test_every_point_is_keyed_or_listed_as_unmatched(registry, tree):
    res = _build(registry, tree)
    assert len(res.rows) == 6
    assert {r["image_sha256"] for r in res.rows} == {tree.shas[q] for q in (Q1, Q2, Q3)}
    assert len({r["ann_id"] for r in res.rows}) == 6
    unmatched = {(u["kind"], u["quadrat_id"]): u for u in res.unmatched}
    assert set(unmatched) == {
        ("annotation_id_without_image", ORPHAN),
        ("image_without_annotation_rows", Q4),
    }
    assert unmatched[("annotation_id_without_image", ORPHAN)]["n_points"] == 2
    assert all(u["reason"] for u in res.unmatched)
    totals = res.summary["totals"]
    assert totals["csv_rows"] == len(POINTS) == totals["points"] + totals["unmatched_points"]
    region = res.summary["regions"][REGION]
    assert (region["csv_rows"], region["points"], region["unmatched_points"]) == (8, 6, 2)
    assert (region["images_in_tree"], region["images_with_points"]) == (4, 3)


def test_provenance_is_the_registry_entry(registry, tree):
    res = _build(registry, tree)
    assert {r["annotator_type"] for r in res.rows} == {"human"}
    assert {r["ann_license"] for r in res.rows} == {P.source_licence(registry, S.SOURCE_ID)}
    assert {r["ann_license"] for r in res.rows} == {"CC-BY-3.0"}
    citation = registry.source(S.SOURCE_ID).citation.strip()
    assert {r["ann_attribution"] for r in res.rows} == {citation}
    assert "doi:10.14264/UQL.2019.930" in citation
    assert res.summary["ann_license"] == "CC-BY-3.0" and res.summary["ann_attribution"] == citation
    assert {r["source_version"] for r in res.rows} == {VERSION}


def test_coordinates_are_normalised_and_out_of_frame_points_flagged(registry, tree):
    res = _build(registry, tree)
    first = res.rows[0]
    assert (first["x_px"], first["y_px"]) == (20, 10)
    assert first["x"] == pytest.approx(20 / 40) and first["y"] == pytest.approx(10 / 30)
    attrs = json.loads(first["attrs"])
    assert attrs["quadrat_id"] == Q1 and attrs["func_group"] == "Hard coral"
    out = [r for r in res.rows if json.loads(r["attrs"]).get("clamped")]
    assert len(out) == 1 and res.summary["clamped_points"] == 1
    assert 0.0 <= out[0]["x"] <= 1.0 and 0.0 <= out[0]["y"] <= 1.0


def test_crosswalk_histogram_and_worms_fallback(registry, tree):
    plain = P.resolver_for(registry, S.CROSSWALK_ID)
    assert plain("ZZZ_not_a_label")[1] == "unmapped"
    assert all(
        plain(c)[1] != "unmapped" for c in ("EAM_DHC", "LSUB_SAND", "DSUB", "AMAT", "EAM_RB")
    )
    res = _build(registry, tree)
    hist = res.summary["match_type"]
    assert hist["unmapped"] == 1 and sum(hist.values()) == 6
    assert set(hist) <= {"exact", "broader", "related", "unmapped"}
    assert res.summary["crosswalked_pct"] == pytest.approx(100 * 5 / 6, abs=1e-3)
    assert res.summary["worms_aphia_pct"] == res.summary["worms_aphia_pct_from_taxonomy_nodes"]
    filled = _build(registry, tree, fallback=lambda node: 424242)
    assert all(r["worms_aphia_id"] is not None for r in filled.rows if r["taxon_node_id"])
    assert all(r["worms_aphia_id"] is None for r in filled.rows if not r["taxon_node_id"])
    assert filled.summary["worms_aphia_pct"] >= res.summary["worms_aphia_pct"]


def test_master_sample_cross_check(registry, tree):
    check = _build(registry, tree, master_sample=MASTER_SAMPLE).summary["master_check"]
    assert check["read"] is True and check["compared"] == 2 and check["fully_equal"] == 1
    assert (check["same_coordinates"], check["same_labels"], check["mismatching_ids"]) == (2, 1, 1)
    assert [m["quadrat_id"] for m in check["mismatches_first_20"]] == [Q2]
    assert _build(registry, tree).summary["master_check"]["read"] is False


def test_worms_cache_is_capped_persisted_and_null_cached(tmp_path):
    path = tmp_path / "taxonomy" / "worms-cache.json"
    asked: list[str] = []

    def fetch(name: str):
        asked.append(name)
        return {"Acropora": 205469, "Nothing": None}[name]

    cache = S.WormsCache(path, fetch=fetch, max_calls=2)
    assert cache.aphia("Acropora") == 205469 and cache.aphia("Acropora") == 205469
    assert cache.aphia("Nothing") is None and cache.aphia("Nothing") is None
    assert cache.aphia("Porites") is None  # over the cap: no call, nothing cached
    assert asked == ["Acropora", "Nothing"] and cache.calls == 2
    cache.save()
    names = json.loads(path.read_text())["names"]
    assert names["Acropora"]["aphia_id"] == 205469 and names["Nothing"]["aphia_id"] is None
    assert "Porites" not in names

    def refuse(name: str):
        raise AssertionError(name)

    again = S.WormsCache(path, fetch=refuse, max_calls=2)
    assert again.aphia("Acropora") == 205469 and again.aphia("Nothing") is None
    assert again.calls == 0


def test_worms_transient_failure_is_not_cached(tmp_path):
    def down(name: str):
        raise OSError("down")

    cache = S.WormsCache(tmp_path / "w.json", fetch=down)
    assert cache.aphia("Acropora") is None
    assert cache.failed == 1 and "Acropora" not in cache.names
    assert S.WormsCache(None, enabled=False).aphia("Acropora") is None


def test_offset_hint_is_advice_only():
    hints = S._offset_hints(["100", "200"], ["1100", "1200"])
    assert "annotation id 1100" in hints["100"] and "+1000" in hints["100"]
    assert "not applied" in hints["1200"]
    assert S._offset_hints(["100", "200"], ["1100", "1300"]) == {}
    assert S._offset_hints(["100"], ["1100"]) == {}


def test_seaview_is_a_point_source_but_not_a_cover_rollup_source():
    assert S.SOURCE_ID in C.POINT_SOURCES
    assert S.SOURCE_ID not in C.ROLLUP_POINT_SOURCES


def test_fixture_config_builds_and_loads_with_datasets(registry, tree, tmp_path, monkeypatch):
    datasets = pytest.importorskip("datasets")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")
    res = _build(registry, tree)
    data = tmp_path / "data"
    S.write_outputs(tmp_path / "out", res, version=VERSION, annotations_root=data)
    cfg = C.build_points_config(registry, data)
    assert 5 <= len(cfg.rows) <= 6
    wanted = {
        "image_sha256", "x", "y", "x_norm", "y_norm", "label_native", "label_native_group",
        "taxon_node_id", "worms_aphia_id", "match_type", "annotator_type", "ann_license",
        "ann_attribution", "source_version", "quadrat_id",
    }  # fmt: skip
    assert wanted <= set(cfg.rows[0])
    folder = tmp_path / "cfg"
    (folder / "images").mkdir(parents=True)
    for q in (Q1, Q2, Q3):
        (folder / "images" / f"{tree.shas[q]}.jpg").write_bytes(tree.jpgs[q])
    pq.write_table(pa.Table.from_pylist(cfg.rows), folder / "train.parquet")
    ds = datasets.load_dataset(
        "parquet",
        data_files={"train": str(folder / "train.parquet")},
        split="train",
        cache_dir=str(tmp_path / "hf"),
    )
    assert ds.num_rows == len(cfg.rows) and wanted <= set(ds.column_names)
    assert set(ds["image_sha256"]) == {tree.shas[q] for q in (Q1, Q2, Q3)}
    assert set(ds["annotator_type"]) == {"human"} and set(ds["source_version"]) == {VERSION}
    for sha in set(ds["image_sha256"]):  # the key resolves to the image bytes
        assert _sha((folder / "images" / f"{sha}.jpg").read_bytes()) == sha


def _objects(tree) -> dict[str, bytes]:
    meta = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(tree.meta), meta)
    files = {
        "metadata.parquet": meta.getvalue(),
        f"{S.TABLES}annotations_{REGION}.csv": tree.csv.encode(),
        S.MASTER: b"quadratid,y,x,label\n",  # never fetched whole: the ranged read is stubbed
    }
    sums = dict(tree.checksums) | {rel: _sha(b) for rel, b in files.items()}
    manifest = "".join(f"{sha}  {rel}\n" for rel, sha in sorted(sums.items()))
    out = {TREE + rel: b for rel, b in files.items()}
    out[TREE + "CHECKSUMS.sha256"] = manifest.encode()
    return out


def test_run_build_points_end_to_end(tree, tmp_path):
    objects = _objects(tree)
    asked: list[str] = []

    def fetch(key: str) -> bytes:
        asked.append(key)
        assert "/images/" not in key
        return objects[key]

    summary = S.run_build_points(
        tmp_path / "out",
        version=VERSION,
        cache_dir=tmp_path / "cache",
        worms_cache_path=tmp_path / "worms.json",
        annotations_root=tmp_path / "data",
        fetch=fetch,
        range_fetch=lambda rel, n: MASTER_SAMPLE,
        worms_fetch=lambda name: 777,
    )
    out = tmp_path / "out"
    assert summary["totals"]["points"] == 6 and summary["master_check"]["read"] is True
    read_annotations(out / "points.parquet", "points")
    assert pq.read_table(out / "points.parquet").num_rows == 6
    lines = (out / "unmatched.tsv").read_text().splitlines()
    assert lines[0].split("\t") == list(S.UNMATCHED_COLUMNS) and len(lines) == 3
    assert json.loads((out / "summary.json").read_text())["totals"] == summary["totals"]
    assert "names" in json.loads((tmp_path / "worms.json").read_text())
    assert summary["worms_lookup"]["calls"] <= S.WORMS_MAX_CALLS
    layer = tmp_path / "data" / "_annotations" / "points" / S.SOURCE_ID / f"{VERSION}.parquet"
    assert layer.is_file()
    assert TREE + "metadata.parquet" in asked and not any(k.endswith(".jpg") for k in asked)


def test_cli_help_source_guard_and_wiring(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["tasks", "build-points", "--help"])
    assert exc.value.code == 0 and "--source" in capsys.readouterr().out
    argv = ["tasks", "build-points", "--source", "reefolution", "--out", str(tmp_path)]
    assert cli.main(argv) == 2
    seen: dict = {}

    def fake(out, **kw):
        seen.update(kw, out=str(out))
        return {
            "source_version": "v",
            "crosswalked_pct": 99.0,
            "worms_aphia_pct": 50.0,
            "totals": {
                "points": 1,
                "images_with_points": 1,
                "annotation_ids_without_image": 0,
                "images_without_annotation_rows": 0,
            },
        }

    monkeypatch.setattr(S, "run_build_points", fake)
    argv = ["tasks", "build-points", "--source", S.SOURCE_ID, "--out", str(tmp_path / "o")]
    assert cli.main([*argv, "--no-worms", "--master-sample-bytes", "0"]) == 0
    assert seen["worms"] is False and seen["master_sample_bytes"] == 0
    assert seen["out"] == str(tmp_path / "o") and "crosswalked 99.0%" in capsys.readouterr().out
