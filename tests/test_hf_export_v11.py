"""PR-E (E1/E2): the v1.1.0 export of a synthetic build dir (no bucket, no Hub)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.gates.join import run_join_gate
from marinedata.gates.masks_palette import check_palette_row
from marinedata.hf_export_v11 import (
    V11ExportError,
    assert_public_safe,
    build_v11_layout,
    drop_hidden_test,
    export_v11,
    hidden_test_drop,
    image_rows,
    is_v11_release,
    label_map_rows,
    read_checksums,
    table_crosswalk_ids,
    v1_sha256s,
)
from marinedata.registry import Registry
from marinedata.task_layers.configs.common import ConfigResult
from marinedata.task_layers.v11_tables import V11Inputs, build_v11_tables, load_v1_config_shas

TAX = "3.0.0"
OPEN_IMG, OPEN_IMG_B, V1_IMG, SKIP_IMG, MASK_IMG, NC_IMG = (f"{i:064x}" for i in range(1, 7))
SHARD = re.compile(
    r"^data/[a-z-]+/(train|validation|test|test_ood)(-[a-z0-9-]+)?-\d{5}-of-\d{5}\.parquet$"
)


@pytest.fixture(scope="module")
def registry():
    return Registry.load()


def _png(value: int) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (4, 4), value).save(buf, format="PNG")
    return buf.getvalue()


def _tree(root: Path, files: dict[str, bytes]) -> dict[str, str]:
    """A staged tree: the files plus a CHECKSUMS.sha256 whose digests are the image keys."""
    root.mkdir(parents=True, exist_ok=True)
    lines = []
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        lines.append(f"{hashlib.sha256(data).hexdigest()}  {rel}")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    return {rel: hashlib.sha256(d).hexdigest() for rel, d in files.items()}


def _meta(sha, sid, config="images", split="train", cls="open", **kw):
    return {
        "image_sha256": sha, "source_id": sid, "bytes_config": config, "split": split,
        "split_group": f"g-{sha[-2:]}", "taxonomy_version": TAX, "licence_class": cls,
        "bytes_sha256": None, **kw,
    }  # fmt: skip


def _write(rel: Path, name: str, rows: list[dict]) -> None:
    d = rel / "task_layers"
    d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), d / f"{name}.parquet")


@pytest.fixture
def build(tmp_path):
    """A 12-table synthetic v1.1 build: open sources, one v1.0-embedded source, one nc source."""
    cache = tmp_path / "cache"
    a = {"images/a.jpg": b"pixels-a" * 40, "images/a2.jpg": b"pixels-a2" * 40}
    keys_a = _tree(cache / "seaview-survey-imagery", a)
    keys_b = _tree(cache / "noaa-pifsc-bleaching", {"images/b.jpg": b"pixels-b" * 40})
    keys_s = _tree(
        cache / "reef-support-bleaching",
        {"images/m.jpg": b"pixels-m" * 40, "labels/m.png": _png(1)},
    )
    keys_n = _tree(cache / "mermaid-aws", {"images/n.jpg": b"pixels-n" * 40})
    # the image keys of the staged trees are the sha256 of their bytes: use them
    sha = {"a": keys_a["images/a.jpg"], "a2": keys_a["images/a2.jpg"], "b": keys_b["images/b.jpg"],
           "m": keys_s["images/m.jpg"], "n": keys_n["images/n.jpg"]}  # fmt: skip
    rel = tmp_path / "rel"
    rel.mkdir()
    (rel / "RELEASE.json").write_text(json.dumps({"flavour": "open", "release": "v1.1"}))
    meta = [
        _meta(sha["a"], "seaview-survey-imagery"),
        _meta(sha["a2"], "seaview-survey-imagery", split="validation"),
        _meta(sha["b"], "noaa-pifsc-bleaching", split="test"),
        _meta(sha["m"], "reef-support-bleaching", split="test_ood"),
        _meta(V1_IMG, "suim", config="scene-masks"),
        _meta(SKIP_IMG, "seaview-survey-imagery"),
        _meta(sha["n"], "mermaid-aws", cls="restricted-nc"),
    ]

    def label(sha_, sid, name, node="HC", **kw):
        split = next(m["split"] for m in meta if m["image_sha256"] == sha_)
        return {
            "image_sha256": sha_, "source_id": sid, "label_native": name, "taxon_node_id": node,
            "split": split, "taxonomy_version": TAX, **kw,
        }  # fmt: skip

    _write(rel, "metadata", meta)
    _write(rel, "points", [label(sha["a"], "seaview-survey-imagery", "Hard coral"),
        label(sha["a2"], "seaview-survey-imagery", "never-heard-of-it", node=None),
                           label(sha["n"], "mermaid-aws", "nc-label")])  # fmt: skip
    _write(rel, "bleaching", [label(sha["b"], "noaa-pifsc-bleaching", "bleached", node="BLEACHED")])
    _write(rel, "boxes", [label(V1_IMG, "suim", "fish", x_min=0.1)])
    cmap = json.dumps([{"id": 1, "label_native": "coral", "taxon_node": "HC", "coarse": "HC"}])
    _write(rel, "semseg", [
        label(
            sha["m"], "reef-support-bleaching", "coral", mask_ref="labels/m.png", mask_config=None,
              ignore_value=None, class_map=cmap, palette="[]"),
        label(V1_IMG, "suim", "water", mask_ref="x.png", mask_config="scene-masks",
              ignore_value=None, class_map=None, palette="[]"),
    ])  # fmt: skip
    _write(rel, "instances", [label(V1_IMG, "suim", "fish", mask_config="instance-masks")])
    _write(
        rel,
        "benthic-cover",
        [{"survey_id": "s1", "source_id": "seaview-survey-imagery", "cover": 0.4}],
    )
    _write(rel, "sites", [{"site_id": "x", "source_id": "seaview-survey-imagery", "n": 2}])
    (rel / "task_layers" / "V11_TABLES.json").write_text(json.dumps({"taxonomy_version": TAX}))
    roots = {p.name: p for p in cache.iterdir()}
    return rel, roots, sha


def _layout(registry, build, flavour="open", **kw):
    rel, roots, _ = build
    return build_v11_layout(registry, rel, roots, flavour=flavour, skip_sha256={SKIP_IMG}, **kw)


def _rows(out: Path, config: str) -> list[dict]:
    files = sorted((out / "data" / config).glob("*.parquet"))
    return [r for f in files for r in pq.read_table(f).to_pylist()]


def test_is_v11_release(build):
    assert is_v11_release(build[0])
    assert not is_v11_release(build[0] / "task_layers")


def test_twelve_configs_with_viewer_friendly_shards(registry, build, tmp_path):
    layout, _ = _layout(registry, build, shard_target_bytes=2000)
    out = tmp_path / "out"
    summary = export_v11(layout, out)
    assert sorted(summary["configs"]) == sorted(
        [
            "images",
            "metadata",
            "points",
            "bleaching",
            "boxes",
            "semseg",
            "instances",
            "benthic-cover",
            "sites",
            "taxonomy",
            "label-map",
            "sources",
        ]
    )
    assert summary["layout"] == "v1.1"
    names = [n for n in summary["files"]]
    assert names and all(SHARD.match(n) for n in names), names
    for spec, groups in layout.values():
        for stem, rows in groups.items():
            from marinedata.hf_parquet import plan_config

            for plan in plan_config(spec, {stem: rows}):
                assert plan.planned_bytes <= 2000 or len(plan.rows) == 1
    for n in names:
        f = pq.ParquetFile(out / n)
        assert f.metadata.num_rows > 0
        if "images" in n.split("/")[1:2]:
            assert max(f.metadata.row_group(i).num_rows for i in range(f.num_row_groups)) <= 100


def test_images_embed_bytes_once_and_skip_v1(registry, build, tmp_path):
    layout, report = _layout(registry, build)
    export_v11(layout, tmp_path / "out")
    rows = _rows(tmp_path / "out", "images")
    keys = [r["image_sha256"] for r in rows]
    assert len(keys) == len(set(keys))
    sha = build[2]
    assert set(keys) == {
        sha["a"],
        sha["a2"],
        sha["b"],
        sha["m"],
    }  # not V1_IMG, SKIP_IMG, the nc image
    for r in rows:
        assert (
            hashlib.sha256(r["image"]["bytes"]).hexdigest()
            == r["image_sha256"]
            == r["bytes_sha256"]
        )
    assert report["images"]["skipped_v1_config"] == 1 and report["images"]["skipped_v1_sha256"] == 1


def test_metadata_bytes_sha256_only_for_embedded(registry, build, tmp_path):
    layout, _ = _layout(registry, build)
    export_v11(layout, tmp_path / "out")
    by = {r["image_sha256"]: r for r in _rows(tmp_path / "out", "metadata")}
    assert by[build[2]["a"]]["bytes_sha256"] == build[2]["a"]
    assert by[V1_IMG]["bytes_sha256"] is None and by[V1_IMG]["bytes_config"] == "scene-masks"
    assert all(r["taxonomy_version"] == TAX for r in by.values())


def test_semseg_palette_png_only_for_masks_no_v1_config_holds(registry, build, tmp_path):
    layout, _ = _layout(registry, build)
    export_v11(layout, tmp_path / "out")
    rows = {r["source_id"]: r for r in _rows(tmp_path / "out", "semseg")}
    new, old = rows["reef-support-bleaching"], rows["suim"]
    png = new["mask"]["bytes"]
    assert Image.open(io.BytesIO(png)).mode == "P"
    assert new["mask_bytes_sha256"] == hashlib.sha256(png).hexdigest()
    assert old["mask"] is None and old["mask_bytes_sha256"] is None


def test_a_bucket_key_mask_ref_embeds_the_same_png_as_a_tree_relative_one(
    registry, build, tmp_path
):
    """The unified masks table writes ``sources/<sid>/<ver>/<path>``, legacy tables ``<path>``."""
    sid, table = "reef-support-bleaching", build[0] / "task_layers" / "semseg.parquet"

    def semseg_row(mask_ref: str, out: str) -> dict:
        rows = [{**r, "mask_ref": mask_ref} if r["source_id"] == sid else r
                for r in pq.read_table(table).to_pylist()]  # fmt: skip
        _write(build[0], "semseg", rows)
        export_v11(_layout(registry, build)[0], tmp_path / out)
        return next(r for r in _rows(tmp_path / out, "semseg") if r["source_id"] == sid)

    relative = semseg_row("labels/m.png", "relative")
    keyed = semseg_row(f"sources/{sid}/2.0/labels/m.png", "keyed")
    assert relative["mask_config"] is keyed["mask_config"] is None
    assert keyed["mask_ref"] == f"sources/{sid}/2.0/labels/m.png"
    assert keyed["mask"]["bytes"] == relative["mask"]["bytes"]
    assert keyed["mask_bytes_sha256"] == relative["mask_bytes_sha256"]
    # only the row's own source prefix is stripped: another source's key is still not found
    with pytest.raises(V11ExportError, match=rf"{sid}: mask sources/suim/2\.0/labels/m\.png not"):
        semseg_row("sources/suim/2.0/labels/m.png", "other")


def test_nc_rows_never_reach_the_open_export(registry, build, tmp_path):
    layout, report = _layout(registry, build)
    export_v11(layout, tmp_path / "out")
    for config in ("metadata", "points", "images", "sources", "label-map"):
        assert "mermaid-aws" not in json.dumps(_rows(tmp_path / "out", config), default=str)
    assert report["licence_rows_dropped"]["metadata"] == 1


def test_nc_flavour_exports_only_its_delta_and_no_taxonomy(registry, build, tmp_path):
    rel, roots, sha = build
    layout, _ = build_v11_layout(registry, rel, roots, flavour="nc", skip_sha256={SKIP_IMG})
    assert "taxonomy" not in layout
    export_v11(layout, tmp_path / "out")
    assert {r["image_sha256"] for r in _rows(tmp_path / "out", "images")} == {sha["n"]}
    assert {r["source_id"] for r in _rows(tmp_path / "out", "sources")} == {"mermaid-aws"}


def test_a_class_that_can_never_ship_raises(registry, build):
    rel, roots, _ = build
    rows = pq.read_table(rel / "task_layers" / "metadata.parquet").to_pylist()
    rows.append(_meta("f" * 64, "suim", cls="internal-only"))
    _write(rel, "metadata", rows)
    with pytest.raises(V11ExportError, match="can never ship"):
        build_v11_layout(registry, rel, roots, flavour="open")


def test_an_image_without_a_split_raises(registry, build):
    rel, _roots, sha = build
    rows = pq.read_table(rel / "task_layers" / "metadata.parquet").to_pylist()
    rows = [{**r, "split": None} if r["image_sha256"] == sha["a"] else r for r in rows]
    _write(rel, "metadata", rows)
    with pytest.raises(V11ExportError, match=r"no Hub split name"):
        _layout(registry, build)


def test_label_map_carries_unmapped_labels_and_stamps_taxonomy(registry, build):
    layout, _ = _layout(registry, build)
    rows = [r.values for rs in layout["label-map"][1].values() for r in rs]
    seaview = [r for r in rows if r["source_id"] == "seaview-survey-imagery"]
    carried = [r for r in seaview if r["native_label"] == "never-heard-of-it"]
    assert carried and carried[0]["mapped"] is False and carried[0]["match_type"] == "unmapped"
    assert all(r["taxonomy_version"] == TAX for r in rows)
    tax = [r.values for rs in layout["taxonomy"][1].values() for r in rs]
    assert {r["kind"] for r in tax} == {"node", "edge"} and all(
        r["taxonomy_version"] == TAX for r in tax
    )


def test_label_map_has_edges_for_box_sources_whose_loader_declares_no_crosswalk(registry):
    rows = label_map_rows(registry, ["roboflow-aquarium", "usis10k"], TAX)
    by = {(r["source_id"], r["native_label"]): r for r in rows}
    puffin, reefs = by[("roboflow-aquarium", "puffin")], by[("usis10k", "reefs")]
    assert puffin["crosswalk_id"] == "roboflow-aquarium" and puffin["mapped"] is True
    assert puffin["taxon_node_id"] == "A137039"
    assert reefs["crosswalk_id"] == "usis-uiis" and reefs["mapped"] is True
    assert reefs["taxon_node_id"] == "NT_REEF"


def test_table_crosswalks_agree_with_the_loaders_that_declare_one(registry):
    for sid, cid in table_crosswalk_ids().items():
        loader = registry.source(sid).loader
        assert loader is None or loader.crosswalk_id in (None, cid), sid


def test_taxonomy_carries_the_edges_of_box_sources_whose_loader_declares_no_crosswalk(
    registry, build
):
    rel, roots, _ = build
    rows = pq.read_table(rel / "task_layers" / "metadata.parquet").to_pylist()
    boxes = (("e", "roboflow-aquarium"), ("d", "usis10k"))
    rows += [_meta(h * 64, sid, config="boxes") for h, sid in boxes]
    _write(rel, "metadata", rows)
    layout, _ = build_v11_layout(registry, rel, roots, flavour="open", skip_sha256={SKIP_IMG})
    edges = {
        r.values["key"]: r.values
        for rs in layout["taxonomy"][1].values()
        for r in rs
        if r.values["kind"] == "edge"
    }
    assert json.loads(edges["roboflow-aquarium:puffin"]["targets"]) == {"taxon": "A137039"}
    assert json.loads(edges["usis-uiis:reefs"]["targets"]) == {"taxon": "NT_REEF"}


def test_open_tables_never_name_a_private_resource(registry, build):
    layout, _ = _layout(registry, build)
    text = json.dumps(
        [r.values for _, g in layout.values() for rs in g.values() for r in rs], default=str
    )
    assert_public_safe(text, "open")
    with pytest.raises(V11ExportError):
        assert_public_safe("see reefsupport/marine-data-nc", "open")


def test_checksums_missing_is_an_error(tmp_path):
    with pytest.raises(V11ExportError, match="CHECKSUMS"):
        read_checksums(tmp_path)


def test_hf_export_main_delegates_a_v11_release(build, monkeypatch, tmp_path):
    from marinedata import hf_export
    from marinedata import hf_export_v11 as v11

    seen = {}

    def fake_run(args):
        seen["release_dir"], seen["v1_dir"] = args.release_dir, args.v1_dir
        return 0

    monkeypatch.setattr(v11, "run", fake_run)
    argv = ["--release-dir", str(build[0]), "--out", str(tmp_path / "o")]
    argv += ["--summary", str(tmp_path / "s.json"), "--v1-dir", str(tmp_path / "v1")]
    assert hf_export.main(argv) == 0
    assert seen == {"release_dir": build[0], "v1_dir": tmp_path / "v1"}


def test_exported_rows_carry_the_hub_split_and_a_licence_class(registry, build, tmp_path):
    """The build calls the validation split `probe`; every exported row names the Hub split of its
    shard, and a task table whose licence_class is empty takes its image's class (VM rehearsal)."""
    rel, _, sha = build
    meta = pq.read_table(rel / "task_layers" / "metadata.parquet").to_pylist()
    probe = [{**m, "split": "probe"} if m["image_sha256"] == sha["a2"] else m for m in meta]
    _write(rel, "metadata", probe)
    task = {
        "taxonomy_version": TAX,
        "licence_class": None,
        "label_native": "x",
        "taxon_node_id": "HC",
    }
    _write(rel, "bleaching", [
        {
            **task,
            "image_sha256": sha["a2"],
            "source_id": "seaview-survey-imagery",
            "split": "probe",
        },
        {**task, "image_sha256": sha["b"], "source_id": "noaa-pifsc-bleaching", "split": "test"},
    ])  # fmt: skip
    layout, _ = _layout(registry, build)
    out = tmp_path / "out"
    export_v11(layout, out)
    for config in ("metadata", "images", "bleaching"):
        shards = sorted((out / "data" / config).glob("*.parquet"))
        assert shards, config
        for shard in shards:
            splits = {r["split"] for r in pq.read_table(shard).to_pylist()}
            assert splits == {shard.name.split("-")[0]}, (shard.name, splits)
    assert {r["split"] for r in _rows(out, "bleaching")} == {"validation", "test"}
    assert {r["licence_class"] for r in _rows(out, "bleaching")} == {"open"}


def test_run_records_the_export_counts_in_release_json(registry, build, monkeypatch, tmp_path):
    import argparse

    from marinedata import hf_export
    from marinedata.hf_export_v11 import run

    rel, roots, _ = build
    monkeypatch.setattr(hf_export, "_roots", lambda release_dir, cache: roots)
    skip = tmp_path / "v1.sha256"
    skip.write_text(f"{SKIP_IMG}\n")
    args = argparse.Namespace(
        release_dir=rel, flavour="open", out=tmp_path / "out", summary=tmp_path / "s.json",
        v1_sha256=skip, v1_dir=None,
    )  # fmt: skip
    assert run(args) == 0
    configs = json.loads((rel / "RELEASE.json").read_text())["configs"]
    exported = json.loads((tmp_path / "s.json").read_text())["configs"]
    assert configs == {
        c: {s: e["rows"] for s, e in v["splits"].items()} for c, v in exported.items()
    }
    assert configs["images"] and configs["metadata"]


# PR-2: bytes_config follows the v1.0 tree, so a new image of a v1.0 source ships its bytes

OWN = "reef-support-benthic-own"


def _shard(root: Path, config: str, rows: list[dict], schema: pa.Schema | None = None) -> None:
    path = root / "data" / config / "train-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _own_tree(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """benthic-own (a v1.0 source), staged with its checksums: w1 is in the frozen v1.0
    ``coral-masks``, w2 joined after v1.0."""
    root = tmp_path / "cache" / OWN
    keys = _tree(
        root, {f"images/SEAFLOWER/{s}.JPG": f"{s}-pixels".encode() * 30 for s in ("w1", "w2")}
    )
    shas = {s: keys[f"images/SEAFLOWER/{s}.JPG"] for s in ("w1", "w2")}
    rows = [
        {
            "sample_id": f"{OWN}/{s}", "source_id": OWN, "source_version": "2026-10-06", "stem": s,
            "image_path": f"images/SEAFLOWER/{s}.JPG", "image_sha256": shas[s],
            "license": "CC-BY-4.0", "split_group": f"{OWN}/SEAFLOWER", "partition": "SEAFLOWER",
        }
        for s in ("w1", "w2")
    ]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(rows), root / "metadata.parquet")
    return root, shas


def test_a_new_image_of_a_v1_source_ships_its_bytes_and_the_join_resolves(registry, tmp_path):
    root, shas = _own_tree(tmp_path)
    v1 = tmp_path / "v1"
    _shard(
        v1, "coral-masks", [{"image_sha256": shas["w1"], "image": {"bytes": b"v1", "path": None}}]
    )
    inputs = V11Inputs(
        taxonomy_version=TAX, roots={OWN: root}, split_of=dict.fromkeys(shas.values(), "train"),
        v1_config_shas=load_v1_config_shas(v1, [OWN]),
    )  # fmt: skip
    meta = build_v11_tables(registry, {}, inputs)["metadata"].rows
    config_of = {r["image_sha256"]: r["bytes_config"] for r in meta}
    assert config_of == {shas["w1"]: "coral-masks", shas["w2"]: "images"}  # v1.0 rows untouched

    stats: Counter = Counter()
    out, bytes_sha = image_rows(meta, {OWN: root}, skip_sha256=v1_sha256s(v1), stats=stats)
    assert [row.values["image_sha256"] for _, row in out] == [shas["w2"]]
    assert row_bytes(out[0][1]) == b"w2-pixels" * 30 and bytes_sha == {shas["w2"]: shas["w2"]}
    assert (stats["skipped_v1_config"], stats["skipped_v1_sha256"]) == (1, 0)

    def joined(build: Path, rows: list[dict], images: list[str]):
        _shard(build, "metadata", [{**r, "licence_class": "open"} for r in rows])
        sha_only = pa.schema([("image_sha256", pa.string())])
        _shard(build, "images", [{"image_sha256": s} for s in images], sha_only)
        return run_join_gate(
            build, v1_sha256=v1_sha256s(v1), v1_configs={"coral-masks"}
        )  # fmt: skip

    keep = ("image_sha256", "source_id", "bytes_config")
    ok = joined(tmp_path / "ok", [{k: r[k] for k in keep} for r in meta], [shas["w2"]])
    assert ok.passed, ok.failures
    assert ok.counts.get("failed_unresolved", 0) == 0
    # the old source-keyed stamp: w2 named coral-masks, which has no row for it (36 in run 3)
    old = [{k: r[k] for k in keep} | {"bytes_config": "coral-masks"} for r in meta]
    assert joined(tmp_path / "old", old, []).counts["failed_unresolved"] == 1


def row_bytes(row) -> bytes:
    return Path(row.file).read_bytes()


# PR-2b: mask_config follows the v1.0 tree too, so the mask of a new image of a v1.0 source ships


def test_a_new_image_of_a_v1_source_embeds_its_mask_and_a_v1_image_does_not(registry, tmp_path):
    root, shas = _own_tree(tmp_path)
    (root / "labels").mkdir()  # outside CHECKSUMS: the export finds a mask by its mask_ref
    for value, stem in enumerate(shas, start=1):
        (root / "labels" / f"{stem}.png").write_bytes(_png(value))
    v1 = tmp_path / "v1"
    _shard(
        v1, "coral-masks", [{"image_sha256": shas["w1"], "image": {"bytes": b"v1", "path": None}}]
    )
    inputs = V11Inputs(
        taxonomy_version=TAX, roots={OWN: root}, split_of=dict.fromkeys(shas.values(), "train"),
        v1_config_shas=load_v1_config_shas(v1, [OWN]),
    )  # fmt: skip
    classes = [{"id": i, "label_native": f"class-{i}"} for i in (1, 2)]
    mask = {
        "source_id": OWN, "label_origin": "human", "class_counts": "{}", "label_status": "ok",
        "canonical_class_counts": "{}", "hf_class_map": json.dumps(classes),
    }  # fmt: skip
    labelled = tuple(
        {**mask, "sha256": sha, "mask_key": f"labels/{stem}.png"} for stem, sha in shas.items()
    )
    semseg = ConfigResult("semseg", labelled, {})
    tables = build_v11_tables(registry, {"semseg": semseg}, inputs)
    rel = tmp_path / "rel"
    for name in ("metadata", "semseg"):
        _write(rel, name, [{**r, "licence_class": "open"} for r in tables[name].rows])
    (rel / "task_layers" / "V11_TABLES.json").write_text(json.dumps({"taxonomy_version": TAX}))

    layout, _ = build_v11_layout(
        registry, rel, {OWN: root}, flavour="open", skip_sha256=v1_sha256s(v1)
    )
    export_v11(layout, tmp_path / "out")
    rows = {r["image_sha256"]: r for r in _rows(tmp_path / "out", "semseg")}
    old, new = rows[shas["w1"]], rows[shas["w2"]]
    # w1 is in the frozen v1.0 coral-masks: its stamp is untouched and nothing is re-shipped
    assert (old["mask_config"], old["bytes_config"]) == ("coral-masks", "coral-masks")
    assert old["mask"] is None and old["mask_bytes_sha256"] is None
    # w2 joined after v1.0: no v1.0 config holds it, so its pixels (images) and mask (here) ship
    assert (new["mask_config"], new["bytes_config"]) == (None, "images")
    png = new["mask"]["bytes"]
    assert Image.open(io.BytesIO(png)).mode == "P"
    assert new["mask_bytes_sha256"] == hashlib.sha256(png).hexdigest()
    assert [r["image_sha256"] for r in _rows(tmp_path / "out", "images")] == [shas["w2"]]
    for row in rows.values():  # every mask resolves: bytes here, or a v1.0 config that holds it
        assert (row["mask"] is not None) != bool(row["mask_config"])
        assert check_palette_row(row) == ([], False)
    assert {r["image_sha256"] for r in rows.values() if r["mask_config"]} <= v1_sha256s(v1)


# Run 6: the open export withholds the hidden-test images that v1.0 does not already publish

HIDDEN_NEW, HIDDEN_V1 = "a" * 64, "b" * 64
"""Synthetic digests: the first is new to the public release, the second v1.0 already ships."""


def _hidden_manifest(tmp_path: Path, *digests: str) -> Path:
    path = tmp_path / "hidden-test.json"
    path.write_text(json.dumps({"images": [{"sha256": d} for d in digests]}))
    return path


def _v1_tree(tmp_path: Path, *digests: str) -> Path:
    """A fake v1.0 export whose ``coral-masks`` shard holds the ``image_sha256`` of ``digests``."""
    _shard(tmp_path / "v1", "coral-masks", [{"image_sha256": d} for d in digests])
    return tmp_path / "v1"


def _run_args(build, tmp_path: Path, **kw):
    import argparse

    skip = tmp_path / "v1.sha256"
    skip.write_text(f"{SKIP_IMG}\n")
    fields = {
        "release_dir": build[0], "flavour": "open", "out": tmp_path / "out",
        "summary": tmp_path / "s.json", "v1_sha256": skip, "v1_dir": None,
        "hidden_test": None, "hidden_test_exempt_v1": None, **kw,
    }  # fmt: skip
    return argparse.Namespace(**fields)


def test_drop_hidden_test_matches_either_key_in_any_case_and_records_counts_only():
    tables = {
        "metadata": [
            {"image_sha256": "1" * 64, "bytes_sha256": None},
            {"image_sha256": "2" * 64, "bytes_sha256": "3" * 64},
            {"image_sha256": ("4" * 64).upper(), "bytes_sha256": None},
            {"image_sha256": "5" * 64, "bytes_sha256": "5" * 64},
        ],
        "points": [{"image_sha256": "1" * 64}, {"image_sha256": "5" * 64}],
        "sites": [{"site_id": "x"}],
    }
    drop = {"1" * 64, "3" * 64, "4" * 64, "9" * 64}
    kept, counts = drop_hidden_test(tables, drop)
    assert [r["image_sha256"] for r in kept["metadata"]] == [
        "5" * 64
    ]  # image_sha256 or bytes_sha256
    assert kept["points"] == [{"image_sha256": "5" * 64}] and kept["sites"] == tables["sites"]
    assert counts == {"requested": 4, "images": 3, "rows": {"metadata": 3, "points": 1, "sites": 0}}
    assert len(tables["metadata"]) == 4 and len(tables["points"]) == 2  # the input is not mutated
    assert not [d for d in drop if d in json.dumps(counts)]


def test_the_drop_set_is_the_manifest_minus_what_v1_already_publishes(tmp_path):
    manifest = _hidden_manifest(tmp_path, HIDDEN_NEW, HIDDEN_V1)
    assert hidden_test_drop(manifest, _v1_tree(tmp_path, HIDDEN_V1, "c" * 64)) == {HIDDEN_NEW}


def test_the_drop_set_fails_closed_on_a_wrong_path(tmp_path):
    v1 = _v1_tree(tmp_path, HIDDEN_V1)
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"images": []}))
    with pytest.raises(V11ExportError, match="not found"):
        hidden_test_drop(tmp_path / "missing.json", v1)
    with pytest.raises(V11ExportError, match="no sha256"):
        hidden_test_drop(empty, v1)
    with pytest.raises(V11ExportError, match=r"no v1\.0 shard rows"):
        hidden_test_drop(_hidden_manifest(tmp_path, HIDDEN_NEW), tmp_path / "no-v1")


def test_the_open_layout_drops_a_digest_from_every_table_before_counting(registry, build, tmp_path):
    sha = build[2]
    base, base_report = _layout(registry, build)
    export_v11(base, tmp_path / "base")
    drop = {sha["a"], sha["m"], "f" * 64}
    layout, report = _layout(registry, build, hidden_test_drop=drop)
    out = tmp_path / "out"
    export_v11(layout, out)
    assert "hidden_test_excluded" not in base_report
    for config in ("images", "metadata", "points", "semseg"):
        keys = {r["image_sha256"] for r in _rows(out, config)}
        assert not keys & {sha["a"], sha["m"]}, config
    assert {r["image_sha256"] for r in _rows(out, "images")} == {sha["a2"], sha["b"]}
    hidden = report["hidden_test_excluded"]
    assert (hidden["requested"], hidden["images"]) == (3, 2)
    assert {c: n for c, n in hidden["rows"].items() if n} == {
        "metadata": 2,
        "points": 1,
        "semseg": 1,
    }
    assert not [d for d in drop if d in json.dumps(report)]

    # the sources config counts what is exported: the dropped image no longer counts
    def n_images(path):
        return {r["source_id"]: r["n_images"] for r in _rows(path, "sources")}

    before, after = n_images(tmp_path / "base"), n_images(out)
    assert before["seaview-survey-imagery"] == 3 and after["seaview-survey-imagery"] == 2
    assert after == Counter(r["source_id"] for r in _rows(out, "metadata"))


def test_the_hidden_test_drop_is_refused_for_the_nc_flavour(registry, build):
    with pytest.raises(V11ExportError, match="open flavour"):
        _layout(registry, build, flavour="nc", hidden_test_drop={HIDDEN_NEW})


@pytest.mark.parametrize("flag", ["hidden_test", "hidden_test_exempt_v1"])
def test_run_needs_the_two_hidden_test_flags_together(build, monkeypatch, tmp_path, flag):
    from marinedata import hf_export
    from marinedata.hf_export_v11 import run

    monkeypatch.setattr(hf_export, "_roots", lambda release_dir, cache: build[1])
    args = _run_args(build, tmp_path, **{flag: _hidden_manifest(tmp_path, HIDDEN_NEW)})
    with pytest.raises(V11ExportError, match="--hidden-test-exempt-v1"):
        run(args)
    assert not (tmp_path / "out").exists()


def test_run_refuses_hidden_test_for_a_release_built_as_nc(build, monkeypatch, tmp_path):
    from marinedata import hf_export
    from marinedata.hf_export_v11 import run

    (build[0] / "RELEASE.json").write_text(json.dumps({"flavour": "nc", "release": "v1.1"}))
    monkeypatch.setattr(hf_export, "_roots", lambda release_dir, cache: build[1])
    args = _run_args(
        build, tmp_path, flavour="nc", hidden_test=_hidden_manifest(tmp_path, HIDDEN_NEW),
        hidden_test_exempt_v1=_v1_tree(tmp_path, HIDDEN_V1),
    )  # fmt: skip
    with pytest.raises(V11ExportError, match="open flavour"):
        run(args)
    assert not (tmp_path / "out").exists()


def test_run_withholds_unpublished_hidden_test_images_and_logs_counts_only(
    build, monkeypatch, tmp_path, capsys
):
    from marinedata import hf_export
    from marinedata.hf_export_v11 import run

    _, roots, sha = build
    monkeypatch.setattr(hf_export, "_roots", lambda release_dir, cache: roots)
    # a: new to the public release; b: v1.0 already publishes it; e: not in this build at all
    listed = (sha["a"], sha["b"], "e" * 64)
    args = _run_args(
        build, tmp_path, hidden_test=_hidden_manifest(tmp_path, *listed),
        hidden_test_exempt_v1=_v1_tree(tmp_path, sha["b"]),
    )  # fmt: skip
    assert run(args) == 0
    seen = capsys.readouterr()
    out = tmp_path / "out"
    images = {r["image_sha256"] for r in _rows(out, "images")}
    assert sha["a"] not in images and sha["b"] in images  # the exemption does not feed --v1-dir
    for config in ("metadata", "points"):
        assert sha["a"] not in {r["image_sha256"] for r in _rows(out, config)}, config
    assert sha["b"] in {r["image_sha256"] for r in _rows(out, "metadata")}
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["v11"]["hidden_test_excluded"]["images"] == 1
    log = seen.out + seen.err
    assert "hidden_test_excluded=1 images, 2 rows" in log and "2 not in v1.0" in log
    assert "metadata=1" in log and "points=1" in log
    assert not [d for d in listed if d in log or d in json.dumps(summary["v11"])]


def test_hf_export_main_forwards_the_hidden_test_flags_to_a_v11_release(
    build, monkeypatch, tmp_path
):
    from marinedata import hf_export
    from marinedata import hf_export_v11 as v11

    seen = {}

    def fake_run(args):
        seen.update(
            hidden_test=args.hidden_test, exempt=args.hidden_test_exempt_v1, v1_dir=args.v1_dir
        )
        return 0

    monkeypatch.setattr(v11, "run", fake_run)
    argv = ["--release-dir", str(build[0]), "--out", str(tmp_path / "o")]
    argv += ["--summary", str(tmp_path / "s.json")]
    argv += ["--hidden-test", str(tmp_path / "ht"), "--hidden-test-exempt-v1", str(tmp_path / "v1")]
    assert hf_export.main(argv) == 0
    assert seen == {"hidden_test": tmp_path / "ht", "exempt": tmp_path / "v1", "v1_dir": None}


def test_hf_export_main_refuses_the_hidden_test_flags_for_a_pre_v11_release(tmp_path, capsys):
    from marinedata import hf_export

    argv = ["--release-dir", str(tmp_path), "--out", str(tmp_path / "o")]
    argv += ["--summary", str(tmp_path / "s.json"), "--hidden-test", str(tmp_path / "ht")]
    with pytest.raises(SystemExit) as exc:
        hf_export.main(argv)
    assert exc.value.code == 2 and "v1.1.0 builds" in capsys.readouterr().err


def test_assert_public_safe_refuses_a_non_open_source_id_in_the_open_flavour(monkeypatch):
    from marinedata.gates import v11_runner

    monkeypatch.setattr(v11_runner, "nc_source_ids", lambda *_: frozenset({"ncsrc"}))
    with pytest.raises(V11ExportError, match="nc_id#") as exc:
        assert_public_safe("sources: a, NCSRC, b", "open", where="README.md")
    assert "ncsrc" not in str(exc.value).lower()  # only the label is echoed
    assert_public_safe("ncsrc2, my-ncsrc and ncsrc_v1 are other names", "open")
    assert_public_safe("the nc card names ncsrc", "nc")
