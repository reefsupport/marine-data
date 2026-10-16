"""PR-E (E1/E2): the v1.1.0 export of a synthetic build dir (no bucket, no Hub)."""

from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.hf_export_v11 import (
    V11ExportError,
    assert_public_safe,
    build_v11_layout,
    export_v11,
    is_v11_release,
    read_checksums,
)
from marinedata.registry import Registry

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
