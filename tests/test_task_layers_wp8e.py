"""WP-8e: S3-keyed task-label producers, streamed Coralseg restage, HF task-layer wiring."""

from __future__ import annotations

import io
import json

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
Image = pytest.importorskip("PIL.Image")

from marinedata import checksums  # noqa: E402
from marinedata.hf_card import IMAGES, render_card  # noqa: E402
from marinedata.hf_export import TASK_LAYER_CONFIGS, build_layout  # noqa: E402
from marinedata.task_layers import hf_wiring, s3_keyed  # noqa: E402
from marinedata.task_layers.configs import ConfigResult  # noqa: E402
from marinedata.task_layers.sources import coralseg_stream  # noqa: E402

SHA_A, SHA_B = "a" * 64, "b" * 64


def _parquet(rows: list[dict]) -> bytes:
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), buf)
    return buf.getvalue()


def _png(values: list[list[int]], mode: str = "L") -> bytes:
    import numpy as np

    arr = np.array(values, dtype="uint8")
    if mode == "RGB":
        arr = np.stack([arr, arr * 0, arr * 0], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(arr, mode).save(buf, format="PNG")
    return buf.getvalue()


def _tree(files: dict[str, bytes], images: dict[str, str]) -> s3_keyed.StagedTree:
    manifest = {rel: sha for rel, sha in images.items()}
    manifest.update({rel: "c" * 64 for rel in files})
    blobs = {f"sources/t/v/{rel}": data for rel, data in files.items()}
    blobs["sources/t/v/CHECKSUMS.sha256"] = "".join(checksums.iter_lines(manifest)).encode()

    def fetch(key: str) -> bytes:
        assert "/images/" not in key, "imagery must never be fetched"
        return blobs[key]

    return s3_keyed.StagedTree("t/v", fetch)


def test_sha_index_partitioned_and_flat_layouts():
    text = "".join(
        checksums.iter_lines(
            {"images/p1/x.jpg": SHA_A, "images/y.png": SHA_B, "labels/masks/p1/x.png": "c" * 64}
        )
    )
    assert s3_keyed.sha_index(text) == {("p1", "x"): SHA_A, ("default", "y"): SHA_B}


def test_fetch_small_refuses_imagery():
    with pytest.raises(s3_keyed.ImageFetchRefused):
        s3_keyed.fetch_small("sources/reefolution/v/images/default/a.jpg")


def test_points_keyed_from_checksums_not_images(tmp_path):
    tree = _tree(
        {
            "metadata.parquet": _parquet(
                [{"stem": "a", "partition": "default", "width": 200, "height": 100}]
            ),
            "labels/points.parquet": _parquet(
                [{"stem": "a", "partition": "default", "row": 50, "col": 50, "label": "Rhy"}]
            ),
        },
        {"images/default/a.JPG": SHA_A},
    )
    out = tmp_path / "points.parquet"
    assert s3_keyed.produce_points(tree, "reefolution", out) == 1
    row = pq.read_table(out).to_pylist()[0]
    assert (row["sha256"], row["native_label"], row["x"], row["y"]) == (SHA_A, "Rhy", 0.25, 0.5)


def test_points_on_unstaged_image_raise(tmp_path):
    tree = _tree(
        {
            "metadata.parquet": _parquet(
                [{"stem": "a", "partition": "default", "width": 2, "height": 2}]
            ),
            "labels/points.parquet": _parquet(
                [{"stem": "a", "partition": "default", "row": 1, "col": 1, "label": "X"}]
            ),
        },
        {},
    )
    with pytest.raises(ValueError, match="unstaged"):
        s3_keyed.produce_points(tree, "reefolution", tmp_path / "p.parquet")


def test_image_labels_and_mask_presence_bleaching(tmp_path):
    tree = _tree(
        {
            "labels/image_labels.parquet": _parquet(
                [{"stem": "a", "partition": "p", "label": "CORAL_BL", "confidence": None}]
            ),
            "labels/masks/p/b.png": _png([[0, 1], [2, 2]]),
        },
        {"images/p/a.jpg": SHA_A, "images/p/b.jpg": SHA_B},
    )
    out = tmp_path / "bl.parquet"
    assert s3_keyed.produce_image_labels(tree, "noaa", out) == 1
    assert pq.read_table(out).to_pylist()[0]["native_label"] == "CORAL_BL"
    out2 = tmp_path / "bl2.parquet"
    assert s3_keyed.produce_mask_presence(tree, "rs", {1: "bleached", 2: "non_bleached"}, out2)
    got = {
        (r["sha256"], r["native_label"], r["pixel_count"]) for r in pq.read_table(out2).to_pylist()
    }
    assert got == {(SHA_B, "bleached", 1), (SHA_B, "non_bleached", 2)}


def test_semseg_histogram_drops_ignore_index(tmp_path):
    tree = _tree(
        {"labels/masks/default/a.png": _png([[0, 3], [3, 7]])},
        {"images/default/a.png": SHA_A},
    )
    out = tmp_path / "semseg.parquet"
    assert s3_keyed.produce_semseg(tree, "coralscapes", {3: "sand"}, out) == 1
    row = pq.read_table(out).to_pylist()[0]
    assert json.loads(row["class_counts"]) == {"sand": 2, "__unknown_index_7": 1}


class _FakeS3:
    def __init__(self, objects: dict[str, bytes], bad_size: str | None = None) -> None:
        self.objects, self.put, self.order, self.bad_size = dict(objects), {}, [], bad_size

    def get_paginator(self, _name):
        objects = self.objects

        class _P:
            def paginate(self, Bucket, Prefix):
                keys = sorted(k for k in objects if k.startswith(Prefix))
                return [{"Contents": [{"Key": k, "Size": len(objects[k])} for k in keys]}]

        return _P()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body, ContentLength):
        self.put[Key] = Body
        self.order.append(Key)

    def head_object(self, Bucket, Key):
        n = len(self.put[Key]) + (1 if Key.endswith(self.bad_size or "\0") else 0)
        return {"ContentLength": n}


def _coralseg_objects() -> dict[str, bytes]:
    buf = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buf, format="JPEG")
    pre = "benthic_datasets/mask_labels/Coralseg/train"
    return {
        f"{pre}/Image/s1_0.jpg": buf.getvalue(),
        f"{pre}/Mask/s1_0.png": _png([[0, 1, 1, 0]] * 3, "RGB"),
    }


def test_coralseg_stream_uploads_from_memory_and_heads_every_object(tmp_path):
    s3 = _FakeS3(_coralseg_objects())
    out = coralseg_stream.restage_streaming(s3, tmp_path, version="vtest")
    prefix = "sources/coralseg-ucsd-mosaics/vtest"
    assert out["head_failed"] == [] and out["images"] == 1
    assert s3.order[-1] == f"{prefix}/CHECKSUMS.sha256"
    assert {k.split("/", 3)[3].split("/")[0] for k in s3.put} == {
        "images",
        "labels",
        "metadata.parquet",
        "CHECKSUMS.sha256",
    }
    assert not list((tmp_path / "stage").rglob("*.jpg")), "no image bytes on local disk"
    assert list(out["class_counts"].values()) == [{"0": 6, "1": 6}]


def test_coralseg_stream_reports_size_mismatch(tmp_path):
    s3 = _FakeS3(_coralseg_objects(), bad_size="metadata.parquet")
    out = coralseg_stream.restage_streaming(s3, tmp_path, version="vtest")
    assert out["head_failed"] == ["metadata.parquet"]


def test_layout_entries_place_rows_in_their_image_split():
    rows = {"points": [{"sha256": SHA_A, "source_id": "r", "x": 0.5}, {"sha256": SHA_B}]}
    layout = hf_wiring.layout_entries(rows, {SHA_A: "test"})
    spec, splits = layout["points"]
    assert spec.name == "points" and list(splits) == ["test"]
    assert splits["test"][0].values == {"sha256": SHA_A, "source_id": "r", "x": "0.5"}
    assert set(TASK_LAYER_CONFIGS) >= {"points", "semseg", "benthic-coarse"}


def test_build_layout_accepts_task_layers_and_drops_rows_outside_release():
    layout = build_layout({}, task_layers={"points": [{"sha256": SHA_A}]})
    assert "points" not in layout


def test_render_card_appends_task_layer_table():
    summary = {"configs": {IMAGES: {"splits": {"train": {"rows": 1}}}}}
    release = {
        "release": "v2",
        "split_map_sha256": "x",
        "never_eval_near_dup_excluded": {"count": 0},
        "near_dup": {
            "algorithm": "dhash-64",
            "pil_version": "12.3.0",
            "union_max_hamming": 4,
            "never_eval_exclude_max_hamming": 8,
            "chain_guard_fraction": 0.01,
        },
    }
    results = {"points": ConfigResult("points", ({"sha256": SHA_A, "label_origin": "human"},), {})}
    card = render_card(summary, release, [], pretty_name="RS v2", task_layers=results)
    assert "## Task layers (v2)" in card and "| points |" in card
    assert "## Task layers" not in render_card(summary, release, [], pretty_name="RS v2")


# --- WP-8e-resume: backoff, skip-and-record, resumable runner ------------------------


def _http_error(code: int):
    import urllib.error

    return urllib.error.HTTPError("https://x", code, "err", None, None)


def test_fetch_small_backs_off_with_jitter_then_succeeds():
    calls, sleeps = [], []

    def opener(url, timeout):
        calls.append(url)
        if len(calls) < 3:
            raise _http_error(403)
        return io.BytesIO(b"ok")

    data = s3_keyed.fetch_small("sources/t/v/labels/a.png", opener=opener, sleep=sleeps.append)
    assert data == b"ok" and len(calls) == 3
    assert len(sleeps) == 2 and 0 <= sleeps[0] <= 0.5 and 0 <= sleeps[1] <= 1.0


def test_fetch_small_gives_up_after_eight_tries_with_last_status():
    sleeps: list[float] = []

    def opener(url, timeout):
        raise _http_error(503)

    with pytest.raises(s3_keyed.FetchFailed) as info:
        s3_keyed.fetch_small("sources/t/v/labels/a.png", opener=opener, sleep=sleeps.append)
    assert info.value.status == "503" and info.value.tries == 8
    assert len(sleeps) == 7 and max(sleeps) <= 20.0


def _flaky_tree(bad: str) -> s3_keyed.StagedTree:
    files = {
        "labels/masks/default/a.png": _png([[1, 2]]),
        "labels/masks/default/b.png": _png([[1, 1]]),
    }
    tree = _tree(files, {"images/default/a.jpg": SHA_A, "images/default/b.jpg": SHA_B})
    inner = tree.fetch

    def fetch(key: str) -> bytes:
        if key.endswith(bad):
            raise s3_keyed.FetchFailed(key, 8, "403")
        return inner(key)

    tree.fetch = fetch
    return tree


def test_semseg_skips_and_records_a_mask_that_never_fetches(tmp_path):
    tree = _flaky_tree("b.png")
    n = s3_keyed.produce_semseg(tree, "t", {1: "x", 2: "y"}, tmp_path / "s.parquet")
    assert n == 1
    assert pq.read_table(tmp_path / "s.parquet").column("sha256").to_pylist() == [SHA_A]
    assert tree.missing == [("sources/t/v/labels/masks/default/b.png", "403")]


def _runner():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "tasklabels_s3.py"
    spec = importlib.util.spec_from_file_location("tasklabels_s3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_records_missing_keys_and_resumes(tmp_path, monkeypatch):
    mod = _runner()
    tree = _flaky_tree("b.png")
    monkeypatch.setattr(mod, "JOBS", {"t": ("t/v", "semseg", "semseg", {"id_to_label": {1: "x"}})})
    first = mod.run("t", tmp_path, fetch=tree.fetch)
    assert first["rows"] == 1 and first["missing"] == 1
    tsv = (tmp_path / "missing_keys.tsv").read_text().splitlines()
    assert tsv == [
        "task\tkey\tlast_status",
        "t/semseg\tsources/t/v/labels/masks/default/b.png\t403",
    ]
    assert mod.run("t", tmp_path, fetch=tree.fetch)["skipped"] == "complete"
    assert not list(tmp_path.rglob("*.partial"))


def test_runner_fails_one_job_on_a_dead_required_file_and_records_it(tmp_path, monkeypatch):
    mod = _runner()
    tree = _flaky_tree("CHECKSUMS.sha256")
    monkeypatch.setattr(mod, "JOBS", {"t": ("t/v", "semseg", "semseg", {})})
    out = mod.run("t", tmp_path, fetch=tree.fetch)
    assert "failed" in out and not (tmp_path / "t" / "semseg.parquet").exists()
    assert "CHECKSUMS.sha256\t403" in (tmp_path / "missing_keys.tsv").read_text()


# --- WP-8e-resume: widened sources, bleaching config, export CLI passes task layers ----


def _write_tasklabels(base, source_id: str, task: str, rows: list[dict]) -> None:
    path = base / "_tasklabels" / source_id / f"{task}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture(scope="module")
def registry():
    from marinedata.registry import Registry

    return Registry.load()


def test_bleaching_config_projects_condition_and_tallies_unmapped(tmp_path, registry):
    from marinedata.task_layers import configs

    base = {"source_id": "reef-support-bleaching", "label_origin": "human", "evidence": "mask"}
    _write_tasklabels(
        tmp_path,
        "reef-support-bleaching",
        "bleaching",
        [
            {**base, "sha256": SHA_A, "native_label": "bleached", "pixel_count": 5},
            {**base, "sha256": SHA_A, "native_label": "non_bleached", "pixel_count": 9},
            {**base, "sha256": SHA_B, "native_label": "not-a-label", "pixel_count": 1},
        ],
    )
    result = configs.build_all_configs(registry, tmp_path)["bleaching"]
    by_label = {r["native_label"]: r for r in result.rows}
    assert by_label["bleached"]["bleaching_condition"] == "BLEACHED"
    assert by_label["bleached"]["coral_health"] == "UNHEALTHY"
    assert by_label["non_bleached"]["bleaching_condition"] == "HEALTHY"
    assert by_label["not-a-label"]["canonical_condition"] is None
    assert result.unmapped_by_source["reef-support-bleaching"] == pytest.approx(1 / 3)
    assert "bleaching" in configs.CONFIG_IDS and "bleaching" in TASK_LAYER_CONFIGS


def test_points_and_semseg_configs_read_mermaid_and_own_masks(tmp_path, registry):
    from marinedata.task_layers import configs

    point = {"label_origin": "human", "x": 0.5, "y": 0.5, "source_id": "mermaid-aws"}
    _write_tasklabels(
        tmp_path, "mermaid-aws", "points", [{**point, "sha256": SHA_A, "native_label": "Rubble"}]
    )
    _write_tasklabels(
        tmp_path,
        "reef-support-benthic-own",
        "semseg",
        [
            {
                "sha256": SHA_B,
                "source_id": "reef-support-benthic-own",
                "label_origin": "human",
                "mask_key": "labels/masks/S/b.png",
                "class_counts": json.dumps({"Hard Coral": 3, "Soft Coral": 1}),
            }
        ],
    )
    out = configs.build_all_configs(registry, tmp_path)
    (prow,) = out["points"].rows
    assert prow["source_id"] == "mermaid-aws" and prow["taxon_node_id"] is not None
    (srow,) = out["semseg"].rows
    assert json.loads(srow["canonical_class_counts"]) and srow["source_id"].startswith("reef-")
    cover = {r["sha256"]: r for r in out["benthic-cover"].rows}
    assert set(cover) == {SHA_A, SHA_B}
    assert out["semseg"].unmapped_by_source["reef-support-benthic-own"] == 0.0


def test_hf_export_cli_passes_release_task_layers_to_build_layout(tmp_path, monkeypatch):
    from marinedata import hf_export

    release = tmp_path / "rel"
    (release / "task_layers").mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"sha256": SHA_A, "source_id": "s", "native_label": "bleached"}]),
        release / "task_layers" / "bleaching.parquet",
    )
    pq.write_table(pa.table({}), release / "task_layers" / "vqa.parquet")  # column-less
    seen: dict = {}

    def fake_build_layout(rows, pseudo, task_layers=None, flavour=None):
        seen["task_layers"] = task_layers
        return {}

    monkeypatch.setattr(hf_export, "_roots", lambda release_dir, cache: {})
    monkeypatch.setattr(hf_export, "collect_rows", lambda *a: {})
    monkeypatch.setattr(hf_export, "build_layout", fake_build_layout)
    monkeypatch.setattr(
        hf_export, "export", lambda layout, out, sample: {"file_count": 0, "configs": {}}
    )
    argv = ["--release-dir", str(release), "--out", str(tmp_path / "o")]
    assert hf_export.main([*argv, "--summary", str(tmp_path / "s.json")]) == 0
    assert list(seen["task_layers"]) == ["bleaching"]
    assert seen["task_layers"]["bleaching"][0]["sha256"] == SHA_A
