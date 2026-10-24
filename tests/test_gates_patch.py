"""Patch-aware release gates (``release gate --patch-of``): v1.1.1 ``condition-masks``.

Each patch path has a passing and a failing case; the last tests show a full (non-patch) release
is gated exactly as before. Offline: the prior release is a card + a ``{path: sha256}`` listing."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")
pytest.importorskip("datasets")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from marinedata import condition_masks as cm
from marinedata import condition_masks_docs as docs
from marinedata.cli import main
from marinedata.gates import GATES_FILE, merge_gates_json, publish_verdict
from marinedata.gates import condition_masks as cmg
from marinedata.gates.card import run_card_gate
from marinedata.gates.load import run_load_gate
from marinedata.gates.masks import run_masks_gate
from marinedata.gates.patch_base import build_base, load_patch_base
from marinedata.gates.patch_checks import (
    has_image_bytes,
    inherit_image_gate,
    run_patch_splits_gate,
)
from marinedata.gates.patch_labelmap import run_patch_labelmap_gate
from marinedata.gates.patch_runner import PATCH_ALL, PatchOptions, run_patch_gate
from marinedata.hf_export_v11 import export_v11
from marinedata.registry import Registry
from marinedata.task_layers import masks_table as mt

STEMS = ("a", "b")
SHAS = {s: hashlib.sha256(s.encode()).hexdigest() for s in STEMS}
PRIOR_CARD = """---
size_categories:
- 100K<n<1M
configs:
- config_name: semseg
  data_files:
  - split: train
    path: data/semseg/train-*.parquet
- config_name: instances
  data_files: []
dataset_info:
- config_name: semseg
  features: []
- config_name: instances
  features: []
---
## Dataset summary
| Configurations | 11: `images`, `semseg`, `instances` |
| Version | v1.1.0, built 2026-10-17 |
## Dataset structure
- `semseg` (45 columns): `a`.
- `instances` (40 columns): `a`.
## Label space
## Versions and changelog
| v1.1.0 | 2026-10-17 | Additive: 11 new configurations. |
  version = {1.1.0},
"""
PRIOR_TREE = {
    "README.md": None,
    "data/semseg/train-x-00000-of-00001.parquet": "a" * 64,
    "data/coral-masks/train-x-00000-of-00001.parquet": "b" * 64,
}


def _png(values) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.array(values, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def patch(tmp_path):
    """``(build, release_dir, prior_dir, staged, v1_dir, split_map)`` of a two-raster patch."""
    root = tmp_path / "tree"
    (root / "labels" / "masks").mkdir(parents=True)
    lines = []
    for stem in STEMS:
        (root / "labels" / "masks" / f"{stem}.png").write_bytes(_png([[0, 1, 2], [2, 1, 0]]))
        lines.append(f"{SHAS[stem]}  images/{stem}.jpg")
        lines.append(f"{hashlib.sha256(stem.encode() + b'm').hexdigest()}  labels/masks/{stem}.png")
    (root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    meta = [{"partition": None, "stem": s, "upstream_split": None, "width": 3, "height": 2}
            for s in STEMS]  # fmt: skip
    pq.write_table(pa.Table.from_pylist(meta), root / "metadata.parquet")
    staged = cm.local_tree(root, mt.CONDITION_MASK_SOURCES[cm.SOURCE_ID].tree)
    layout, _ = cm.build_condition_layout(
        Registry.load(), staged, {sha: "train" for sha in SHAS.values()}
    )
    build = tmp_path / "build"
    export_v11(layout, build)
    (build / "README.md").write_text(
        docs.patch_card(PRIOR_CARD, cm.COLUMNS, rows=len(STEMS), date="2026-10-24")
    )
    release = tmp_path / "release"
    release.mkdir()
    (release / "RELEASE.json").write_text(
        json.dumps({"configs": {"condition-masks": {"train": len(STEMS)}}})
    )
    prior = tmp_path / "prior"
    prior.mkdir()
    (prior / "README.md").write_text(PRIOR_CARD)
    v1 = tmp_path / "v1"
    (v1 / "data" / "coral-masks").mkdir(parents=True)
    pq.write_table(
        pa.table({"image_sha256": list(SHAS.values())}),
        v1 / "data" / "coral-masks" / "train-x-00000-of-00001.parquet",
    )
    smap = tmp_path / "SPLIT_MAP.json"
    smap.write_text(
        json.dumps({"assignments": {f"{cm.SOURCE_ID}/sha:{sha}": "train" for sha in SHAS.values()}})
    )
    return build, release, prior, staged, v1, smap


def _base(prior_dir, tree=None):
    return build_base("v1.1.0", PRIOR_TREE if tree is None else tree, prior_dir / "README.md")


def _shard(build):
    return next((build / "data" / "condition-masks").glob("*.parquet"))


def _rewrite(build, **columns):
    """The condition-masks shard with some columns replaced by a constant (the string cells)."""
    path = _shard(build)
    table = pq.read_table(path)
    for name, value in columns.items():
        index = table.schema.get_field_index(name)
        table = table.set_column(index, name, pa.array([value] * table.num_rows, pa.string()))
    pq.write_table(table, path)


# ------------------------------------------------------------------ G5 masks: condition-masks


def test_masks_gate_recognises_condition_masks(patch):
    build = patch[0]
    result = run_masks_gate(build)
    assert result.passed and result.counts["condition_rows"] == len(STEMS)


@pytest.mark.parametrize(
    ("column", "value", "kind"),
    [
        ("class_map", '{"1": "bleached"}', "failed_unknown_class_id"),  # the mask also paints 2
        ("class_map", '{"1": "a", "255": "b", "2": "c"}', "failed_class_map"),
        ("class_map", "not json", "failed_class_map"),
        ("ignore_value", "none", "failed_ignore_value"),
        ("mask_bytes_sha256", "0" * 64, "failed_mask_sha256"),
    ],
)
def test_masks_gate_fails_a_bad_condition_row(patch, column, value, kind):
    build = patch[0]
    _rewrite(build, **{column: value})
    result = run_masks_gate(build)
    assert not result.passed and result.counts[kind] == len(STEMS)


def test_patch_masks_also_check_the_raster_size_against_the_staged_metadata(patch):
    build, release, prior, _, v1, _ = patch
    options = PatchOptions(base=_base(prior), release_dir=release, v1_dir=v1, staged_root=None)
    assert "no_staged_root" in run_patch_gate("masks", build, options).failures[0]
    good = replace(options, staged_root=staged_root(patch))
    assert run_patch_gate("masks", build, good).passed


def staged_root(patch):
    return patch[0].parent / "tree"


def test_patch_masks_fail_when_a_raster_is_not_the_staged_size(patch):
    build, _, prior, _, v1, _ = patch
    meta = pq.read_table(staged_root(patch) / "metadata.parquet").to_pylist()
    pq.write_table(
        pa.Table.from_pylist([{**r, "width": 9} for r in meta]),
        staged_root(patch) / "metadata.parquet",
    )
    options = PatchOptions(base=_base(prior), staged_root=staged_root(patch), v1_dir=v1)
    result = run_patch_gate("masks", build, options)
    assert not result.passed and result.counts["sizes_failed_size"] == len(STEMS)


# ------------------------------------------------------------------ frozen: root GATES.json


def test_frozen_allows_only_the_gates_json_that_release_gate_writes(patch):
    build = patch[0]
    (build / GATES_FILE).write_text("{}")
    assert cmg.run_frozen_gate(build, prior=PRIOR_TREE).passed
    (build / "NOTES.txt").write_text("x")
    result = cmg.run_frozen_gate(build, prior=PRIOR_TREE)
    assert not result.passed and "NOTES.txt" in result.failures[0]


# ------------------------------------------------------------------ card against the Hub card


def test_card_gate_freezes_the_prior_configs(patch):
    build, release, prior, *_ = patch
    base = _base(prior)
    result = run_card_gate(build, release_dir=release, prior=base)
    assert result.passed, result.failures
    assert result.counts["configs_from_v1_dir"] == 2


@pytest.mark.parametrize(
    ("old", "new", "kind"),
    [
        (
            "path: data/semseg/train-*.parquet",
            "path: data/semseg/other-*.parquet",
            "frozen_card_entry",
        ),
        ("- 100K<n<1M\n", "- 1K<n<10K\n", "size_category"),
        ("- config_name: instances\n  data_files: []\n", "", "frozen_card_entry"),
    ],
)
def test_card_gate_fails_a_changed_prior_entry(patch, old, new, kind):
    build, release, prior, *_ = patch
    card = build / "README.md"
    text = card.read_text()
    assert old in text
    card.write_text(text.replace(old, new, 1))
    result = run_card_gate(build, release_dir=release, prior=_base(prior))
    assert result.counts[f"failed_{kind}"] >= 1 and not result.passed


def test_card_gate_without_prior_is_the_stock_gate(patch):
    """A full release (no `prior`) still fails a card that lists configs the build lacks."""
    build, release, *_ = patch
    result = run_card_gate(build, release_dir=release)
    assert not result.passed and result.counts["failed_extra_config_section"] == 2


# ------------------------------------------------------------------ load: prior configs frozen


def test_load_gate_loads_only_the_patch_config(patch):
    build, release, prior, *_ = patch
    result = run_load_gate(build, release_dir=release, prior=_base(prior))
    assert result.passed, result.failures
    assert result.counts["configs"] == 1 and result.counts["skipped_frozen_prior"] == 2


def test_load_gate_fails_a_wrong_patch_row_count(patch):
    build, release, prior, *_ = patch
    (release / "RELEASE.json").write_text(
        json.dumps({"configs": {"condition-masks": {"train": 9}}})
    )
    result = run_load_gate(build, release_dir=release, prior=_base(prior))
    assert not result.passed and result.counts["failed_count_mismatch"] == 1


def test_load_gate_without_prior_is_the_stock_gate(patch):
    build, release, *_ = patch
    result = run_load_gate(build, release_dir=release)
    assert not result.passed and result.counts["failed_config_files_missing"] >= 1


# ------------------------------------------------------------------ dedup / privacy inheritance


def test_dedup_and_privacy_are_inherited_only_for_a_patch_without_pixels(patch, tmp_path):
    build, _, prior, *_ = patch
    base = _base(prior)
    assert not has_image_bytes(build)
    stock = []
    result = inherit_image_gate("dedup", build, base, lambda: stock.append(1))
    assert result.passed and result.counts["inherited_from"] == "v1.1.0" and not stock
    pixels = tmp_path / "pixels"
    (pixels / "data" / "images").mkdir(parents=True)
    pq.write_table(
        pa.table({"image": [b"x"]}), pixels / "data" / "images" / "train-00000-of-00001.parquet"
    )
    assert has_image_bytes(pixels)
    sentinel = object()
    assert inherit_image_gate("privacy", pixels, base, lambda: sentinel) is sentinel


# ------------------------------------------------------------------ splits


def test_patch_splits_pass_and_name_both_verdicts(patch):
    build, release, prior, _, v1, smap = patch
    options = PatchOptions(base=_base(prior), v1_dir=v1, split_map=smap, release_dir=release)
    for name in ("splits", "splits_v11"):
        result = run_patch_gate(name, build, options)
        assert result.name == name and result.passed and result.counts["rows"] == len(STEMS)


def test_patch_splits_fail_on_a_map_mismatch_a_reassignment_and_a_missing_map(patch, tmp_path):
    build, _, _, _, v1, smap = patch
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"assignments": {f"{cm.SOURCE_ID}/sha:{SHAS['a']}": "test"}}))
    result = run_patch_splits_gate(build, split_map=other, v1_dir=v1)
    assert result.counts["failed_map_mismatch"] == len(STEMS)
    (v1 / "data" / "coral-masks" / "train-x-00000-of-00001.parquet").rename(
        v1 / "data" / "coral-masks" / "test-x-00000-of-00001.parquet"
    )
    assert (
        run_patch_splits_gate(build, split_map=smap, v1_dir=v1).counts["failed_reassigned_v1"] == 2
    )
    assert (
        run_patch_splits_gate(build, split_map=None, v1_dir=v1).counts["failed_no_split_map"] == 1
    )


# ------------------------------------------------------------------ labelmap


def test_patch_labelmap_resolves_the_condition_nodes(patch):
    result = run_patch_labelmap_gate(patch[0])
    assert result.passed, result.failures
    assert result.counts["rows"] == len(STEMS)


def test_patch_labelmap_fails_an_unknown_node_and_a_pixel_mismatch(patch):
    build = patch[0]
    attrs = json.dumps(
        {"class_resolution": {"bleached": {"pixel": 1, "condition_node_id": "NOPE"}}}
    )
    _rewrite(build, class_map='{"1": "bleached"}', attrs=attrs)
    assert run_patch_labelmap_gate(build).counts["failed_unknown_node"] == 1
    attrs = json.dumps(
        {"class_resolution": {"bleached": {"pixel": 7, "condition_node_id": "BLEACHED"}}}
    )
    _rewrite(build, class_map='{"1": "bleached"}', attrs=attrs)
    assert run_patch_labelmap_gate(build).counts["failed_class_map"] == 1
    _rewrite(build, attrs="{}")
    assert run_patch_labelmap_gate(build).counts["failed_class_resolution"] == 1


# ------------------------------------------------------------------ join / decon / bytes_once


def test_patch_join_decon_and_bytes_once(patch, monkeypatch):
    build, _, prior, _, v1, _ = patch
    options = PatchOptions(base=_base(prior), v1_dir=v1, hidden_test=(build,))
    assert run_patch_gate("join", build, options).passed
    assert run_patch_gate("bytes_once", build, options).passed
    assert not run_patch_gate("join", build, replace(options, v1_dir=None)).passed
    monkeypatch.setattr("marinedata.hf_export_v11.hidden_test_drop", lambda m, v: {"z" * 64})
    assert run_patch_gate("decon", build, options).passed
    monkeypatch.setattr("marinedata.hf_export_v11.hidden_test_drop", lambda m, v: {SHAS["a"]})
    assert not run_patch_gate("decon", build, options).passed
    assert not run_patch_gate("decon", build, replace(options, hidden_test=())).passed


def test_a_gate_that_raises_is_a_failed_gate(patch, monkeypatch):
    from marinedata.gates import patch_runner

    def boom(build, options):
        raise RuntimeError("boom")

    monkeypatch.setitem(patch_runner.RUNNERS, "load", boom)
    build, _, prior, *_ = patch
    result = run_patch_gate("load", build, PatchOptions(base=_base(prior)))
    assert not result.passed and "gate_error" in result.failures[0]


# ------------------------------------------------------------------ the verdict publish reads


def test_patch_gates_cover_everything_publish_requires(tmp_path):
    results = [
        replace(run_patch_gate("patch_base", tmp_path, PatchOptions(base=_base_stub())), name=n)
        for n in PATCH_ALL
    ]
    merge_gates_json(tmp_path / GATES_FILE, results)
    required, problems = publish_verdict(tmp_path)
    assert not problems and set(required) <= set(PATCH_ALL)


def test_one_failing_patch_gate_blocks_publish(tmp_path):
    ok = run_patch_gate("patch_base", tmp_path, PatchOptions(base=_base_stub()))
    results = [replace(ok, name=n) for n in PATCH_ALL if n != "labelmap"]
    merge_gates_json(tmp_path / GATES_FILE, results)
    assert publish_verdict(tmp_path)[1] == ["gate labelmap: missing"]


def _base_stub():
    return build_base("v1.1.0", PRIOR_TREE, _card_file())


def _card_file():
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "README.md"
    path.write_text(PRIOR_CARD)
    return path


# ------------------------------------------------------------------ the prior release


def test_prior_release_offline_and_anonymous(patch, tmp_path, monkeypatch):
    _, _, prior, *_ = patch
    listing = tmp_path / "tree.json"
    listing.write_text(json.dumps(PRIOR_TREE))
    base = load_patch_base("v1.1.0", prior_dir=prior, tree_json=listing)
    assert base.configs == {"semseg", "instances", "coral-masks"}
    assert base.size_categories == ("100K<n<1M",)

    seen = {}

    class FakeApi:
        def __init__(self, **kw):
            seen.update(kw)

        def list_repo_tree(self, *a, **kw):
            return []

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    cmg.hub_tree("o/r", "v1.1.0")
    assert seen == {"token": False}  # a stored login is never used


# ------------------------------------------------------------------ CLI


def test_cli_patch_of_refuses_when_the_prior_release_cannot_be_read(patch, capsys):
    build, _, prior, *_ = patch
    argv = ["release", "gate", str(build), "--patch-of", "v1.1.0"]
    rc = main([*argv, "--patch-prior-dir", str(prior / "nope"),
               "--patch-prior-tree-json", str(prior / "nope.json")])  # fmt: skip
    assert rc == 1 and "cannot read v1.1.0" in capsys.readouterr().err
    assert not (build / GATES_FILE).exists()


def test_cli_patch_run_writes_every_verdict(patch, tmp_path, capsys):
    build, release, prior, _staged, v1, smap = patch
    listing = tmp_path / "tree.json"
    listing.write_text(json.dumps(PRIOR_TREE))
    argv = ["release", "gate", str(build), "--patch-of", "v1.1.0", "--all"]
    argv += ["--patch-prior-dir", str(prior), "--patch-prior-tree-json", str(listing)]
    argv += ["--v1-dir", str(v1), "--release-dir", str(release), "--split-map", str(smap)]
    main([*argv, "--staged-root", str(staged_root(patch))])
    gates = json.loads((build / GATES_FILE).read_text())
    assert set(PATCH_ALL) <= set(gates)
    assert gates["frozen"]["passed"] and gates["join"]["passed"] and gates["masks"]["passed"]
    assert gates["card"]["passed"] and gates["load"]["passed"] and gates["labelmap"]["passed"]
    assert "publish verdict" in capsys.readouterr().out


def test_without_patch_of_the_stock_flow_is_untouched(patch, capsys):
    """No `--patch-of`: the v1.0 `--all` path runs (stock gates, no patch entries in GATES.json)."""
    build = patch[0]
    main(["release", "gate", str(build), "--licence"])
    gates = json.loads((build / GATES_FILE).read_text())
    assert set(gates) <= {"licence", "summary"}
    capsys.readouterr()
