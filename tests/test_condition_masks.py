"""v1.1.1 ``condition-masks`` patch: rows, hidden-test drop, unchanged v1.1.0 attrs, card, gates."""

from __future__ import annotations

import hashlib
import io
import json

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from marinedata import condition_masks as cm
from marinedata import condition_masks_docs as docs
from marinedata.gates import condition_masks as gates
from marinedata.hf_export_v11 import V11ExportError, export_v11
from marinedata.registry import Registry
from marinedata.task_layers import masks_table as mt

STEMS = ("a", "b", "c")
SHAS = {s: hashlib.sha256(s.encode()).hexdigest() for s in STEMS}


def _png(values: list[list[int]]) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.array(values, dtype=np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def staged(tmp_path):
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
    return cm.local_tree(root, mt.CONDITION_MASK_SOURCES[cm.SOURCE_ID].tree)


def _layout(staged, hidden=()):
    splits = {sha: "train" for sha in SHAS.values()}
    return cm.build_condition_layout(Registry.load(), staged, splits, hidden_test_drop=hidden)


def test_three_rasters_one_hidden_test_gives_two_rows(staged):
    layout, report = _layout(staged, hidden=[SHAS["b"]])
    groups = layout[cm.CONFIG][1]
    assert list(groups) == ["train-reef-support-bleaching"]
    rows = groups["train-reef-support-bleaching"]
    assert {r.values["image_sha256"] for r in rows} == {SHAS["a"], SHAS["c"]}
    assert (report["rasters"], report["rows"], report["hidden_test_excluded"]) == (3, 2, 1)
    attrs = json.loads(rows[0].values["attrs"])["class_resolution"]
    assert attrs["bleached"]["condition_node_id"] == "BLEACHED"
    assert attrs["non_bleached"]["condition_node_id"] == "HEALTHY"
    assert rows[0].values["mask_config"] is None


def test_split_comes_from_the_map_only(staged):
    with pytest.raises(V11ExportError, match="SPLIT_MAP"):
        cm.build_condition_layout(Registry.load(), staged, {})


def test_export_writes_only_the_new_folder_and_gates_pass(staged, tmp_path):
    layout, _ = _layout(staged, hidden=[SHAS["b"]])
    out = tmp_path / "out"
    summary = export_v11(layout, out)
    assert all(
        n.startswith("data/condition-masks/train-reef-support-bleaching-") for n in summary["files"]
    )
    sizes = gates.staged_sizes(staged)
    assert gates.run_masks_gate(out, sizes=sizes).passed
    assert gates.run_join_gate(out, v1_shas=set(SHAS.values())).passed
    assert not gates.run_join_gate(out, v1_shas={SHAS["a"]}).passed
    assert gates.run_decon_gate(out, hidden=[SHAS["b"]]).passed
    assert not gates.run_decon_gate(out, hidden=[SHAS["a"]]).passed
    assert not gates.run_masks_gate(out, sizes={k: (9, 9) for k in sizes}).passed


def test_frozen_gate_against_the_prior_tree(tmp_path):
    shard = tmp_path / "data" / "semseg" / "train-x-00000-of-00001.parquet"
    new = tmp_path / "data" / "condition-masks" / "train-y-00000-of-00001.parquet"
    for path, body in ((shard, b"old"), (new, b"new")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    (tmp_path / "README.md").write_text("card")
    prior = {"data/semseg/train-x-00000-of-00001.parquet": hashlib.sha256(b"old").hexdigest()}
    assert gates.run_frozen_gate(tmp_path, prior=prior).passed
    assert not gates.run_frozen_gate(tmp_path, prior={k: "0" * 64 for k in prior}).passed
    (tmp_path / "data" / "semseg" / "train-z-00000-of-00001.parquet").write_bytes(b"x")
    assert not gates.run_frozen_gate(tmp_path, prior=prior).passed
    assert not gates.run_frozen_gate(tmp_path, prior={}).passed


def test_v110_rebuild_is_unchanged():
    """The new source is outside every v1.1.0 source list and `condition_node_id` is opt-in."""
    assert cm.SOURCE_ID not in mt.MASK_SOURCES
    spec = mt.MASK_SOURCES["reef-support-benthic-own"]
    kw = {"spec": spec, "resolve": lambda _l: (None, "unmapped", None, None), "ordinal": 0,
          "image_sha256": "a" * 64, "mask_ref": "k", "counts": {1: 3, 2: 1}}  # fmt: skip
    plain = mt.semantic_row(**kw)
    with_none = mt.semantic_row(**kw, condition=lambda _l: None)
    assert plain == with_none
    assert "condition_node_id" not in plain["attrs"]


CARD = """---
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


def test_card_patch_adds_the_config_and_refuses_a_second_run():
    out = docs.patch_card(CARD, cm.COLUMNS, rows=634, date="2026-10-24")
    assert out.count("config_name: condition-masks") == 2
    assert "| Configurations | 12: `images`, `semseg`, `condition-masks`, `instances` |" in out
    assert "| v1.1.1 | 2026-10-24 |" in out and "version = {1.1.1}" in out
    assert "pixel `255` is ignore" in out.replace("Pixel `255`", "pixel `255`")
    with pytest.raises(V11ExportError):
        docs.patch_card(out, cm.COLUMNS, rows=634, date="2026-10-24")
    with pytest.raises(V11ExportError, match="anchor"):
        docs.patch_card(CARD.replace("## Label space", ""), cm.COLUMNS, rows=1, date="d")


def test_lock_and_citation_patch():
    lock = docs.patch_lock({"release_id": "v1.1.0", "configs": ["semseg"]}, created="t")
    assert lock == {
        "release_id": "v1.1.1",
        "configs": ["semseg", "condition-masks"],
        "created": "t",
    }
    cff = docs.patch_citation('version: "1.0.0"\ndate-released: "2026-10-07"\n', "2026-10-24")
    assert 'version: "1.1.1"' in cff and 'date-released: "2026-10-24"' in cff
