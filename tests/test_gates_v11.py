"""v1.1.0 gates G1-G7 and the G8 packaging pins: a passing and a failing synthetic case each."""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from marinedata.gates import FAIL_CLOSED, REQUIRED, V11_REQUIRED, publish_verdict, yunet_model
from marinedata.gates.bytes_once import run_bytes_once_gate
from marinedata.gates.decon_gate import run_decon_gate
from marinedata.gates.frozen import run_frozen_gate
from marinedata.gates.join import run_join_gate
from marinedata.gates.leak import run_leak_gate
from marinedata.gates.masks import run_masks_gate
from marinedata.gates.splits_v11 import run_splits_v11_gate
from marinedata.task_layers.mask_classmap import palette_json, palette_rgb

ROOT = Path(__file__).resolve().parents[1]
SHAS = [hashlib.sha256(f"img{i}".encode()).hexdigest() for i in range(4)]
SPLIT_OF = {0: "train", 1: "validation", 2: "test", 3: "train"}
NC_REPO = "marine-data-nc"


def card(configs: dict[str, str]) -> str:
    body = "".join(
        f"- config_name: {n}\n  data_files:\n  - split: train\n    path: {p}\n"
        for n, p in configs.items()
    )
    return f"---\nconfigs:\n{body}---\n# Card\n"


V1_CARD = card({"coralseg": "data/coralseg/train-*.parquet"})


def write_shard(root: Path, config: str, split: str, rows: list[dict], shard: int = 0) -> Path:
    path = root / "data" / config / f"{split}-{shard:05d}-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def write_by_split(root: Path, config: str, rows: list[dict], split_of) -> None:
    for split in sorted({split_of(r) for r in rows}):
        write_shard(root, config, split, [r for r in rows if split_of(r) == split])


def write_v1(tmp_path: Path) -> Path:
    v1 = tmp_path / "v1"
    write_shard(
        v1,
        "coralseg",
        "train",
        [{"image_sha256": SHAS[0], "image": {"bytes": b"v1", "path": None}}],
    )
    (v1 / "README.md").write_text(V1_CARD)
    return v1


def build_release(
    tmp_path: Path, *, split_of: dict | None = None, group_of: dict | None = None
) -> Path:
    """A small v1.1.0 build: img0 is a v1.0 image (bytes stay in coralseg), img1-3 are new."""
    build, splits, groups = tmp_path / "build", {**SPLIT_OF, **(split_of or {})}, group_of or {}
    sha_split = {SHAS[i]: splits[i] for i in range(4)}
    digest = lambda i: hashlib.sha256(f"bytes{i}".encode()).hexdigest()  # noqa: E731
    new = [
        {"image_sha256": SHAS[i], "image": {"bytes": f"bytes{i}".encode(), "path": None},
         "bytes_sha256": digest(i), "source_id": "srcA", "source_ids": ["srcA"],
         "split_group": groups.get(i, f"g{i}"), "split": splits[i]}
        for i in (1, 2, 3)
    ]  # fmt: skip
    meta = [
        {"image_sha256": SHAS[i], "source_id": "srcA", "licence_class": "open",
         "split_group": groups.get(i, f"g{i}"), "bytes_config": "images" if i else "coralseg",
         "bytes_sha256": digest(i), "split": splits[i], "taxonomy_version": "1"}
        for i in range(4)
    ]  # fmt: skip
    points = [
        {"image_sha256": SHAS[i], "source_id": "srcA", "label_native": "x", "label_native_id": "1",
         "label_set": "s", "taxon_node_id": "n", "match_type": "exact", "fidelity": "exact",
         "upstream_split": None, "bytes_config": "images" if i else "coralseg"}
        for i in range(4)
    ]  # fmt: skip
    write_by_split(build, "images", new, lambda r: r["split"])
    write_by_split(build, "metadata", meta, lambda r: r["split"])
    write_by_split(build, "points", points, lambda r: sha_split[r["image_sha256"]])
    (build / "README.md").write_text(
        card({"coralseg": "data/coralseg/train-*.parquet", "images": "data/images/train-*.parquet"})
    )
    (build / "task_layers").mkdir()
    write_report(build)
    return build


def write_report(build: Path, **fields) -> None:
    report = {"unsplit_images": 0, "allow_unsplit": False, **fields}
    (build / "task_layers" / "V11_TABLES.json").write_text(json.dumps(report))


def failed(result, kind: str) -> int:
    return result.counts.get(f"failed_{kind}", 0)


# G1 join


def test_join_passes_a_fully_joined_build(tmp_path):
    result = run_join_gate(build_release(tmp_path), v1_sha256={SHAS[0]}, v1_configs={"coralseg"})
    assert result.passed, result.failures
    assert result.counts["points_joined"] == 4


def test_join_fails_an_orphan_row_and_an_unresolved_v1_image(tmp_path):
    build = build_release(tmp_path)
    write_shard(
        build, "points", "train", [{"image_sha256": "f" * 64, "source_id": "srcA"}], shard=1
    )
    result = run_join_gate(build, v1_sha256=set())
    assert not result.passed
    assert failed(result, "orphan_row") == 1 and failed(result, "unresolved") == 1


def test_join_fails_an_open_row_that_joins_an_nc_source(tmp_path):
    result = run_join_gate(build_release(tmp_path), nc_ids={"srcA"})
    assert failed(result, "not_flavour") == 4 and failed(result, "joins_non_open") == 4


# G2 frozen v1.0


def frozen_pair(tmp_path: Path) -> tuple[Path, Path]:
    v1, build = write_v1(tmp_path), build_release(tmp_path)
    shutil.copytree(v1 / "data", build / "data", dirs_exist_ok=True)
    return v1, build


def test_frozen_passes_identical_v1_files_and_added_configs(tmp_path):
    v1, build = frozen_pair(tmp_path)
    result = run_frozen_gate(build, v1_dir=v1)
    assert result.passed, result.failures
    assert result.counts.get("changed", 0) == 0 and result.counts.get("overlap", 0) >= 1


def test_frozen_fails_a_one_byte_change_under_a_v1_path(tmp_path):
    v1, build = frozen_pair(tmp_path)
    shard = build / "data" / "coralseg" / "train-00000-of-00001.parquet"
    data = bytearray(shard.read_bytes())
    data[20] ^= 1
    shard.write_bytes(bytes(data))
    result = run_frozen_gate(build, v1_dir=v1)
    assert not result.passed and result.counts.get("changed", 0) == 1


def test_frozen_fails_on_a_configs_edit_and_a_new_file_in_a_v1_config(tmp_path):
    v1, build = frozen_pair(tmp_path)
    (build / "README.md").write_text(card({"coralseg": "data/coralseg/other-*.parquet"}))
    write_shard(build, "coralseg", "train", [{"image_sha256": SHAS[1]}], shard=1)
    result = run_frozen_gate(build, v1_dir=v1)
    text = " ".join(result.failures)
    assert not result.passed and "extends_v1_config" in text and "coralseg" in text


# G3 leak


def test_leak_passes_a_clean_open_set(tmp_path):
    result = run_leak_gate(build_release(tmp_path), nc_ids={"ncsrc"})
    assert result.passed, result.failures


def test_leak_fails_on_the_nc_repo_name_in_any_open_file_without_echoing_it(tmp_path):
    build = build_release(tmp_path)
    (build / "README.md").write_text(V1_CARD + f"Mirror: {NC_REPO}\n")
    result = run_leak_gate(build, nc_ids={"ncsrc"})
    assert not result.passed and NC_REPO not in " ".join(result.failures)


def test_leak_fails_on_an_nc_id_in_a_parquet_column_and_on_a_private_bucket(tmp_path):
    build = build_release(tmp_path)
    write_shard(
        build, "metadata", "train", [{"image_sha256": "e" * 64, "source_id": "ncsrc"}], shard=1
    )
    (build / "croissant.json").write_text('{"url": "s3://rs-storage-private/x"}')
    result = run_leak_gate(build, nc_ids={"ncsrc"})
    assert result.counts["leaks"] >= 2 and "ncsrc" not in " ".join(result.failures)


# G4 bytes once


def test_bytes_once_passes_a_build_that_embeds_each_image_once(tmp_path):
    result = run_bytes_once_gate(build_release(tmp_path), v1_sha256={SHAS[0]})
    assert result.passed, result.failures


def test_bytes_once_fails_a_duplicate_a_v1_reupload_and_a_second_bytes_column(tmp_path):
    build = build_release(tmp_path)
    dup = {
        "image_sha256": SHAS[3],
        "bytes_sha256": "a" * 64,
        "image": {"bytes": b"x", "path": None},
    }
    write_shard(build, "images", "train", [dup], shard=1)
    write_shard(
        build, "boxes", "train", [{"image_sha256": SHAS[1], "image": {"bytes": b"y", "path": None}}]
    )
    result = run_bytes_once_gate(build, v1_sha256={SHAS[1]})
    assert failed(result, "duplicate") == 1 and failed(result, "v1_reupload") == 1
    assert failed(result, "bytes_outside_images") == 1


# G5 masks: mode P with the declared palette


def mask_png(values: list[int]) -> bytes:
    image = Image.new("P", (8, 8), 0)
    image.putpalette([c for i in range(256) for c in palette_rgb(i)])
    image.putdata([values[i % len(values)] for i in range(64)])
    out = io.BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def semseg_build(tmp_path: Path, values: list[int]) -> Path:
    png = mask_png(values)
    row = {"image_sha256": SHAS[1], "source_id": "srcA", "mask": {"bytes": png, "path": None},
           "mask_bytes_sha256": hashlib.sha256(png).hexdigest(), "mask_config": None,
           "palette": palette_json(json.dumps([{"id": 0}, {"id": 1}]))}  # fmt: skip
    write_shard(tmp_path / "build", "semseg", "train", [row])
    return tmp_path / "build"


def test_masks_gate_accepts_mode_p_with_the_declared_palette(tmp_path):
    result = run_masks_gate(semseg_build(tmp_path, [0, 1, 255]))
    assert result.passed, result.failures
    assert result.counts["palette_rows"] == 1


def test_masks_gate_fails_an_undeclared_colour(tmp_path):
    result = run_masks_gate(semseg_build(tmp_path, [0, 7]))
    assert not result.passed and failed(result, "undeclared_colour") == 1


# G6 splits


def test_splits_pass_a_consistent_build(tmp_path):
    build = build_release(tmp_path)
    result = run_splits_v11_gate(build, v1_dir=write_v1(tmp_path))
    assert result.passed, result.failures
    assert result.counts["images_validation"] == 1


def test_splits_fail_a_v1_reassignment_and_a_straddling_group(tmp_path):
    build = build_release(tmp_path, split_of={0: "test"}, group_of={1: "g0"})
    result = run_splits_v11_gate(build, v1_dir=write_v1(tmp_path))
    assert failed(result, "reassigned_v1") == 1 and failed(result, "group_straddles") == 1


def write_map(build: Path, overridden: dict | None) -> None:
    pins = {} if overridden is None else {"v1_continuity_overridden": overridden}
    (build / "SPLIT_MAP.json").write_text(json.dumps({"pins": pins, "assignments": {}}))


def test_splits_accept_exactly_the_groups_a_pin_moved_off_their_v1_split(tmp_path):
    v1 = write_v1(tmp_path)
    build = build_release(tmp_path, split_of={0: "test"})  # img0, group g0: v1.0 train -> test
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "reassigned_v1") == 1  # no SPLIT_MAP.json
    write_map(build, {"images": 1, "groups": 1, "by_source": {}, "group_ids": ["g0"]})
    listed = run_splits_v11_gate(build, v1_dir=v1)
    assert listed.passed, listed.failures
    assert listed.counts["v1_override_groups"] == 1
    write_map(build, {"images": 1, "groups": 1, "by_source": {}, "group_ids": ["g9"]})
    other = run_splits_v11_gate(build, v1_dir=v1)  # g0 changed unlisted; the listed g9 did not
    assert failed(other, "reassigned_v1") == 1 and failed(other, "override_unchanged") == 1


def test_splits_fail_a_listed_group_that_did_not_change_and_check_the_count_when_ids_are_cut(
    tmp_path,
):
    v1, build = write_v1(tmp_path), build_release(tmp_path)  # nothing left its v1.0 split
    write_map(build, {"images": 1, "groups": 1, "by_source": {}, "group_ids": ["g0"]})
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "override_unchanged") == 1
    write_map(build, {"images": 0, "groups": 0, "by_source": {}, "group_ids": []})
    assert run_splits_v11_gate(build, v1_dir=v1).passed
    moved = build_release(tmp_path / "moved", split_of={0: "test"})
    write_map(moved, {"images": 1, "groups": 1, "by_source": {}})  # over 200 groups: count only
    assert run_splits_v11_gate(moved, v1_dir=v1).passed
    write_map(moved, {"images": 1, "groups": 2, "by_source": {}})
    assert failed(run_splits_v11_gate(moved, v1_dir=v1), "override_unchanged") == 1


def test_splits_count_withheld_images_as_dropped_not_unsplit(tmp_path):
    build, v1 = build_release(tmp_path), write_v1(tmp_path)
    write_report(
        build, withheld_images=3, withheld_groups=2,
        withheld_by_source={"usis10k": {"images": 3, "groups": 2}},
    )  # fmt: skip
    result = run_splits_v11_gate(build, v1_dir=v1)
    assert result.passed, result.failures
    assert result.counts["unsplit"] == 0


def test_splits_fail_unsplit_and_allow_unsplit_builds_unless_rehearsal(tmp_path):
    build, v1 = build_release(tmp_path), write_v1(tmp_path)
    write_report(build, unsplit_images=2, allow_unsplit=True)
    assert not run_splits_v11_gate(build, v1_dir=v1).passed
    rehearsal = run_splits_v11_gate(build, v1_dir=v1, rehearsal=True)
    assert rehearsal.skipped and not rehearsal.passed  # a skip never publishes
    (build / "task_layers" / "V11_TABLES.json").unlink()
    assert failed(run_splits_v11_gate(build, v1_dir=v1, rehearsal=True), "no_report") == 1


def test_splits_ood_guards_and_the_non_upstream_ratio(tmp_path):
    build, v1 = build_release(tmp_path, split_of={3: "test_ood"}), write_v1(tmp_path)
    assert failed(run_splits_v11_gate(build, v1_dir=v1), "ood_size") == 2
    guards = {"min_images": 1, "min_groups": 1, "max_total_holdout_share": 0.5}
    assert run_splits_v11_gate(build, v1_dir=v1, ood_guards=guards).passed
    assert failed(
        run_splits_v11_gate(build, v1_dir=v1, ood_guards=guards, ratio_min_images=3), "ratio_off"
    )


# G7 decontamination


def test_decon_passes_without_overlap_and_fails_a_shared_sha256(tmp_path):
    build = build_release(tmp_path)
    clean, hidden, bench = tmp_path / "clean.json", tmp_path / "hidden.json", tmp_path / "bench.txt"
    clean.write_text(json.dumps({"sha256": [hashlib.sha256(b"other").hexdigest()]}))
    bench.write_text(hashlib.sha256(b"bench").hexdigest())
    assert run_decon_gate(build, hidden_test=[clean], benchmarks=[bench]).passed
    hidden.write_text(json.dumps({"images": [{"sha256": SHAS[2]}]}))
    result = run_decon_gate(build, hidden_test=[hidden], benchmarks=[bench])
    assert not result.passed and result.counts["overlaps"] == 1


def test_decon_requires_both_manifests_and_a_rehearsal_only_skips(tmp_path):
    build = build_release(tmp_path)
    assert not run_decon_gate(build).passed
    skipped = run_decon_gate(build, rehearsal=True)
    assert skipped.skipped and not skipped.passed


# publish: a v1.1.0 release must pass the v1.1.0 gates


def test_publish_requires_the_v11_gates_for_a_v11_release(tmp_path):
    ok = {"passed": True, "counts": {}, "failures": []}
    gates = {name: ok for name in (*REQUIRED, *FAIL_CLOSED)}
    (tmp_path / "GATES.json").write_text(json.dumps(gates))
    assert publish_verdict(tmp_path)[1] == []
    (tmp_path / "task_layers").mkdir()
    (tmp_path / "task_layers" / "V11_TABLES.json").write_text("{}")
    required, problems = publish_verdict(tmp_path)
    assert set(V11_REQUIRED) <= set(required) and len(problems) == len(V11_REQUIRED)
    (tmp_path / "GATES.json").write_text(json.dumps({**gates, **dict.fromkeys(V11_REQUIRED, ok)}))
    assert publish_verdict(tmp_path)[1] == []


# G8 packaging: CPU torch, privacy extra, pinned YuNet


def test_dockerfile_carries_the_privacy_extra_and_the_pinned_yunet_model():
    docker = (ROOT / "Dockerfile").read_text()
    extras = re.search(r'ARG EXTRAS="([^"]*)"', docker)[1].split()
    assert "privacy" in extras
    assert yunet_model.YUNET_URL in docker and f"sha256:{yunet_model.YUNET_SHA256}" in docker
    assert f"{yunet_model.ENV_VAR}=" in docker and "chmod 0444 /opt/models/" in docker


def test_lock_has_no_cuda_wheels():
    lock = (ROOT / "uv.lock").read_text()
    assert 'name = "nvidia-' not in lock and 'name = "triton"' not in lock
    assert "download.pytorch.org/whl/cpu" in lock


def test_yunet_loader_verifies_the_sha256_and_never_downloads(tmp_path, monkeypatch):
    model = tmp_path / "yunet.onnx"
    model.write_bytes(b"not the model")
    with pytest.raises(yunet_model.YuNetModelError, match="sha256"):
        yunet_model.verified_model_path(model)
    with pytest.raises(yunet_model.YuNetModelError, match="not set"):
        monkeypatch.delenv(yunet_model.ENV_VAR, raising=False)
        yunet_model.verified_model_path()
    monkeypatch.setattr(yunet_model, "YUNET_SHA256", hashlib.sha256(b"not the model").hexdigest())
    assert yunet_model.verified_model_path(model) == model
