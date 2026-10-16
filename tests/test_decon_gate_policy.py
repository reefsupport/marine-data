"""G7 follows registry/benchmarks.yaml: a pinned image passes in test/ood, fails elsewhere."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.benchmarks import BenchmarkRegistry
from marinedata.gates.decon_gate import is_pin_home, read_manifests, run_decon_gate

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = BenchmarkRegistry.load(ROOT / "registry" / "benchmarks.yaml")
PIN = "route-to-our-test"


def sha(name: str, salt: str = "") -> str:
    return hashlib.sha256(f"{salt}{name}".encode()).hexdigest()


def write_shard(root: Path, config: str, split: str, rows: list[dict]) -> None:
    stem = split.replace("-", "_")  # shard names carry the Hub split; ood-* is a column value
    path = root / "data" / config / f"{stem}-00000-of-00001.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def build(tmp_path: Path, splits: dict[str, str], *, split_column: bool = True) -> Path:
    """A build whose images and metadata shards hold one image per ``{name: split}``."""
    root = tmp_path / "build"
    for config in ("images", "metadata"):
        for split in sorted(set(splits.values())):
            rows = [
                {
                    "image_sha256": sha(name),
                    "bytes_sha256": sha(name, "bytes:"),
                    "source_id": f"src-{name[0]}",
                    **({"split": split} if split_column else {}),
                }
                for name, s in splits.items()
                if s == split
            ]
            write_shard(root, config, split, rows)
    return root


def manifest(path: Path, benchmark_id: str | None, shas: list[str]) -> Path:
    columns: dict = {"sha256": shas, "upstream_split": ["test"] * len(shas)}
    if benchmark_id is not None:
        columns = {"benchmark_id": [benchmark_id] * len(shas), **columns}
    pq.write_table(pa.table(columns), path)
    return path


def hidden(tmp_path: Path, *names: str) -> Path:
    path = tmp_path / "hidden.json"
    path.write_text(json.dumps({"images": [{"sha256": sha(n, "hidden:")} for n in names]}))
    return path


def run(build_dir, bench, hid, **kw):
    return run_decon_gate(build_dir, hidden_test=[hid], benchmarks=[bench], registry=REGISTRY, **kw)


def record(result, file: str) -> dict:
    counts = result if isinstance(result, dict) else result.counts
    return next(r for r in counts["manifests"] if r["file"] == file)


def test_route_to_our_test_overlap_in_test_or_ood_is_pinned_ok(tmp_path):
    names = {"a1": "test", "b2": "test_ood", "c3": "ood-geo-mediterranean"}
    bench = manifest(tmp_path / "coralscapes.parquet", "coralscapes", [sha(n) for n in names])
    result = run(build(tmp_path, names), bench, hidden(tmp_path, "other"))
    assert result.passed, result.failures
    assert (result.counts["overlaps"], result.counts["pinned_ok"]) == (3, 3)
    assert result.counts["failed"] == 0 and result.counts["failing"] == []
    assert record(result, "coralscapes.parquet")["policy"] == PIN


def test_route_to_our_test_overlap_in_train_or_validation_fails(tmp_path):
    names = {"a1": "train", "b2": "validation", "c3": "test"}
    shas = [sha("a1", "bytes:"), sha("b2"), sha("c3")]  # a bytes_sha256 digest counts too
    result = run(
        build(tmp_path, names),
        manifest(tmp_path / "coralscapes.parquet", "coralscapes", shas),
        hidden(tmp_path, "other"),
    )
    assert not result.passed
    assert (result.counts["pinned_ok"], result.counts["failed"]) == (1, 2)
    assert record(result, "coralscapes.parquet")["by_split"] == {
        "test": 1, "train": 1, "validation": 1,
    }  # fmt: skip
    assert {(f["split"], f["source"], f["benchmark_id"]) for f in result.counts["failing"]} == {
        ("train", "src-a", "coralscapes"),
        ("validation", "src-b", "coralscapes"),
    }
    assert "ship outside test/ood (train 1, validation 1)" in result.failures[0]


def test_exclude_benchmark_fails_even_in_test(tmp_path):
    for benchmark_id in ("marineeval", "u45"):
        bench = manifest(tmp_path / f"{benchmark_id}.parquet", benchmark_id, [sha("a1")])
        result = run(build(tmp_path / benchmark_id, {"a1": "test"}), bench, hidden(tmp_path, "x"))
        assert not result.passed and result.counts["failed"] == 1
        assert result.counts["pinned_ok"] == 0
        assert record(result, f"{benchmark_id}.parquet")["policy"] == "exclude"


def test_hidden_test_overlap_fails_in_test(tmp_path):
    build_dir = build(tmp_path, {"a1": "test"})
    bench = manifest(tmp_path / "coralscapes.parquet", "coralscapes", [sha("other")])
    hid = tmp_path / "hidden.json"
    hid.write_text(json.dumps({"images": [{"sha256": sha("a1")}]}))
    result = run(build_dir, bench, hid)
    assert not result.passed and result.counts["failed"] == 1
    assert result.counts["hidden_test_shas"] == 1
    assert record(result, "hidden.json")["policy"] == "hidden-test"
    assert result.counts["failing"][0]["split"] == "test"


def test_unknown_manifest_fails_closed_on_overlap_and_is_counted(tmp_path):
    build_dir = build(tmp_path, {"a1": "test"})
    overlap = manifest(tmp_path / "mystery.parquet", "not-a-benchmark", [sha("a1")])
    result = run(build_dir, overlap, hidden(tmp_path, "x"))
    assert not result.passed and result.counts["unknown_benchmark"] == 1
    assert result.counts["failed"] == 1 and "unknown_benchmark" in result.failures[0]
    clean = manifest(tmp_path / "mystery2.parquet", None, [sha("zzz")])
    result = run(build_dir, clean, hidden(tmp_path, "x"))
    assert result.passed and result.counts["unknown_benchmark"] == 1  # no overlap: nothing to fail


def test_a_folder_of_manifests_is_resolved_per_file(tmp_path):
    names = {"a1": "test", "b2": "test", "c3": "test", "d4": "train"}
    folder = tmp_path / "manifests"
    folder.mkdir()
    manifest(folder / "coralscapes.parquet", "coralscapes", [sha("a1")])  # id column
    manifest(folder / "marineeval.parquet", None, [sha("b2")])  # no column: the stem
    manifest(folder / "renamed.parquet", "suim", [sha("c3")])  # the column beats the stem
    (folder / "trashcan.txt").write_text(sha("d4"))  # text: the stem
    result = run(build(tmp_path, names), folder, hidden(tmp_path, "x"))
    got = {r["file"]: (r["benchmark_id"], r["policy"], r["pinned_ok"], r["failed"])
           for r in result.counts["manifests"] if r["kind"] == "benchmark"}  # fmt: skip
    assert got == {
        "coralscapes.parquet": ("coralscapes", PIN, 1, 0),
        "marineeval.parquet": ("marineeval", "exclude", 0, 1),
        "renamed.parquet": ("suim", PIN, 1, 0),
        "trashcan.txt": ("trashcan", PIN, 0, 1),
    }
    assert not result.passed and result.counts["benchmark_manifests"] == 4
    assert (result.counts["pinned_ok"], result.counts["failed"]) == (2, 2)
    assert len(result.failures) == 2


def test_one_parquet_with_two_benchmark_ids_is_judged_per_id(tmp_path):
    path = tmp_path / "all.parquet"
    pq.write_table(
        pa.table({"benchmark_id": ["coralscapes", "u45"], "sha256": [sha("a1"), sha("b2")]}), path
    )
    assert [(m.benchmark_id, len(m.shas)) for m in read_manifests(path, "benchmark")] == [
        ("coralscapes", 1),
        ("u45", 1),
    ]
    result = run(build(tmp_path, {"a1": "test", "b2": "test"}), path, hidden(tmp_path, "x"))
    assert (result.counts["pinned_ok"], result.counts["failed"]) == (1, 1)


def test_split_falls_back_to_the_shard_name_and_every_listing_counts(tmp_path):
    bench = manifest(tmp_path / "coralscapes.parquet", "coralscapes", [sha("a1"), sha("b2")])
    no_column = build(tmp_path / "x", {"a1": "train", "b2": "test"}, split_column=False)
    result = run(no_column, bench, hidden(tmp_path, "x"))
    assert [f["split"] for f in result.counts["failing"]] == ["train"]
    both = build(tmp_path / "y", {"b2": "test"})  # images: test; metadata: another split
    write_shard(both, "metadata", "train", [{"image_sha256": sha("b2"), "split": "train"}])
    result = run(both, bench, hidden(tmp_path, "x"))
    assert not result.passed and result.counts["failing"][0]["split"] == "test,train"


def test_failing_images_are_capped_at_twenty(tmp_path):
    names = {f"n{i:02d}": "train" for i in range(25)}
    bench = manifest(tmp_path / "coralscapes.parquet", "coralscapes", [sha(n) for n in names])
    result = run(build(tmp_path, names), bench, hidden(tmp_path, "x"))
    assert result.counts["failed"] == 25 and len(result.counts["failing"]) == 20
    dumped = json.loads(json.dumps(result.to_dict()))["counts"]
    assert record(dumped, "coralscapes.parquet")["failed"] == 25


@pytest.mark.parametrize(
    ("split", "home"),
    [("test", True), ("test_ood", True), ("ood-geo-mediterranean", True), ("OOD_x", True),
     ("train", False), ("validation", False), ("val", False), ("testing", False), ("", False)],
)  # fmt: skip
def test_pin_homes_are_test_and_every_ood_split(split, home):
    assert is_pin_home(split) is home


def test_every_shipped_manifest_resolves_to_a_registry_entry():
    manifests = [
        m
        for path in sorted((ROOT / "registry" / "benchmarks" / "manifests").glob("*.parquet"))
        for m in read_manifests(path, "benchmark")
    ]
    assert len(manifests) == 17
    policy = {m.benchmark_id: REGISTRY.by_id(m.benchmark_id).policy for m in manifests}
    assert {i for i, p in policy.items() if p == "exclude"} == {"marineeval", "u45"}
    for in_scope in ("coralscapes", "suim", "uiis", "uiis10k", "usis10k"):
        assert policy[in_scope] == PIN
