"""registry/benchmarks.yaml: schema validation, hash determinism, manifest coverage.

Design: docs/design/eval-decontamination-split-v2.md §1. `benchmarks_sha256` feeds the
split-map hash (§3.5), so its determinism and its sensitivity to real content changes
are both load-bearing, not just "loads without throwing".
"""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from marinedata.benchmarks import (
    BenchmarkRegistry,
    BenchmarksError,
    benchmarks_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "registry" / "benchmarks.yaml"
MANIFEST_DIR = REPO_ROOT / "registry" / "benchmarks" / "manifests"


@pytest.fixture()
def registry() -> BenchmarkRegistry:
    return BenchmarkRegistry.load(REGISTRY_PATH)


# ── schema ────────────────────────────────────────────────────────────────


def test_loads_and_validates_every_entry(registry: BenchmarkRegistry) -> None:
    assert len(registry.benchmarks) >= 1
    ids = [e.id for e in registry.benchmarks]
    assert len(ids) == len(set(ids)), "duplicate benchmark ids"
    for entry in registry.benchmarks:
        assert entry.upstream_split.definition_url.startswith("http")
        assert entry.obtain.status in {
            "staged",
            "registry-only",
            "w1",
            "w2",
            "w3",
            "needs-yohan",
            "unfetchable",
        }
        if entry.policy == "exclude":
            assert entry.policy_reason.strip()


def test_split_verified_true_requires_evidence() -> None:
    fixture = _fixture_doc()
    fixture["benchmarks"][0]["split_verified"] = True
    fixture["benchmarks"][0]["verified_by"] = "not fetched"
    with pytest.raises(BenchmarksError):
        _load_fixture(fixture)


def test_rejects_unknown_field(tmp_path: Path) -> None:
    fixture = _fixture_doc()
    fixture["benchmarks"][0]["bogus_field"] = "x"
    with pytest.raises(BenchmarksError):
        _load_fixture(fixture, tmp_path)


def test_rejects_duplicate_id(tmp_path: Path) -> None:
    fixture = _fixture_doc()
    dup = dict(fixture["benchmarks"][0])
    fixture["benchmarks"].append(dup)
    with pytest.raises(BenchmarksError):
        _load_fixture(fixture, tmp_path)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BenchmarksError):
        BenchmarkRegistry.load(tmp_path / "nope.yaml")


# ── benchmarks_sha256 determinism ───────────────────────────────────────────


def test_hash_is_deterministic(registry: BenchmarkRegistry) -> None:
    a = benchmarks_sha256(registry)
    b = benchmarks_sha256(BenchmarkRegistry.load(REGISTRY_PATH))
    assert a == b
    assert len(a) == 64


def test_hash_changes_with_a_threshold(tmp_path: Path) -> None:
    fixture = _fixture_doc()
    base_hash = benchmarks_sha256(_load_fixture(fixture, tmp_path, name="a.yaml"))
    fixture["thresholds"]["s4_tau_dedup"] = 0.5
    changed_hash = benchmarks_sha256(_load_fixture(fixture, tmp_path, name="b.yaml"))
    assert base_hash != changed_hash


def test_hash_changes_with_a_benchmark_entry(tmp_path: Path) -> None:
    fixture = _fixture_doc()
    base_hash = benchmarks_sha256(_load_fixture(fixture, tmp_path, name="a.yaml"))
    fixture["benchmarks"][0]["policy_reason"] = "changed"
    changed_hash = benchmarks_sha256(_load_fixture(fixture, tmp_path, name="b.yaml"))
    assert base_hash != changed_hash


def test_hash_ignores_comment_only_edits(tmp_path: Path) -> None:
    """A file re-saved with different key order/whitespace hashes the same content."""
    fixture = _fixture_doc()
    path_a = tmp_path / "a.yaml"
    path_a.write_text(yaml.safe_dump(fixture, sort_keys=True))
    path_b = tmp_path / "b.yaml"
    path_b.write_text(yaml.safe_dump(fixture, sort_keys=False))
    a = benchmarks_sha256(BenchmarkRegistry.load(path_a))
    b = benchmarks_sha256(BenchmarkRegistry.load(path_b))
    assert a == b


# ── manifest coverage (real artefacts built by this package) ───────────────


REQUIRED_MANIFEST_COLUMNS = {
    "benchmark_id",
    "upstream_path",
    "stem",
    "upstream_split",
    "sha256",
    "pixel_sha256",
    "width",
    "height",
    "dhash",
    "phash",
    "phash64",
    "margin",
    "embedding_ref",
    "embedding_model",
}


def _built_manifests() -> list[Path]:
    if not MANIFEST_DIR.is_dir():
        return []
    return sorted(MANIFEST_DIR.glob("*.parquet"))


@pytest.mark.skipif(not _built_manifests(), reason="no manifests built yet")
def test_manifest_schema() -> None:
    for path in _built_manifests():
        table = pq.read_table(path)
        cols = set(table.schema.names)
        missing = REQUIRED_MANIFEST_COLUMNS - cols
        assert not missing, f"{path.name}: missing columns {missing}"
        assert table.num_rows > 0, f"{path.name}: empty manifest"
        ids = table.column("benchmark_id").to_pylist()
        assert set(ids) == {path.stem}, f"{path.name}: benchmark_id must equal the file stem"


@pytest.mark.skipif(not _built_manifests(), reason="no manifests built yet")
def test_manifest_coverage_at_least_five_or_documented(registry: BenchmarkRegistry) -> None:
    """§5 P1 acceptance: manifest row count >= 99% of eval n for >= 5 benchmarks.

    Only 2 benchmarks (coralscapes, suim) have images staged now under the brief's
    staged-only constraint (see the P1 report's Open section) — this test pins the
    coverage ratio for whichever manifests exist rather than asserting a count of 5,
    so it stays meaningful as more benchmarks land.
    """
    for path in _built_manifests():
        entry = registry.by_id(path.stem)
        table = pq.read_table(path)
        eval_n = sum(
            n
            for split, n in entry.upstream_split.counts.items()
            if split in {entry.upstream_split.eval_split, entry.upstream_split.heldout_val}
            and n is not None
        )
        if eval_n == 0:
            continue  # counts not published for this entry; row-count alone can't be checked
        assert table.num_rows / eval_n >= 0.99, (
            f"{path.name}: {table.num_rows} rows / {eval_n} eval images < 99%"
        )


def test_pending_lists_benchmarks_without_a_manifest(registry: BenchmarkRegistry) -> None:
    pending = set(registry.pending())
    built = {p.stem for p in _built_manifests()}
    assert pending.isdisjoint(built)
    assert pending | built == {e.id for e in registry.benchmarks}


# ── fixtures ─────────────────────────────────────────────────────────────


def _fixture_doc() -> dict:
    return {
        "schema_version": 1,
        "thresholds": {
            "s3_dhash64_candidate_max": 12,
            "s3_phash256_confirm_max": 32,
            "s3_phash256_review_max": 48,
            "s3_min_entropy_bits": 4.0,
            "s4_embedding_model": "facebook/dinov2-small",
            "s4_embedding_revision": None,
            "s4_tau_dedup": 0.93,
            "s4_tau_dedup_calibration_sha256": None,
            "s4_tau_decon_offset": -0.03,
            "s4_review_width": 0.05,
            "s4_ann_top_k": 10,
            "s5_parent_ncc_min": 0.90,
            "s5_parent_ncc_review_min": 0.80,
            "gate_review_band_max": {"absolute": 5, "fraction_of_eval": 0.01},
            "gate_manifest_min_coverage": 0.99,
        },
        "benchmarks": [
            {
                "id": "fixture-bench",
                "name": "Fixture Bench",
                "task": "cls",
                "catalog_id": "fixture-bench",
                "registry_id": None,
                "upstream_split": {
                    "rule": "train/test",
                    "eval_split": "test",
                    "heldout_val": None,
                    "counts": {"train": 10, "test": 5},
                    "definition_url": "https://example.org/fixture",
                },
                "split_verified": True,
                "verified_by": "fixture data, not a real source",
                "obtain": {"status": "unfetchable", "via": "n/a"},
                "policy": "route-to-our-test",
                "policy_reason": "fixture",
                "chain": [],
            }
        ],
    }


def _load_fixture(doc: dict, tmp_path: Path | None = None, name: str = "fixture.yaml"):
    import tempfile

    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp())
    path = tmp_path / name
    path.write_text(json.dumps(doc))
    return BenchmarkRegistry.load(path)
