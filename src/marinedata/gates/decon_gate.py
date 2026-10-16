"""G7 decontamination: no release image is a hidden-test image, and a benchmark's eval images ship
only where ``registry/benchmarks.yaml`` allows.

Exact ``sha256`` overlap between the release (``image_sha256`` and ``bytes_sha256`` of the
``images`` and ``metadata`` configs, v1.0 images included) and every manifest given with
``--hidden-test`` and ``--benchmark`` (a file or a folder of parquet / json / jsonl / csv / tsv /
txt manifests: any 64-hex digest in them counts). Each manifest file is judged on its own:

* hidden-test manifest: any overlap fails, in any split;
* benchmark manifest: resolved to its registry entry (the ``benchmark_id`` column of a parquet
  manifest, else the file stem), then judged by that entry's ``policy`` (design §2.2):
  ``route-to-our-test`` passes an overlap as ``pinned_ok`` when the image is in ``test`` or an OOD
  split (``test_ood``, ``ood-*``) and fails it in train, validation or any other split (rules 1
  and 3); ``exclude`` fails any overlap (rule 2); a manifest with no registry entry reports
  ``unknown_benchmark`` and fails any overlap (fail closed).

An image's split is the ``split`` column the build writes into its ``images`` and ``metadata``
rows (the shard name when a shard has no such column); an image listed in several splits must be
in a pin home in every one of them. Both inputs are required: without them the gate fails; with
``--rehearsal`` it is ``skipped`` instead, which ``release publish`` still refuses.
Near-duplicate and embedding decontamination stay with ``marinedata decon check`` (S0-S5)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..benchmarks import BenchmarkRegistry, BenchmarksError
from . import GateResult, finish, skip
from .shards import norm_split, shard_split
from .v11_common import HEX64, IMAGES, METADATA, shards, tree_files

NAME = "decon"
SHA_COLUMNS = ("image_sha256", "sha256", "bytes_sha256", "sha")
KEY_COLUMNS = ("image_sha256", "bytes_sha256")
ID_COLUMN = "benchmark_id"
TEXT_SUFFIXES = frozenset({"", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".sha256"})
HIDDEN, BENCHMARK = "hidden_test", "benchmark"
PIN_POLICY, HIDDEN_POLICY, UNKNOWN_POLICY = "route-to-our-test", "hidden-test", "unknown"
PIN_HOMES = frozenset({"test", "test_ood", "test-ood"})
MAX_FAILING = 20
_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _clean(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if HEX64.match(text) else ""


def is_pin_home(split: str) -> bool:
    """``test`` and every OOD split (``test_ood``, ``ood-*``): where a pinned image may ship."""
    name = str(split or "").strip().lower()
    return name in PIN_HOMES or name.startswith(("ood-", "ood_"))


@dataclass(frozen=True)
class Manifest:
    """The digests of one hidden-test file or one benchmark (``kind``: hidden_test | benchmark)."""

    kind: str
    file: str
    benchmark_id: str
    shas: frozenset[str]


def _parquet_groups(file: Path) -> dict[str, set[str]]:
    """``{benchmark_id: digests}`` of a parquet manifest; a row with no id falls under the stem."""
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(file)
    columns = [c for c in (ID_COLUMN, *SHA_COLUMNS) if c in shard.schema_arrow.names]
    found: dict[str, set[str]] = defaultdict(set)
    for batch in shard.iter_batches(batch_size=8192, columns=columns):
        data = batch.to_pydict()
        ids = data.get(ID_COLUMN) or [None] * batch.num_rows
        for column in SHA_COLUMNS:
            for raw_id, value in zip(ids, data.get(column, ()), strict=False):
                if sha := _clean(value):
                    found[str(raw_id or file.stem).strip().lower()].add(sha)
    return dict(found)


def _file_groups(file: Path) -> dict[str, set[str]]:
    suffix = file.suffix.lower()
    if suffix == ".parquet":
        return _parquet_groups(file)
    if suffix in TEXT_SUFFIXES:
        found = {m.lower() for m in _HEX.findall(file.read_text(errors="ignore"))}
        return {file.stem.lower(): found} if found else {}
    return {}


def read_manifests(path: Path, kind: str) -> list[Manifest]:
    """The manifests of a file or a folder: one per (file, benchmark id) that lists a digest."""
    path = Path(path)
    files = list(tree_files(path)) if path.is_dir() else [path]
    return [
        Manifest(kind, file.name, benchmark_id, frozenset(shas))
        for file in files
        for benchmark_id, shas in sorted(_file_groups(file).items())
    ]


def manifest_shas(path: Path) -> set[str]:
    """Every sha256 a manifest file (or a folder of them) lists."""
    return set().union(*(m.shas for m in read_manifests(path, BENCHMARK)))


@dataclass
class ReleaseHit:
    """Where a digest that a manifest lists ships: every split and source it appears under."""

    splits: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)


def scan_release(build_dir: Path, wanted: frozenset[str]) -> tuple[int, dict[str, ReleaseHit]]:
    """``(distinct release digests, {digest in wanted: ReleaseHit})`` over ``image_sha256`` and
    ``bytes_sha256`` of the ``images`` and ``metadata`` shards. The split is the row's ``split``
    column, else the shard's name; the ``image`` bytes column is never read."""
    import pyarrow.parquet as pq

    seen: set[str] = set()
    hits: dict[str, ReleaseHit] = {}
    for config in (IMAGES, METADATA):
        for path in shards(Path(build_dir), config):
            shard = pq.ParquetFile(path)
            names = shard.schema_arrow.names
            if not set(KEY_COLUMNS) & set(names):
                continue
            columns = [c for c in (*KEY_COLUMNS, "split", "source_id") if c in names]
            fallback = shard_split(path)
            for batch in shard.iter_batches(batch_size=8192, columns=columns):
                data, size = batch.to_pydict(), batch.num_rows
                splits = data.get("split") or [None] * size
                sources = data.get("source_id") or [None] * size
                for column in KEY_COLUMNS:
                    for value, split, source in zip(
                        data.get(column, ()), splits, sources, strict=False
                    ):
                        if not (sha := _clean(value)):
                            continue
                        seen.add(sha)
                        if sha in wanted:
                            hit = hits.setdefault(sha, ReleaseHit())
                            hit.splits.add(norm_split(str(split or fallback).strip().lower()))
                            if source:
                                hit.sources.add(str(source))
    return len(seen), hits


def _policy(manifest: Manifest, registry: BenchmarkRegistry) -> str:
    if manifest.kind == HIDDEN:
        return HIDDEN_POLICY
    try:
        return registry.by_id(manifest.benchmark_id).policy
    except KeyError:
        return UNKNOWN_POLICY


def _judge(manifest: Manifest, policy: str, hits: dict[str, ReleaseHit], failing: list) -> dict:
    """The manifest's GATES.json record; adds failing images to ``failing`` (up to 20)."""
    overlap = sorted(manifest.shas & hits.keys())
    by_split: Counter = Counter()
    pinned_ok = 0
    for sha in overlap:
        hit = hits[sha]
        by_split.update(hit.splits)
        if policy == PIN_POLICY and all(is_pin_home(s) for s in hit.splits):
            pinned_ok += 1
        elif len(failing) < MAX_FAILING:
            failing.append(
                {
                    "sha256": sha,
                    "source": ",".join(sorted(hit.sources)),
                    "split": ",".join(sorted(hit.splits)),
                    "benchmark_id": manifest.benchmark_id,
                }
            )
    return {
        "kind": manifest.kind,
        "file": manifest.file,
        "benchmark_id": manifest.benchmark_id,
        "policy": policy,
        "manifest_shas": len(manifest.shas),
        "overlap": len(overlap),
        "pinned_ok": pinned_ok,
        "failed": len(overlap) - pinned_ok,
        "by_split": dict(sorted(by_split.items())),
    }


def _failure(record: dict) -> str:
    where = f"{record['kind']}/{record['file']}"
    ident, policy, n = record["benchmark_id"], record["policy"], record["failed"]
    if policy == PIN_POLICY:
        outside = ", ".join(f"{s} {c}" for s, c in record["by_split"].items() if not is_pin_home(s))
        return (
            f"{where} ({ident}, {policy}): {n} of {record['overlap']} overlapping image(s) "
            f"ship outside test/ood ({outside})"
        )
    if policy == UNKNOWN_POLICY:
        return (
            f"{where} unknown_benchmark '{ident}': {n} overlapping image(s); "
            "no registry entry allows a pin (fail closed)"
        )
    return f"{where} ({ident}, {policy}): {n} overlapping image(s); no split may ship them"


def run_decon_gate(
    build_dir: Path,
    *,
    hidden_test: Sequence[Path] = (),
    benchmarks: Sequence[Path] = (),
    rehearsal: bool = False,
    registry: BenchmarkRegistry | None = None,
) -> GateResult:
    required = (("hidden-test manifest", hidden_test), ("benchmark manifests", benchmarks))
    missing = [label for label, paths in required if not paths]
    if missing:
        reason = "required input missing: " + ", ".join(missing)
        return skip(NAME, f"rehearsal: {reason}") if rehearsal else finish(NAME, {}, [reason])
    try:
        registry = registry or BenchmarkRegistry.load()
    except BenchmarksError as exc:
        return finish(NAME, {}, [f"benchmark registry unreadable: {str(exc)[:200]}"])
    failures: list[str] = []
    manifests: list[Manifest] = []
    for kind, paths in ((HIDDEN, hidden_test), (BENCHMARK, benchmarks)):
        for path in map(Path, paths):
            if not path.exists():
                failures.append(f"{kind} manifest {path.name}: not found")
                continue
            found = read_manifests(path, kind)
            if not found:
                failures.append(f"{kind} manifest {path.name}: empty (no sha256 found)")
            manifests.extend(found)
    total, hits = scan_release(Path(build_dir), frozenset().union(*(m.shas for m in manifests)))
    if not total:
        return finish(NAME, {}, ["the build holds no image_sha256 to check", *failures])
    counts: Counter = Counter(
        {"release_shas": total, "overlaps": 0, "pinned_ok": 0, "failed": 0, "unknown_benchmark": 0}
    )
    failing: list[dict] = []
    records: list[dict] = []
    for manifest in manifests:
        record = _judge(manifest, _policy(manifest, registry), hits, failing)
        records.append(record)
        counts[f"{manifest.kind}_manifests"] += 1
        counts["unknown_benchmark"] += int(record["policy"] == UNKNOWN_POLICY)
        counts["overlaps"] += record["overlap"]
        counts["pinned_ok"] += record["pinned_ok"]
        counts["failed"] += record["failed"]
        if record["failed"]:
            failures.append(_failure(record))
    for kind in (HIDDEN, BENCHMARK):
        counts[f"{kind}_shas"] = len(
            frozenset().union(*(m.shas for m in manifests if m.kind == kind))
        )
    return finish(NAME, {**counts, "manifests": records, "failing": failing}, failures)
