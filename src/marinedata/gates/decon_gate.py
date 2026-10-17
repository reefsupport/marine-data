"""G7 decontamination: no release image is a hidden-test image, and a benchmark's eval images ship
only where ``registry/benchmarks.yaml`` allows.

Exact ``sha256`` overlap between the release (``image_sha256`` and ``bytes_sha256`` of the
``images`` and ``metadata`` configs, v1.0 images included) and every manifest given with
``--hidden-test`` and ``--benchmark`` (a file or a folder of parquet / json / jsonl / csv / tsv /
txt manifests: any 64-hex digest in them counts). Each manifest file is judged on its own:

* hidden-test manifest: any overlap fails, in any split. With
  ``--hidden-test-published-exception`` (off by default; it needs the v1.0 tree, ``--v1-dir``)
  an overlap whose digest is an ``image_sha256`` of the frozen v1.0 tree is exempt, because that
  image is public already (``counts.hidden_test_already_published`` and a sample of 20); the
  others keep failing as ``hidden_test_new`` (``counts.hidden_test_new``). A benchmark overlap
  is never exempt;
* benchmark manifest: resolved to its registry entry (the ``benchmark_id`` column of a parquet
  manifest, else the file stem), then judged by that entry's ``policy`` (design §2.2):
  ``route-to-our-test`` passes an overlap as ``pinned_ok`` when the image is in ``test`` or an OOD
  split (``test_ood``, ``ood-*``) and fails it in train, validation or any other split (rules 1
  and 3), except an image of the entry's second held-out split (``heldout_val``, else the
  upstream ``val`` when the eval split is another one), which also passes in ``validation``;
  ``exclude`` fails any overlap (rule 2); a manifest with no registry entry reports
  ``unknown_benchmark`` and fails any overlap (fail closed).

An image's split is the ``split`` column the build writes into its ``images`` and ``metadata``
rows (the shard name when a shard has no such column); an image listed in several splits must be
in a pin home in every one of them. Both inputs are required: without them the gate fails; with
``--rehearsal`` it is ``skipped`` instead, which ``release publish`` still refuses.
Near-duplicate and embedding decontamination stay with ``marinedata decon check`` (S0-S5)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..benchmark_pins import EVAL, HELDOUT, split_role
from ..benchmarks import BenchmarkRegistry, BenchmarksError
from . import GateResult, finish, skip
from .shards import norm_split, shard_split
from .v11_common import HEX64, IMAGES, METADATA, shards, tree_files

NAME = "decon"
SHA_COLUMNS = ("image_sha256", "sha256", "bytes_sha256", "sha")
KEY_COLUMNS = ("image_sha256", "bytes_sha256")
ID_COLUMN = "benchmark_id"
SPLIT_COLUMN = "upstream_split"
TEXT_SUFFIXES = frozenset({"", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".sha256"})
HIDDEN, BENCHMARK = "hidden_test", "benchmark"
PIN_POLICY, HIDDEN_POLICY, UNKNOWN_POLICY = "route-to-our-test", "hidden-test", "unknown"
PIN_HOMES = frozenset({"test", "test_ood", "test-ood"})
MAX_FAILING = 20
HIDDEN_NEW, HIDDEN_PUBLISHED = "hidden_test_new", "hidden_test_already_published"
NO_V1_SET = (
    "--hidden-test-published-exception: no v1.0 sha256 set; pass --v1-dir DIR, the frozen v1.0 "
    "release dir with image_sha256 under data/<config>/*.parquet"
)
ROLE_KEYS = ("eval_pinned_ok", "eval_failed", "heldout_pinned_ok", "heldout_failed")
_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _clean(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if HEX64.match(text) else ""


def is_pin_home(split: str) -> bool:
    """``test`` and every OOD split (``test_ood``, ``ood-*``): where a pinned image may ship."""
    name = str(split or "").strip().lower()
    return name in PIN_HOMES or name.startswith(("ood-", "ood_"))


def is_accepted_home(split: str, role: str = EVAL) -> bool:
    """Where a pinned image may ship: an eval image in ``test`` or an OOD split, a held-out image
    (the benchmark's second held-out split) also in ``validation``."""
    if is_pin_home(split):
        return True
    return role == HELDOUT and norm_split(str(split or "").strip().lower()) in {"val", "validation"}


@dataclass(frozen=True)
class Manifest:
    """The digests of one hidden-test file or one benchmark (``kind``: hidden_test | benchmark)."""

    kind: str
    file: str
    benchmark_id: str
    shas: frozenset[str]
    splits: Mapping[str, frozenset[str]] = field(default_factory=dict, compare=False, hash=False)
    """``sha256 -> its upstream_split labels`` for a parquet manifest (empty for other files)."""


def _parquet_groups(file: Path) -> dict[str, dict[str, set[str]]]:
    """``{benchmark_id: {digest: upstream_split labels}}`` of a parquet manifest.

    A row with no id falls under the file stem; a file with no ``upstream_split`` column labels
    every digest ``""``."""
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(file)
    names = shard.schema_arrow.names
    columns = [c for c in (ID_COLUMN, SPLIT_COLUMN, *SHA_COLUMNS) if c in names]
    found: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for batch in shard.iter_batches(batch_size=8192, columns=columns):
        data = batch.to_pydict()
        ids = data.get(ID_COLUMN) or [None] * batch.num_rows
        labels = data.get(SPLIT_COLUMN) or [None] * batch.num_rows
        for column in SHA_COLUMNS:
            for raw_id, label, value in zip(ids, labels, data.get(column, ()), strict=False):
                if sha := _clean(value):
                    digests = found[str(raw_id or file.stem).strip().lower()]
                    digests.setdefault(sha, set()).add(str(label or "").strip())
    return dict(found)


def _file_groups(file: Path) -> dict[str, dict[str, set[str]]]:
    suffix = file.suffix.lower()
    if suffix == ".parquet":
        return _parquet_groups(file)
    if suffix in TEXT_SUFFIXES:
        found = {m.lower() for m in _HEX.findall(file.read_text(errors="ignore"))}
        return {file.stem.lower(): {sha: set() for sha in sorted(found)}} if found else {}
    return {}


def read_manifests(path: Path, kind: str) -> list[Manifest]:
    """The manifests of a file or a folder: one per (file, benchmark id) that lists a digest."""
    path = Path(path)
    files = list(tree_files(path)) if path.is_dir() else [path]
    return [
        Manifest(
            kind,
            file.name,
            benchmark_id,
            frozenset(shas),
            {sha: frozenset(labels) for sha, labels in shas.items()},
        )
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


def _v1_published(v1_sha256: Collection[str] | None, v1_dir: Path | None) -> frozenset[str]:
    """The ``image_sha256`` set of the frozen v1.0 tree: what G4 calls a ``v1_reupload``. The
    given set wins, else ``<v1_dir>/data/*/*.parquet`` is read; empty when neither is there."""
    if v1_sha256 is None and v1_dir is not None:
        from ..hf_export_v11 import v1_sha256s

        v1_sha256 = v1_sha256s(v1_dir)
    return frozenset(filter(None, map(_clean, v1_sha256 or ())))


def _policy(manifest: Manifest, registry: BenchmarkRegistry) -> str:
    if manifest.kind == HIDDEN:
        return HIDDEN_POLICY
    try:
        return registry.by_id(manifest.benchmark_id).policy
    except KeyError:
        return UNKNOWN_POLICY


def _heldout(manifest: Manifest, registry: BenchmarkRegistry) -> frozenset[str]:
    """The benchmark manifest's digests that all belong to the entry's second held-out split."""
    if manifest.kind != BENCHMARK:
        return frozenset()
    try:
        entry = registry.by_id(manifest.benchmark_id)
    except KeyError:
        return frozenset()
    if entry.policy != PIN_POLICY:
        return frozenset()
    return frozenset(
        sha
        for sha, labels in manifest.splits.items()
        if labels and all(split_role(entry, label) == HELDOUT for label in labels)
    )


def _example(sha: str, hit: ReleaseHit, manifest: Manifest) -> dict:
    return {
        "sha256": sha,
        "source": ",".join(sorted(hit.sources)),
        "split": ",".join(sorted(hit.splits)),
        "benchmark_id": manifest.benchmark_id,
    }


def _judge(
    manifest: Manifest,
    policy: str,
    hits: dict[str, ReleaseHit],
    failing: list,
    heldout: frozenset[str] = frozenset(),
    published: frozenset[str] | None = None,
    exempt: list | None = None,
) -> dict:
    """The manifest's GATES.json record; adds failing images to ``failing`` (up to 20).

    Counts are split by role (``eval`` / ``heldout``) and verdict (``pinned_ok`` / ``failed``).
    ``published`` (the v1.0 tree's ``image_sha256`` set) turns the published exception on for a
    hidden-test manifest: an overlap in it is not judged but counted as ``already_published`` and
    listed in ``exempt`` (up to 20); every other overlap fails as ``hidden_test_new``."""
    overlap = sorted(manifest.shas & hits.keys())
    exempting = published if manifest.kind == HIDDEN else None
    by_split: Counter = Counter()
    failed_by_split: Counter = Counter()
    verdicts: Counter = Counter()
    already_published = 0
    for sha in overlap:
        hit = hits[sha]
        by_split.update(hit.splits)
        if exempting is not None and sha in exempting:
            already_published += 1
            if exempt is not None and len(exempt) < MAX_FAILING:
                exempt.append(_example(sha, hit, manifest))
            continue
        role = HELDOUT if policy == PIN_POLICY and sha in heldout else EVAL
        outside = {s for s in hit.splits if not is_accepted_home(s, role)}
        if policy == PIN_POLICY and not outside:
            verdicts[role, "pinned_ok"] += 1
            continue
        verdicts[role, "failed"] += 1
        failed_by_split.update(outside or hit.splits)
        if len(failing) < MAX_FAILING:
            kind = {"kind": HIDDEN_NEW} if exempting is not None else {}
            failing.append({**_example(sha, hit, manifest), **kind})
    pinned_ok = verdicts[EVAL, "pinned_ok"] + verdicts[HELDOUT, "pinned_ok"]
    return {
        "kind": manifest.kind,
        "file": manifest.file,
        "benchmark_id": manifest.benchmark_id,
        "policy": policy,
        "manifest_shas": len(manifest.shas),
        "overlap": len(overlap),
        "pinned_ok": pinned_ok,
        "failed": len(overlap) - pinned_ok - already_published,
        "eval_pinned_ok": verdicts[EVAL, "pinned_ok"],
        "eval_failed": verdicts[EVAL, "failed"],
        "heldout_pinned_ok": verdicts[HELDOUT, "pinned_ok"],
        "heldout_failed": verdicts[HELDOUT, "failed"],
        "by_split": dict(sorted(by_split.items())),
        "failed_by_split": dict(sorted(failed_by_split.items())),
        **({"already_published": already_published} if exempting is not None else {}),
    }


def _failure(record: dict) -> str:
    where = f"{record['kind']}/{record['file']}"
    ident, policy, n = record["benchmark_id"], record["policy"], record["failed"]
    if policy == PIN_POLICY:
        failed = record.get("failed_by_split") or record["by_split"]
        outside = ", ".join(f"{s} {c}" for s, c in failed.items() if not is_pin_home(s))
        return (
            f"{where} ({ident}, {policy}): {n} of {record['overlap']} overlapping image(s) "
            f"ship outside test/ood ({outside}); eval_failed={record.get('eval_failed', n)}, "
            f"heldout_failed={record.get('heldout_failed', 0)} "
            "(a held-out image may also ship in validation)"
        )
    if "already_published" in record:
        return (
            f"{where} ({ident}, {policy}): {HIDDEN_NEW}: {n} of {record['overlap']} overlapping "
            f"image(s) are not in the v1.0 tree ({record['already_published']} already published "
            "are exempt); no split may ship them"
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
    hidden_test_published_exception: bool = False,
    v1_sha256: Collection[str] | None = None,
    v1_dir: Path | None = None,
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
    published: frozenset[str] | None = None
    if hidden_test_published_exception:
        published = _v1_published(v1_sha256, v1_dir)
        if not published:
            return finish(NAME, {}, [NO_V1_SET])
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
        {
            "release_shas": total,
            "overlaps": 0,
            "pinned_ok": 0,
            "failed": 0,
            "unknown_benchmark": 0,
            **dict.fromkeys(ROLE_KEYS, 0),
            **({HIDDEN_PUBLISHED: 0, HIDDEN_NEW: 0} if published is not None else {}),
        }
    )
    failing: list[dict] = []
    exempt: list[dict] = []
    records: list[dict] = []
    for manifest in manifests:
        record = _judge(
            manifest,
            _policy(manifest, registry),
            hits,
            failing,
            _heldout(manifest, registry),
            published,
            exempt,
        )
        records.append(record)
        counts[f"{manifest.kind}_manifests"] += 1
        counts["unknown_benchmark"] += int(record["policy"] == UNKNOWN_POLICY)
        counts["overlaps"] += record["overlap"]
        counts["pinned_ok"] += record["pinned_ok"]
        counts["failed"] += record["failed"]
        for key in ROLE_KEYS:
            counts[key] += record[key]
        if "already_published" in record:
            counts[HIDDEN_PUBLISHED] += record["already_published"]
            counts[HIDDEN_NEW] += record["failed"]
        if record["failed"]:
            failures.append(_failure(record))
    for kind in (HIDDEN, BENCHMARK):
        counts[f"{kind}_shas"] = len(
            frozenset().union(*(m.shas for m in manifests if m.kind == kind))
        )
    sample = {f"{HIDDEN_PUBLISHED}_sample": exempt} if published is not None else {}
    return finish(NAME, {**counts, "manifests": records, "failing": failing, **sample}, failures)
