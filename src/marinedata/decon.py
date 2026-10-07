"""``marinedata decon check`` — benchmark decontamination gate (WP-12/P2).

Design: ``docs/design/eval-decontamination-split-v2.md`` §2. Stages S0-S5 are cheapest
first; the first stage that fires on a (benchmark, our-image) pair wins. Candidate
generation and per-image signals reuse :mod:`marinedata.dedup` verbatim (features,
:class:`~marinedata.dedup.mih.MultiIndexHamming`, SSCD embeddings, the patch matcher) —
this module only adds the decon-specific classification and the gate/report on top.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .benchmarks import (
    BenchmarkEntry,
    BenchmarkRegistry,
    Thresholds,
    benchmarks_sha256,
    expected_eval_count,
)
from .dedup.corpus import Item, hash_items, read_payloads
from .dedup.crop import thumb_pair
from .dedup.embed import DEFAULT_SCALES, EMBED_SIDE, match_patch, prepare
from .dedup.features import LOWTEX_MARGIN, phash_distance
from .dedup.mih import MultiIndexHamming, unique_pairs

STAGE_ORDER = ("S0", "S1", "S2", "S3", "S4", "S5")
_STAGE_RANK = {s: i for i, s in enumerate(STAGE_ORDER)}
# Splits that are never a legitimate home for training-visible data: an ID pool, not
# an eval split. §2.2 rule 1: any hit here (for any policy) fails. `test`/`ood-*` are
# the only splits a `route-to-our-test` pin is allowed to land in (§2.2 rule 3).
_EVAL_HOMES = {"test"}


class DeconError(Exception):
    """Bad input to the gate — never a contamination finding."""


def _is_eval_home(split: str) -> bool:
    return split in _EVAL_HOMES or split.startswith("ood-")


def _file_loader(path: Path) -> Callable[[], bytes]:
    def _load() -> bytes:
        return path.read_bytes()

    return _load


def _payload_loader(ref: Path | tuple[str, int, int]) -> Callable[[], bytes]:
    def _load() -> bytes:
        return read_payloads([ref])[0]

    return _load


@dataclass(frozen=True)
class ImageRecord:
    """One image's decon signals, on either the benchmark or our side.

    ``loader`` returns the encoded bytes on demand (S3 entropy guard, S4 embedding,
    S5 patch match) — most pairs never reach a stage that needs it.
    """

    sha256: str
    pixel_sha256: str
    dhash: int
    phash: bytes
    phash64: int
    margin: float
    area: int
    upstream_id: str | None = None
    splits: frozenset[str] = frozenset()
    embedding: np.ndarray | None = None
    loader: Callable[[], bytes] | None = None

    @property
    def lowtex(self) -> bool:
        return self.margin < LOWTEX_MARGIN


@dataclass(frozen=True)
class ReviewHit:
    benchmark_id: str
    stage: str
    our_sha256: str
    bench_sha256: str
    split: str
    signal: float


@dataclass(frozen=True)
class BenchmarkOverlap:
    benchmark_id: str
    task: str
    eval_n: int
    hashed_n: int
    policy: str
    status: str  # clean | contaminated | not-checked | exempt | uncovered
    counts: dict[str, dict[str, int]]  # split -> stage -> count, FAILING placements only
    home_counts: dict[str, dict[str, int]]  # split -> stage -> count, test/ood-* (informational)
    excluded: int
    review: tuple[ReviewHit, ...]
    coverage_fail: bool
    review_band_fail: bool
    reason: str = ""
    # WP-R11: a reviewed ``decon_exempt_reason`` on a benchmark that HAS a manifest the registry
    # cannot size (no independent eval count, or a short manifest): hit detection ran, the
    # coverage gate is waived, and the benchmark is reported as partially verified.
    partial_reason: str = ""

    @property
    def partial(self) -> bool:
        return bool(self.partial_reason)

    @property
    def ok(self) -> bool:
        if self.status in {"not-checked", "exempt"}:
            return True
        any_hit = any(n for stages in self.counts.values() for n in stages.values())
        return not any_hit and not self.coverage_fail and not self.review_band_fail


@dataclass(frozen=True)
class DeconResult:
    overlaps: tuple[BenchmarkOverlap, ...]
    thresholds_sha256: str
    registry_sha256: str

    @property
    def ok(self) -> bool:
        return all(o.ok for o in self.overlaps)

    @property
    def exempt(self) -> dict[str, str]:
        """``benchmark id -> reviewed reason`` for every benchmark decon did not verify."""
        return {o.benchmark_id: o.reason for o in self.overlaps if o.status == "exempt"}

    @property
    def partial(self) -> dict[str, str]:
        """``benchmark id -> reviewed reason`` for every benchmark checked on a manifest whose
        coverage the registry could not verify."""
        return {o.benchmark_id: o.partial_reason for o in self.overlaps if o.partial}

    @property
    def failures(self) -> list[str]:
        out = []
        for o in self.overlaps:
            if o.ok:
                continue
            if o.status == "uncovered":
                out.append(f"{o.benchmark_id}: no manifest and no decon_exempt_reason")
            elif o.coverage_fail:
                have = (
                    f"{o.hashed_n} hashed manifest rows / {o.eval_n or 'no'} registry eval images"
                )
                out.append(
                    f"{o.benchmark_id}: manifest coverage below gate_manifest_min_coverage ({have})"
                )
            if o.review_band_fail:
                out.append(f"{o.benchmark_id}: review-band hits exceed the max")
            if any(n for stages in o.counts.values() for n in stages.values()):
                out.append(f"{o.benchmark_id}: confirmed contamination — {o.counts}")
        return out


def _entropy_bits(loader: Callable[[], bytes] | None) -> float | None:
    """8-bit grayscale Shannon entropy of the whole image — decon's own low-texture
    guard (§2.1 S3); WP-10's ``margin``/``lowtex`` measures a different, cheaper thing
    (dHash-thumbnail contrast) and is not reused here."""
    if loader is None:
        return None
    from PIL import Image

    with Image.open(__import__("io").BytesIO(loader())) as img:
        gray = np.asarray(img.convert("L"))
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


def _exact_hits(
    bench: list[ImageRecord], corpus: list[ImageRecord], key: str
) -> list[tuple[int, int]]:
    by_key: dict[Any, list[int]] = defaultdict(list)
    for ci, c in enumerate(corpus):
        v = getattr(c, key)
        if v is not None:
            by_key[v].append(ci)
    pairs = []
    for bi, b in enumerate(bench):
        v = getattr(b, key)
        if v is None:
            continue
        pairs.extend((bi, ci) for ci in by_key.get(v, ()))
    return pairs


def _dhash_candidates(
    bench: list[ImageRecord], corpus: list[ImageRecord], radius: int
) -> list[tuple[int, int]]:
    if not bench or not corpus:
        return []
    index = MultiIndexHamming.build(np.array([c.dhash for c in corpus], dtype=np.uint64))
    queries = np.array([b.dhash for b in bench], dtype=np.uint64)
    qi, ci, _ = unique_pairs(index.search(queries, radius), len(corpus))
    return list(zip(qi.tolist(), ci.tolist(), strict=True))


def _embed_candidates(
    bench: list[ImageRecord], corpus: list[ImageRecord], k: int, min_cos: float
) -> list[tuple[int, int]]:
    have_b = [i for i, b in enumerate(bench) if b.embedding is not None]
    have_c = [i for i, c in enumerate(corpus) if c.embedding is not None]
    if not have_b or not have_c:
        return []
    be = np.stack([bench[i].embedding for i in have_b])  # type: ignore[arg-type]
    ce = np.stack([corpus[i].embedding for i in have_c])  # type: ignore[arg-type]
    sims = be @ ce.T
    kk = min(k, sims.shape[1])
    idx = np.argpartition(-sims, kk - 1, axis=1)[:, :kk]
    rows = np.arange(len(have_b))[:, None].repeat(kk, axis=1)
    keep = sims[rows, idx] >= min_cos
    pairs = []
    for r, cset in zip(rows[keep], idx[keep], strict=True):
        pairs.append((have_b[r], have_c[cset]))
    return pairs


def classify_pair(
    bench: ImageRecord, corp: ImageRecord, thresholds: Thresholds, dedup_crop: bool = False
) -> tuple[str, str, float]:
    """``(stage, band, signal)``; ``band`` is ``confirmed``, ``review`` or ``""``."""
    if bench.upstream_id is not None and bench.upstream_id == corp.upstream_id:
        return "S0", "confirmed", 1.0
    if bench.sha256 == corp.sha256:
        return "S1", "confirmed", 1.0
    if bench.pixel_sha256 == corp.pixel_sha256:
        return "S2", "confirmed", 1.0

    d_dhash = int(bin(bench.dhash ^ corp.dhash).count("1"))
    if d_dhash <= thresholds.s3_dhash64_candidate_max:
        d_phash = phash_distance(bench.phash, corp.phash)
        if d_phash <= thresholds.s3_phash256_confirm_max:
            raw = (_entropy_bits(bench.loader), _entropy_bits(corp.loader))
            samples = [x for x in raw if x is not None]
            entropy = min(samples) if samples else None
            if entropy is None or entropy >= thresholds.s3_min_entropy_bits:
                return "S3", "confirmed", float(d_phash)
            # low-texture: perceptual hash proves nothing here, fall through to S4
        elif d_phash <= thresholds.s3_phash256_review_max:
            return "S3", "review", float(d_phash)

    if bench.embedding is not None and corp.embedding is not None:
        cos = float(np.dot(bench.embedding, corp.embedding))
        tau = thresholds.tau_decon
        if cos >= tau:
            return "S4", "confirmed", cos
        if cos >= tau - thresholds.s4_review_width:
            return "S4", "review", cos

    # D-T: the crop/patch stage is WP-10's crop channel applied to decon; it stays off
    # unless the caller opts in (``--dedup-crop``, default off — same switch, same
    # default, as WP-10c landed it for corpus-internal dedup).
    if dedup_crop and bench.loader is not None and corp.loader is not None:
        area_ratio = min(bench.area, corp.area) / max(max(bench.area, corp.area), 1)
        if area_ratio <= 0.95:
            from .dedup.confirm import ConfirmRules

            rules = ConfirmRules()
            scales = [s for s in DEFAULT_SCALES if s >= rules.crop_scale_min] or None
            patch_rec, parent_rec = (bench, corp) if bench.area <= corp.area else (corp, bench)
            patch_img = thumb_pair(patch_rec.loader()).gray  # type: ignore[misc]
            parent_img = thumb_pair(parent_rec.loader()).gray  # type: ignore[misc]
            ncc = match_patch(patch_img, parent_img, scales=scales).score
            if ncc >= thresholds.s5_parent_ncc_min:
                return "S5", "confirmed", ncc
            if ncc >= thresholds.s5_parent_ncc_review_min:
                return "S5", "review", ncc
    return "", "", 0.0


def check_benchmark(
    entry: BenchmarkEntry,
    thresholds: Thresholds,
    bench_records: list[ImageRecord],
    corpus_records: list[ImageRecord],
    dedup_crop: bool = False,
) -> BenchmarkOverlap:
    # WP-R9: the denominator is the registry's eval-split image count, never the manifest under
    # judgement (R8c: 8.5k of 28353 deepseagrass images read as "100%"). A staged/w1/w2
    # benchmark whose registry has no such count cannot be verified and fails.
    eval_n = expected_eval_count(entry)
    # WP-R12: the numerator is the UNIQUE non-null sha256 in the manifest, capped at the registry
    # eval count (<= 100%). Duplicate rows (QA rows per image, repeated images) never raise
    # coverage: marineeval has 2672 rows for 2643 unique images.
    hashed_n = len({r.sha256 for r in bench_records if r.sha256})
    if eval_n is not None:
        hashed_n = min(hashed_n, eval_n)
    coverage_fail = entry.obtain.status in {"staged", "w1", "w2"} and (
        hashed_n == 0
        or (eval_n is None and bool(bench_records))
        or (eval_n is not None and hashed_n < thresholds.gate_manifest_min_coverage * eval_n)
    )
    eval_n = eval_n or 0
    # A reviewed reason waives the coverage gate for a benchmark that has a manifest (hit
    # detection below still runs); one with neither a count nor a reason fails as before.
    partial_reason = (entry.decon_exempt_reason or "") if coverage_fail else ""
    coverage_fail = coverage_fail and not partial_reason

    candidates: set[tuple[int, int]] = set()
    candidates.update(_exact_hits(bench_records, corpus_records, "upstream_id"))
    candidates.update(_exact_hits(bench_records, corpus_records, "sha256"))
    candidates.update(_exact_hits(bench_records, corpus_records, "pixel_sha256"))
    candidates.update(
        _dhash_candidates(bench_records, corpus_records, thresholds.s3_dhash64_candidate_max)
    )
    # S4 candidates: embedding kNN at the review floor (bounded by ``s4_ann_top_k`` per
    # benchmark image). S5 reuses WP-10's own crop-candidate cosine floor
    # (:class:`~marinedata.dedup.confirm.ConfirmRules.cos_crop`) — well below
    # ``tau_decon``, since a crop's cosine rarely clears the copy bar — so its patch
    # matcher runs on a real crop candidate, never a brute area cross-join.
    from .dedup.confirm import ConfirmRules

    candidates.update(
        _embed_candidates(
            bench_records,
            corpus_records,
            thresholds.s4_ann_top_k,
            thresholds.tau_decon - thresholds.s4_review_width,
        )
    )
    if dedup_crop:
        candidates.update(
            _embed_candidates(
                bench_records, corpus_records, thresholds.s4_ann_top_k, ConfirmRules().cos_crop
            )
        )

    best_per_corpus: dict[int, tuple[str, str, float]] = {}
    review: list[ReviewHit] = []
    for bi, ci in candidates:
        b, corp = bench_records[bi], corpus_records[ci]
        stage, band, signal = classify_pair(b, corp, thresholds, dedup_crop=dedup_crop)
        if not stage:
            continue
        if band == "review":
            for split in corp.splits or {"test"}:
                review.append(ReviewHit(entry.id, stage, corp.sha256, b.sha256, split, signal))
            continue
        prev = best_per_corpus.get(ci)
        if prev is None or _STAGE_RANK[stage] < _STAGE_RANK[prev[0]]:
            best_per_corpus[ci] = (stage, band, signal)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    home_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    excluded = 0
    for ci, (stage, _band, _signal) in best_per_corpus.items():
        corp = corpus_records[ci]
        splits = corp.splits or frozenset({"test"})
        for split in splits:
            if entry.policy == "exclude":
                # §2.2 rule 2: an exclude benchmark fails on ANY hit, in any split,
                # including test — there is no legitimate home for it.
                excluded += 1
                counts[split][stage] += 1
            elif _is_eval_home(split):
                # §2.2 rule 3: a route-to-our-test pin correctly living in test/ood-*
                # is not a failure — reported for the card only.
                home_counts[split][stage] += 1
            else:
                counts[split][stage] += 1

    review_max = max(
        thresholds.gate_review_band_max.get("absolute", 5),
        thresholds.gate_review_band_max.get("fraction_of_eval", 0.01) * (eval_n or hashed_n),
    )
    review_band_fail = len(review) > review_max

    status = "not-checked" if hashed_n == 0 and not coverage_fail else "clean"
    if any(n for stages in counts.values() for n in stages.values()):
        status = "contaminated"

    return BenchmarkOverlap(
        benchmark_id=entry.id,
        task=entry.task,
        eval_n=eval_n,
        hashed_n=hashed_n,
        policy=entry.policy,
        status=status,
        counts={k: dict(v) for k, v in counts.items()},
        home_counts={k: dict(v) for k, v in home_counts.items()},
        excluded=excluded,
        review=tuple(review),
        coverage_fail=coverage_fail,
        review_band_fail=review_band_fail,
        partial_reason=partial_reason,
    )


def load_release_pool(release_dir: str | Path) -> dict[str, frozenset[str]]:
    """``sha256 -> splits it ships in``, from every ``tasks/*.tsv`` under ``release_dir``."""
    release_dir = Path(release_dir)
    tasks_dir = release_dir / "tasks"
    pool: dict[str, set[str]] = defaultdict(set)
    for tsv in sorted(tasks_dir.glob("*.tsv")) if tasks_dir.is_dir() else []:
        with tsv.open() as fh:
            next(fh, None)  # header
            for line in fh:
                sha256, _, split = line.rstrip("\n").partition("\t")
                if sha256:
                    pool[sha256].add(split)
    return {k: frozenset(v) for k, v in pool.items()}


def load_benchmark_records(
    manifest_path: Path, image_root: Path | None = None
) -> list[ImageRecord]:
    import pyarrow.parquet as pq

    if not manifest_path.is_file():
        return []
    rows = pq.read_table(manifest_path).to_pylist()
    out = []
    for r in rows:
        loader = None
        if image_root is not None:
            candidate = image_root / r["upstream_path"]
            if candidate.is_file():
                loader = _file_loader(candidate)
        out.append(
            ImageRecord(
                sha256=r["sha256"],
                pixel_sha256=r["pixel_sha256"],
                dhash=int(r["dhash"], 16),
                phash=bytes.fromhex(r["phash"]),
                phash64=int(r["phash64"], 16),
                margin=float(r["margin"]),
                area=int(r["width"]) * int(r["height"]),
                upstream_id=r["stem"],
                loader=loader,
            )
        )
    return out


def build_corpus_records(
    items: list[Item], pool: dict[str, frozenset[str]], workers: int = 4
) -> list[ImageRecord]:
    wanted = [it for it in items if it.sha256 in pool]
    if not wanted:
        return []
    feats = hash_items(wanted, workers=workers, log=lambda *_a: None)
    out = []
    for it, feat in zip(wanted, feats, strict=True):
        if isinstance(feat, str):
            continue
        loader = _payload_loader(it.payload)
        out.append(
            ImageRecord(
                sha256=feat.sha256,
                pixel_sha256=feat.pixel_sha256,
                dhash=feat.dhash,
                phash=feat.phash,
                phash64=feat.phash64,
                margin=feat.margin,
                area=feat.width * feat.height,
                upstream_id=it.record_id,
                splits=pool[it.sha256],
                loader=loader,
            )
        )
    return out


def add_embeddings(
    records: list[ImageRecord], weights: Path, cache_path: Path
) -> list[ImageRecord]:
    """Attach SSCD embeddings in place (returns new records; ``torch`` loaded lazily)."""
    from .dedup.embed import EmbeddingCache, SSCDEmbedder

    needing = [r for r in records if r.loader is not None]
    if not needing:
        return records
    embedder = SSCDEmbedder(weights)
    cache = EmbeddingCache(cache_path)

    def _load(key: str) -> np.ndarray:
        data = next(r.loader() for r in needing if r.sha256 == key)  # type: ignore[misc]
        from PIL import Image

        with Image.open(__import__("io").BytesIO(data)) as img:
            return prepare(img, EMBED_SIDE)

    keys = [r.sha256 for r in needing]
    vecs = embedder.embed_keys(keys, _load, cache=cache)
    by_sha = dict(zip(keys, vecs, strict=True))
    return [r if r.sha256 not in by_sha else _with_embedding(r, by_sha[r.sha256]) for r in records]


def _with_embedding(r: ImageRecord, vec: np.ndarray) -> ImageRecord:
    from dataclasses import replace

    return replace(r, embedding=vec)


def check(
    release_dir: str | Path,
    registry: BenchmarkRegistry,
    corpus_items: list[Item],
    *,
    manifests_root: Path | None = None,
    image_roots: dict[str, Path] | None = None,
    embed_weights: Path | None = None,
    embed_cache: Path | None = None,
    workers: int = 4,
    dedup_crop: bool = False,
) -> DeconResult:
    """Run the S0-S5 gate for every benchmark with a manifest against ``release_dir``.

    A benchmark with no manifest passes only with a reviewed ``decon_exempt_reason``
    (status ``exempt``, recorded in ``RELEASE.json`` and the dataset card); with neither
    it is ``uncovered`` and fails coverage. A short manifest on a staged/w1/w2 benchmark
    fails coverage too (§2.2 rule 4).

    ``dedup_crop`` (D-T, default off): gates the S5 patch/crop stage, same switch and
    same default as WP-10c's ``--dedup-crop`` for corpus-internal dedup.
    """
    pool = load_release_pool(release_dir)
    corpus_records = build_corpus_records(corpus_items, pool, workers=workers)
    if embed_weights is not None and embed_cache is not None:
        corpus_records = add_embeddings(corpus_records, embed_weights, embed_cache)

    overlaps = []
    for entry in registry.benchmarks:
        manifest_path = registry.manifest_path(entry.id, manifests_root)
        bench_records = load_benchmark_records(manifest_path, (image_roots or {}).get(entry.id))
        if embed_weights is not None and embed_cache is not None:
            bench_records = add_embeddings(bench_records, embed_weights, embed_cache)
        if not bench_records and not entry.decon_exempt_reason:
            # Default deny: no manifest AND no reviewed exemption fails coverage, whatever
            # the obtain status (an unfetched benchmark must be exempted explicitly).
            overlaps.append(
                BenchmarkOverlap(
                    benchmark_id=entry.id,
                    task=entry.task,
                    eval_n=expected_eval_count(entry) or 0,
                    hashed_n=0,
                    policy=entry.policy,
                    status="uncovered",
                    counts={},
                    home_counts={},
                    excluded=0,
                    review=(),
                    coverage_fail=True,
                    review_band_fail=False,
                    reason="no manifest and no decon_exempt_reason",
                )
            )
            continue
        if not bench_records:
            overlaps.append(
                BenchmarkOverlap(
                    benchmark_id=entry.id,
                    task=entry.task,
                    eval_n=expected_eval_count(entry) or 0,
                    hashed_n=0,
                    policy=entry.policy,
                    status="exempt",
                    counts={},
                    home_counts={},
                    excluded=0,
                    review=(),
                    coverage_fail=False,
                    review_band_fail=False,
                    reason=entry.decon_exempt_reason or "",
                )
            )
            continue
        overlaps.append(
            check_benchmark(
                entry, registry.thresholds, bench_records, corpus_records, dedup_crop=dedup_crop
            )
        )

    return DeconResult(
        overlaps=tuple(overlaps),
        thresholds_sha256=benchmarks_sha256(registry),
        registry_sha256=benchmarks_sha256(registry),
    )


def overlap_table_md(result: DeconResult, registry: BenchmarkRegistry) -> str:
    header = (
        "| benchmark | task | eval split (n) | hashed | policy | train S0-2/S3/S4/S5 "
        "| val | train-only | test | ood-* | excluded | status |"
    )
    sep = "|---" * 12 + "|"
    lines = [header, sep]
    for o in result.overlaps:
        entry = registry.by_id(o.benchmark_id)
        train = o.counts.get("train", {})
        s012 = train.get("S0", 0) + train.get("S1", 0) + train.get("S2", 0)
        pct = (100 * o.hashed_n / o.eval_n) if o.eval_n else 0
        ood_stages = (stages for s, stages in o.home_counts.items() if s.startswith("ood-"))
        ood_n = sum(v for stages in ood_stages for v in stages.values())
        row = (
            f"| {o.benchmark_id} | {o.task} | {entry.upstream_split.eval_split} ({o.eval_n}) "
            f"| {o.hashed_n} ({pct:.0f}%) | {o.policy} "
            f"| {s012}/{train.get('S3', 0)}/{train.get('S4', 0)}/{train.get('S5', 0)} "
            f"| {sum(o.counts.get('val', {}).values())} "
            f"| {sum(o.counts.get('train-only', {}).values())} "
            f"| {sum(o.home_counts.get('test', {}).values())} "
            f"| {ood_n} "
            f"| {o.excluded} | {o.status} |"
        )
        lines.append(row)
    lines.append(
        f"\nThresholds: S3 <= {registry.thresholds.s3_phash256_confirm_max} / "
        f"S4 >= {registry.thresholds.tau_decon:.2f} {registry.thresholds.s4_embedding_model} / "
        f"S5 NCC >= {registry.thresholds.s5_parent_ncc_min}. "
        f"Registry sha256: {result.registry_sha256}."
    )
    not_checked = [o for o in result.overlaps if o.status in {"not-checked", "exempt", "uncovered"}]
    if not_checked:
        lines.append("\nNot checked (exempt = reviewed reason; uncovered = gate failure):")
        for o in not_checked:
            lines.append(f"- {o.benchmark_id} [{o.status}]: {o.reason}")
    partial = [o for o in result.overlaps if o.partial]
    if partial:
        lines.append("\nPartially verified (hit detection ran; coverage not verifiable):")
        for o in partial:
            lines.append(
                f"- {o.benchmark_id} [{o.status}, {o.hashed_n} hashed]: {o.partial_reason}"
            )
    return "\n".join(lines) + "\n"


def write_outputs(result: DeconResult, registry: BenchmarkRegistry, out_dir: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for o in result.overlaps:
        for is_home, counts in ((False, o.counts), (True, o.home_counts)):
            for split, stages in counts.items():
                for stage, n in stages.items():
                    rows.append(
                        {
                            "benchmark_id": o.benchmark_id,
                            "split": split,
                            "stage": stage,
                            "count": n,
                            "is_eval_home": is_home,
                        }
                    )
    table = pa.table(
        {
            "benchmark_id": [r["benchmark_id"] for r in rows],
            "split": [r["split"] for r in rows],
            "stage": [r["stage"] for r in rows],
            "count": [r["count"] for r in rows],
            "is_eval_home": [r["is_eval_home"] for r in rows],
        }
    )
    pq.write_table(table, out_dir / "overlap.parquet")
    (out_dir / "overlap.md").write_text(overlap_table_md(result, registry))


def decon_record(result: DeconResult) -> dict[str, Any]:
    """The ``RELEASE.json`` ``decon`` block: what was verified, and what was exempted."""
    exempt = result.exempt
    return {
        "gate": "pass" if result.ok else "fail",
        "benchmarks_checked": sum(1 for o in result.overlaps if o.status not in {"exempt"}),
        "exempt": dict(sorted(exempt.items())),
        "partially_verified": dict(sorted(result.partial.items())),
    }


def run_decon_gate(
    release_root: Path, registry: Any, roots: dict[str, Path], dedup_crop: bool = False
) -> DeconResult:
    """The ``--decon`` hook release.py calls: our own registry's admitted roots, staged.

    ``dedup_crop`` defaults off (D-T); flipping it for the real v2 build (D-T2: crop
    channel on) is the integrator's call, same as ``--decon`` itself.
    """
    from .benchmarks import BenchmarkRegistry as _BR
    from .dedup.corpus import iter_staged

    items: list[Item] = []
    for source_id, root in roots.items():
        items.extend(iter_staged(root, source_id))
    result = check(release_root, _BR.load(), items, dedup_crop=dedup_crop)
    write_outputs(result, _BR.load(), release_root / "decon")
    if not result.ok:
        raise DeconError("; ".join(result.failures))
    return result
