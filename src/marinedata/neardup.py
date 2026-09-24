"""Near-duplicate detection for release splitting (WS-D S47).

sha256 only catches byte-identical copies. A re-encoded, resized or re-exported copy of
the same photo has a different digest and walked straight past the release's leakage
check (S46: 42 ``coralscop-masks-rs``/eval-split pairs at dHash Hamming <= 4, the top
five at distance 1 — the same photos re-encoded). This module is the perceptual check:

* :func:`dhash_file` — 64-bit difference hash, S46's exact definition so its numbers stay
  comparable: grayscale, LANCZOS resize to 9x8, one bit per horizontally adjacent pixel
  pair (``right > left``), row-major, most significant bit first.
* :func:`compute_dhashes` — hashes cached image bytes, memoised in a sqlite file under
  ``<cache_dir>`` keyed by image sha256. One file per Pillow version: LANCZOS output is
  a Pillow implementation detail, so a hash computed under one version is never reused
  under another.
* :func:`near_pairs` — exact Hamming-radius search by pigeonhole on bit bands. Split the
  64 bits into ``k + 1`` contiguous bands; two hashes within distance ``k`` agree exactly
  on at least one band, so only same-bucket candidates are ever compared (vectorised
  popcount). Never an all-pairs matrix. Output is sorted — deterministic.

The thresholds are release policy (manager decision, WS-D S47), recorded verbatim in
``SPLIT_MAP.json`` and ``RELEASE.json`` by :func:`near_dup_record`.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

DHASH_ALGORITHM = "dhash-64: grayscale L, LANCZOS resize 9x8, right>left, row-major, MSB first"
UNION_MAX_HAMMING = 4
"""Eval-capable rows this close are one component in split-map generation (rule B)."""
NEVER_EVAL_EXCLUDE_MAX_HAMMING = 8
"""A never-eval row this close to any eval-capable row is dropped from the release (A)."""
CHAIN_GUARD_FRACTION = 0.01
"""Largest near-dup-merged component, as a fraction of all images, before generation
refuses to continue (dHash links are not transitive-safe: A~B~C can chain)."""

_BLOCK_CELLS = 4_000_000
_COMMIT_EVERY = 5_000


class NearDupError(ValueError):
    """An image could not be hashed — fail closed, the near-dup guarantee has a hole."""


class NearDupChainError(NearDupError):
    """Near-dup unions chained into a component larger than the guard allows."""


@dataclass(frozen=True)
class NearDupConfig:
    """Turns rules (A) and (B) on for ``generate_split_map`` / ``build_release``."""

    cache_dir: Path
    union_max: int = UNION_MAX_HAMMING
    exclude_max: int = NEVER_EVAL_EXCLUDE_MAX_HAMMING
    chain_fraction: float = CHAIN_GUARD_FRACTION
    workers: int = 1


def pil_version() -> str:
    import PIL

    return str(PIL.__version__)


def near_dup_record(config: NearDupConfig) -> dict[str, object]:
    """What a map/release records about the check that shaped it."""
    return {
        "algorithm": DHASH_ALGORITHM,
        "pil_version": pil_version(),
        "union_max_hamming": config.union_max,
        "never_eval_exclude_max_hamming": config.exclude_max,
        "chain_guard_fraction": config.chain_fraction,
    }


def dhash_file(path: str | Path) -> int:
    """64-bit dHash of one image file (see module docstring for the exact definition)."""
    from PIL import Image

    try:
        with Image.open(path) as img:
            small = img.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = small.tobytes()
    except Exception as exc:  # any decode failure is a hole in the guarantee
        raise NearDupError(f"cannot dHash {path}: {exc}") from exc
    value = 0
    for row in range(8):
        base = row * 9
        for col in range(8):
            value = (value << 1) | int(pixels[base + col + 1] > pixels[base + col])
    return value


def _dhash_worker(path: str) -> tuple[int | None, str]:
    try:
        return dhash_file(path), ""
    except NearDupError as exc:
        return None, str(exc)


def _cache_path(cache_dir: Path) -> Path:
    return Path(cache_dir) / f"dhash-pillow-{pil_version()}.sqlite"


def _open_cache(cache_dir: Path) -> sqlite3.Connection:
    path = _cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS dhash (sha256 TEXT PRIMARY KEY, dhash TEXT NOT NULL)")
    return conn


def compute_dhashes(
    paths: Mapping[str, Path], *, cache_dir: str | Path, workers: int = 1
) -> dict[str, int]:
    """``{image_sha256: dhash}`` for every entry of ``paths`` (sha256 -> any file holding
    those bytes). Cache hits are read back; misses are hashed (``workers`` processes) and
    written through. Raises :class:`NearDupError` naming every file that failed."""
    conn = _open_cache(Path(cache_dir))
    try:
        cached = {sha: int(h, 16) for sha, h in conn.execute("SELECT sha256, dhash FROM dhash")}
        missing = sorted(sha for sha in paths if sha not in cached)
        failures: list[str] = []
        pending: list[tuple[str, str]] = []

        def flush() -> None:
            conn.executemany("INSERT OR REPLACE INTO dhash VALUES (?, ?)", pending)
            conn.commit()
            pending.clear()

        files = [str(paths[sha]) for sha in missing]
        if workers > 1 and len(files) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results: Iterable[tuple[int | None, str]] = list(
                    pool.map(_dhash_worker, files, chunksize=64)
                )
        else:
            results = (_dhash_worker(f) for f in files)
        for sha, (value, error) in zip(missing, results, strict=True):
            if value is None:
                failures.append(error)
                continue
            cached[sha] = value
            pending.append((sha, f"{value:016x}"))
            if len(pending) >= _COMMIT_EVERY:
                flush()
        flush()
    finally:
        conn.close()
    if failures:
        shown = "; ".join(failures[:5])
        raise NearDupError(f"{len(failures)} image(s) could not be dHashed: {shown}")
    return {sha: cached[sha] for sha in paths}


def _bands(max_distance: int) -> list[tuple[int, int]]:
    """``(shift, width)`` for ``max_distance + 1`` contiguous bands covering 64 bits."""
    count = max_distance + 1
    if not 1 <= count <= 64:
        raise ValueError(f"max_distance must be in 0..63, got {max_distance}")
    base, extra = divmod(64, count)
    bands, shift = [], 64
    for index in range(count):
        width = base + (1 if index < extra else 0)
        shift -= width
        bands.append((shift, width))
    return bands


def _buckets(values):  # type: ignore[no-untyped-def]
    import numpy as np

    order = np.argsort(values, kind="stable")
    uniq, start, count = np.unique(values[order], return_index=True, return_counts=True)
    return order, uniq, start, count


def near_pairs(
    query: Mapping[str, int], max_distance: int, target: Mapping[str, int] | None = None
) -> list[tuple[str, str, int]]:
    """Every ``(query_key, target_key, hamming)`` with distance <= ``max_distance``.

    ``target=None`` is a self-join: pairs ``(a, b)`` with ``a < b`` only. A cross-join
    keeps every match, including a key present on both sides (distance 0 to itself).
    Sorted by ``(a, b)``.
    """
    import numpy as np

    q_keys = sorted(query)
    q_hash = np.array([query[k] for k in q_keys], dtype=np.uint64)
    self_join = target is None
    t_keys = q_keys if self_join else sorted(target or {})
    t_hash = q_hash if self_join else np.array([(target or {})[k] for k in t_keys], dtype=np.uint64)
    if not len(q_keys) or not len(t_keys):
        return []
    found: dict[tuple[int, int], int] = {}
    for shift, width in _bands(max_distance):
        mask, sh = np.uint64((1 << width) - 1), np.uint64(shift)
        q_order, q_vals, q_start, q_count = _buckets((q_hash >> sh) & mask)
        t_order, t_vals, t_start, t_count = _buckets((t_hash >> sh) & mask)
        _, qi, ti = np.intersect1d(q_vals, t_vals, assume_unique=True, return_indices=True)
        for a, b in zip(qi.tolist(), ti.tolist(), strict=True):
            q_idx = q_order[q_start[a] : q_start[a] + q_count[a]]
            t_idx = t_order[t_start[b] : t_start[b] + t_count[b]]
            step = max(1, _BLOCK_CELLS // len(t_idx))
            for lo in range(0, len(q_idx), step):
                q_part = q_idx[lo : lo + step]
                dist = np.bitwise_count(q_hash[q_part][:, None] ^ t_hash[t_idx][None, :])
                rows, cols = np.nonzero(dist <= max_distance)
                for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                    i, j = int(q_part[r]), int(t_idx[c])
                    if self_join and i >= j:
                        continue
                    found[(i, j)] = int(dist[r, c])
    pairs = [(q_keys[i], t_keys[j], d) for (i, j), d in found.items()]
    pairs.sort()
    return pairs


def default_workers() -> int:
    return max(1, min(8, (os.cpu_count() or 2) - 1))
