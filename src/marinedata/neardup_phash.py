"""The gates' DCT pHash-64 over the eval-capable images, as split-map links (v1.1.0).

The dedup gate (``near_dup_across_splits``) and the splits gate (``near_dup_leak``) fail a build
when two images of a config, in different splits, sit within ``DEFAULT_PHASH_BITS`` bits of
:func:`marinedata.gates.phash.phash`. Rule B (dHash <= 4 plus the pHash-256 confirm) does not imply
that: run 5 shipped 70 such pairs across splits. :func:`phash_links` hashes the way the gates do and
returns every pair within the radius, for ``generate_split_map`` to union into one component under
the chain guard rule B already uses (:class:`marinedata.neardup.NearDupChainError`).

The hashes are memoised like :func:`marinedata.neardup.compute_dhashes`: a sqlite file under the
cache dir keyed by image sha256, one file per Pillow version (the resize is a Pillow detail).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .neardup import NearDupError, near_pairs, pil_version

_COMMIT_EVERY = 2000


def _phash_worker(path: str) -> tuple[int | None, str]:
    """``(hash, "")`` for one file, or ``(None, error)``: a bad image names itself, not the pool."""
    try:
        from PIL import Image

        from .gates.phash import phash

        with Image.open(path) as image:
            image.load()
            return phash(image), ""
    except Exception as exc:  # reported with the file name by compute_phashes
        return None, f"{path}: {type(exc).__name__}: {exc}"


def _open_cache(cache_dir: Path) -> sqlite3.Connection:
    cache_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(cache_dir / f"phash-pillow-{pil_version()}.sqlite")
    conn.execute("CREATE TABLE IF NOT EXISTS phash (sha256 TEXT PRIMARY KEY, phash TEXT NOT NULL)")
    return conn


def compute_phashes(
    paths: Mapping[str, Path], *, cache_dir: str | Path, workers: int = 1
) -> dict[str, int]:
    """``{image_sha256: DCT pHash-64}`` for every entry of ``paths`` (sha256 -> any file holding
    those bytes). Cache hits are read back; misses are hashed (``workers`` processes) and written
    through. Raises :class:`~marinedata.neardup.NearDupError` naming every file that failed."""
    conn = _open_cache(Path(cache_dir))
    try:
        cached = {sha: int(h, 16) for sha, h in conn.execute("SELECT sha256, phash FROM phash")}
        missing = sorted(sha for sha in paths if sha not in cached)
        failures: list[str] = []
        pending: list[tuple[str, str]] = []

        def flush() -> None:
            conn.executemany("INSERT OR REPLACE INTO phash VALUES (?, ?)", pending)
            conn.commit()
            pending.clear()

        files = [str(paths[sha]) for sha in missing]
        if workers > 1 and len(files) > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results: Iterable[tuple[int | None, str]] = list(
                    pool.map(_phash_worker, files, chunksize=64)
                )
        else:
            results = (_phash_worker(f) for f in files)
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
        raise NearDupError(f"{len(failures)} image(s) could not be pHashed: {shown}")
    return {sha: cached[sha] for sha in paths}


def phash_links(
    paths: Mapping[str, Path], *, cache_dir: str | Path, workers: int = 1, max_distance: int
) -> list[tuple[str, str]]:
    """Every pair ``(a, b)`` of ``paths`` whose pHash-64 are within ``max_distance`` bits, sorted:
    the images the gates call near-duplicates, for ``generate_split_map`` to union."""
    hashes = compute_phashes(paths, cache_dir=cache_dir, workers=workers)
    return [(a, b) for a, b, _ in near_pairs(hashes, max_distance)]
