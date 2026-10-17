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

import heapq
import sqlite3
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .neardup import NearDupError, UnionConfirm, confirm_pairs, near_pairs, pil_version

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


@dataclass(frozen=True)
class PhashSelection:
    """Which DCT pHash-64 pairs ``release split-map`` unions when ``union_confirm`` is set.

    A pHash-64 pair is evidence about two images, but a union merges their whole split groups. In
    run 8 a few generic look-alikes (pHash-64 at 8 bits, pHash-256 76-112 bits apart) between NOAA
    site groups of hundreds of images chained 17 groups into one 1906-image component. A pair is
    unioned when rule B's :class:`~marinedata.neardup.UnionConfirm` accepts it (``confirmed``), or
    when both images sit in single-image groups (``single``: the union moves two images, never a
    site). A pair of two prior-release images is never unioned (``prior_pairs``): their splits are
    frozen, and a union could only withhold the new groups around them. ``pairs`` keeps every
    pair for :func:`phash_exclusions` after assignment.
    """

    pairs: list[tuple[str, str, int]]
    links: list[tuple[str, str]] = field(default_factory=list)
    confirmed: int = 0
    single: int = 0
    prior_pairs: int = 0
    rejected_phash: int = 0
    rejected_entropy: int = 0


def phash_pairs(
    paths: Mapping[str, Path], *, cache_dir: str | Path, workers: int = 1, max_distance: int
) -> list[tuple[str, str, int]]:
    """:func:`phash_links` with the distance: every ``(a, b, bits)``, sorted."""
    hashes = compute_phashes(paths, cache_dir=cache_dir, workers=workers)
    return [(a, b, int(bits)) for a, b, bits in near_pairs(hashes, max_distance)]


def _needs_confirm(a: str, b: str, single: frozenset[str], prior: frozenset[str]) -> bool:
    return not (a in prior and b in prior) and not (a in single and b in single)


def phash_confirm_shas(
    pairs: Iterable[tuple[str, str, int]], *, single: frozenset[str], prior: frozenset[str]
) -> set[str]:
    """The images whose confirm features :func:`select_phash_links` reads."""
    return {sha for a, b, _ in pairs if _needs_confirm(a, b, single, prior) for sha in (a, b)}


def select_phash_links(
    pairs: Iterable[tuple[str, str, int]],
    *,
    features: Mapping[str, tuple[bytes, float]],
    rule: UnionConfirm,
    single: frozenset[str],
    prior: frozenset[str],
) -> PhashSelection:
    """Apply the :class:`PhashSelection` rule. ``single``: images alone in every split group they
    belong to; ``prior``: prior-release images; ``features``: the confirm features of
    :func:`phash_confirm_shas` (rule B's pHash-256 and grey entropy)."""
    found = list(pairs)
    singles = [
        (a, b)
        for a, b, _ in found
        if a in single and b in single and not (a in prior and b in prior)
    ]
    decided = confirm_pairs(
        [pair for pair in found if _needs_confirm(pair[0], pair[1], single, prior)], features, rule
    )
    return PhashSelection(
        pairs=found,
        links=sorted([*singles, *decided.links]),
        confirmed=len(decided.links),
        single=len(singles),
        prior_pairs=sum(1 for a, b, _ in found if a in prior and b in prior),
        rejected_phash=decided.rejected_phash,
        rejected_entropy=decided.rejected_entropy,
    )


def phash_exclusions(
    pairs: Iterable[tuple[str, str, int]],
    splits: Mapping[str, frozenset[str]],
    *,
    protected: frozenset[str] = frozenset(),
) -> tuple[dict[str, tuple[int, str, str]], int]:
    """The images to drop so no pHash-64 pair ships across splits, and the frozen pair count.

    ``splits`` maps every image that ships to the splits it ships in. A pair counts when both
    images ship and their splits differ. A pair of two ``protected`` (prior-release) images is
    frozen: the v1.0 configs are never rewritten, so it is counted and left. Otherwise images are
    dropped greedily, the unprotected image with the most cross-split partners left first, ties by
    sha256: deterministic, close to the fewest images, never a protected one. Each dropped image
    records ``(bits, other_sha256, other_split)`` for its closest partner.

    One pass is enough: dropping an image only removes pairs and moves no group, so no new
    cross-split pair can appear, and the loop ends only when no unprotected image has one left.
    """
    partners: dict[str, dict[str, int]] = {}
    frozen = 0
    for a, b, bits in pairs:
        split_a, split_b = splits.get(a), splits.get(b)
        if not split_a or not split_b or len(split_a | split_b) < 2:
            continue  # an image that ships nowhere, or both in one split
        if a in protected and b in protected:
            frozen += 1
            continue
        for x, y in ((a, b), (b, a)):
            mine = partners.setdefault(x, {})
            mine[y] = min(bits, mine.get(y, bits))
    heap = [(-len(mine), sha) for sha, mine in partners.items() if sha not in protected]
    heapq.heapify(heap)
    excluded: dict[str, tuple[int, str, str]] = {}
    while heap:
        degree, sha = heapq.heappop(heap)
        mine = partners[sha]
        if not mine:
            continue
        if -degree != len(mine):  # stale entry: queue it again with what is left
            heapq.heappush(heap, (-len(mine), sha))
            continue
        bits, other = min((d, o) for o, d in mine.items())
        excluded[sha] = (bits, other, min(splits[other]))
        for o in mine:
            partners[o].pop(sha, None)
        mine.clear()
    return dict(sorted(excluded.items())), frozen
