"""Multi-index Hamming search over 64-bit codes (4 x 16-bit bands).

Pigeonhole: if two 64-bit codes differ in ``<= r`` bits, at least one of the four
16-bit bands differs in ``<= r // 4`` bits. Each band is a CSR bucket table
(``order`` + 65,537 offsets), so the index costs ``4 * N * 4`` bytes plus the codes —
about 160 MB at 5M codes. A query probes every band key within ``r // 4`` bits of its
own (1, 17 or 137 probes per band for band radius 0, 1, 2), expands the matching
buckets in chunks of at most ``max_pairs`` candidates, and keeps pairs whose full
64-bit distance is ``<= r``. Peak memory is bounded by ``max_pairs`` regardless of
bucket skew; time is not, so a degenerate bucket (thousands of flat images sharing one
band key) shows up as time, never as an out-of-memory.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations

import numpy as np

BANDS = 4
BAND_BITS = 16
_KEYS = 1 << BAND_BITS
_MASK = np.uint64(_KEYS - 1)


def probe_masks(radius: int) -> np.ndarray:
    """Every 16-bit XOR mask with popcount ``<= radius``."""
    masks = [0]
    for r in range(1, radius + 1):
        for bits in combinations(range(BAND_BITS), r):
            masks.append(sum(1 << b for b in bits))
    return np.array(masks, dtype=np.uint16)


@dataclass
class MultiIndexHamming:
    codes: np.ndarray  # uint64 (N,)
    orders: list[np.ndarray]  # per band: int32 row order sorted by band key
    offsets: list[np.ndarray]  # per band: int64 (65537,) CSR offsets

    @classmethod
    def build(cls, codes: np.ndarray) -> MultiIndexHamming:
        codes = np.ascontiguousarray(codes, dtype=np.uint64)
        if len(codes) >= 2**31:
            raise ValueError("MultiIndexHamming holds at most 2**31 - 1 codes")
        orders, offsets = [], []
        for band in range(BANDS):
            key = ((codes >> np.uint64(band * BAND_BITS)) & _MASK).astype(np.uint16)
            order = np.argsort(key, kind="stable").astype(np.int32)
            counts = np.bincount(key, minlength=_KEYS)
            offs = np.zeros(_KEYS + 1, dtype=np.int64)
            np.cumsum(counts, out=offs[1:])
            orders.append(order)
            offsets.append(offs)
        return cls(codes=codes, orders=orders, offsets=offsets)

    def __len__(self) -> int:
        return len(self.codes)

    def nbytes(self) -> int:
        return int(
            self.codes.nbytes
            + sum(o.nbytes for o in self.orders)
            + sum(o.nbytes for o in self.offsets)
        )

    def max_bucket(self) -> int:
        return int(max(np.diff(o).max() for o in self.offsets)) if len(self) else 0

    def search(
        self,
        queries: np.ndarray,
        radius: int,
        *,
        self_join: bool = False,
        max_pairs: int = 8_000_000,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield ``(query_idx, index_idx, distance)`` chunks with distance ``<= radius``.

        A pair can be yielded by more than one band; :func:`unique_pairs` removes the
        repeats. ``self_join=True`` means ``queries`` *is* the indexed array and only
        ``query_idx < index_idx`` is kept.
        """
        queries = np.ascontiguousarray(queries, dtype=np.uint64)
        masks = probe_masks(radius // BANDS)
        for band in range(BANDS):
            shift = np.uint64(band * BAND_BITS)
            qkey = ((queries >> shift) & _MASK).astype(np.uint16)
            order, offs = self.orders[band], self.offsets[band]
            for mask in masks.tolist():
                probe = qkey ^ np.uint16(mask)
                starts = offs[probe]
                counts = offs[probe.astype(np.int64) + 1] - starts
                yield from self._expand(
                    queries, starts, counts, order, radius, self_join, max_pairs
                )

    def _expand(self, queries, starts, counts, order, radius, self_join, max_pairs):  # type: ignore[no-untyped-def]
        live = np.nonzero(counts)[0]
        if not len(live):
            return
        csum = np.cumsum(counts[live])
        lo = 0
        while lo < len(live):
            base = csum[lo - 1] if lo else 0
            hi = int(np.searchsorted(csum, base + max_pairs, side="right"))
            hi = max(hi, lo + 1)
            qsel = live[lo:hi]
            cnt = counts[qsel]
            total = int(cnt.sum())
            qi = np.repeat(qsel, cnt)
            first = np.repeat(np.cumsum(cnt) - cnt, cnt)
            ti = order[np.repeat(starts[qsel], cnt) + (np.arange(total) - first)].astype(np.int64)
            dist = np.bitwise_count(queries[qi] ^ self.codes[ti]).astype(np.int16)
            keep = dist <= radius
            if self_join:
                keep &= qi < ti
            if keep.any():
                yield qi[keep].astype(np.int64), ti[keep], dist[keep]
            lo = hi


def unique_pairs(
    chunks: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]], n_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate :meth:`MultiIndexHamming.search` chunks and drop repeated pairs."""
    qs, ts, ds = [], [], []
    for q, t, d in chunks:
        qs.append(q)
        ts.append(t)
        ds.append(d)
    if not qs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0, dtype=np.int16)
    q = np.concatenate(qs)
    t = np.concatenate(ts)
    d = np.concatenate(ds)
    _, first = np.unique(q * np.int64(n_index) + t, return_index=True)
    return q[first], t[first], d[first]
