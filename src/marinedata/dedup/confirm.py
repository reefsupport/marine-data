"""Candidate generation and confirmation over a table of :class:`ImageFeatures`.

Channels (each yields candidate pairs; the union is scored once):

* ``pixel`` — equal decoded-pixel sha256 (covers exact byte duplicates too).
* ``dhash`` / ``phash64`` — :class:`MultiIndexHamming` at ``radius``, probing the
  identity, horizontal-flip and vertical-flip variants of every query.
* ``embed`` — optional SSCD cosine kNN; the only channel that reaches 60 % crops.

Confirmation (:class:`ConfirmRules`): exact/pixel always; otherwise the SSCD cosine
must clear ``cos_copy`` on its own, or clear ``cos_agree`` while a perceptual hash
agrees (``d_dhash <= dhash_agree`` or ``d_phash <= phash_agree``). The texture guard:
if either image is low-texture (:data:`features.LOWTEX_MARGIN`), a perceptual hash
proves nothing, so only ``cos >= cos_lowtex`` (or exact/pixel) confirms.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import ImageFeatures
from .mih import MultiIndexHamming, unique_pairs

HASH_FIELDS = ("dhash", "dhash_h", "dhash_v", "phash64", "phash64_h", "phash64_v")
PHASH_FIELDS = ("phash", "phash_h", "phash_v")


@dataclass(frozen=True)
class ConfirmRules:
    radius: int = 8
    cos_copy: float = 0.75
    cos_agree: float = 0.50
    cos_lowtex: float = 0.85
    dhash_agree: int = 8
    phash_agree: int = 56
    knn_k: int = 10
    knn_min_cos: float = 0.50
    cos_crop: float = 0.50
    ncc_crop: float = 0.50
    crop_area_max: float = 0.95
    crop_scale_min: float = 0.30
    cos_box_crop: float = 0.60

    def record(self) -> dict[str, object]:
        return dict(self.__dict__)


@dataclass
class FeatureTable:
    cols: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_features(cls, feats: list[ImageFeatures]) -> FeatureTable:
        cols: dict[str, np.ndarray] = {
            "sha256": np.array([f.sha256 for f in feats], dtype=object),
            "pixel_sha256": np.array([f.pixel_sha256 for f in feats], dtype=object),
            "margin": np.array([f.margin for f in feats], dtype=np.float32),
            "lowtex": np.array([f.lowtex for f in feats], dtype=bool),
            "area": np.array([f.width * f.height for f in feats], dtype=np.int64),
        }
        for name in HASH_FIELDS:
            cols[name] = np.array([getattr(f, name) for f in feats], dtype=np.uint64)
        for name in PHASH_FIELDS:
            raw = b"".join(getattr(f, name) for f in feats)
            cols[name] = np.frombuffer(raw, dtype=np.uint8).reshape(len(feats), 32)
        return cls(cols)

    @classmethod
    def from_arrow(cls, table) -> FeatureTable:  # type: ignore[no-untyped-def]
        cols: dict[str, np.ndarray] = {
            "sha256": np.array(table.column("sha256").to_pylist(), dtype=object),
            "pixel_sha256": np.array(table.column("pixel_sha256").to_pylist(), dtype=object),
            "margin": table.column("margin").to_numpy().astype(np.float32),
            "lowtex": table.column("lowtex").to_numpy(zero_copy_only=False).astype(bool),
            "area": table.column("width").to_numpy().astype(np.int64)
            * table.column("height").to_numpy().astype(np.int64),
        }
        for name in HASH_FIELDS:
            cols[name] = table.column(name).to_numpy().astype(np.uint64)
        for name in PHASH_FIELDS:
            raw = b"".join(table.column(name).to_pylist())
            cols[name] = np.frombuffer(raw, dtype=np.uint8).reshape(table.num_rows, 32)
        return cls(cols)

    def __len__(self) -> int:
        return len(self.cols["sha256"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.cols[name]


def _norm(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(a, b), np.maximum(a, b)


class DedupIndex:
    """The corpus side: MIH over dHash-64 and pHash-64, optional embeddings."""

    def __init__(self, table: FeatureTable, emb: np.ndarray | None, rules: ConfirmRules) -> None:
        self.t = table
        self.emb = emb
        self.rules = rules
        self.mih = {
            "dhash": MultiIndexHamming.build(table["dhash"]),
            "phash64": MultiIndexHamming.build(table["phash64"]),
        }

    def _hash_candidates(self, q: FeatureTable, self_join: bool) -> tuple[np.ndarray, np.ndarray]:
        qs, ts = [], []
        for key, index in self.mih.items():
            for suffix in ("", "_h", "_v"):
                sj = self_join and suffix == ""
                chunks = index.search(q[key + suffix], self.rules.radius, self_join=sj)
                a, b, _ = unique_pairs(chunks, len(index))
                qs.append(a)
                ts.append(b)
        return np.concatenate(qs), np.concatenate(ts)

    def self_candidates(self, embed_knn: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        pix = self.t["pixel_sha256"]
        order = np.argsort(pix, kind="stable")
        same = pix[order][1:] == pix[order][:-1]
        out["pixel"] = _norm(order[:-1][same].astype(np.int64), order[1:][same].astype(np.int64))
        a, b = self._hash_candidates(self.t, self_join=True)
        keep = a != b
        out["hash"] = _norm(a[keep], b[keep])
        if embed_knn and self.emb is not None:
            from .embed import knn_pairs

            i, j, _ = knn_pairs(
                self.emb, self.rules.knn_k, self.rules.knn_min_cos, device=_device()
            )
            out["embed"] = (i, j)
        return out

    def query_candidates(
        self, q: FeatureTable, qemb: np.ndarray | None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Candidates of external queries against the corpus: ``(query_idx, corpus_idx)``."""
        out = {"hash": self._hash_candidates(q, self_join=False)}
        if qemb is not None and self.emb is not None:
            sims = qemb @ self.emb.T
            k = min(self.rules.knn_k, sims.shape[1])
            idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            rows = np.arange(len(q))[:, None].repeat(k, axis=1)
            keep = sims[rows, idx] >= self.rules.knn_min_cos
            out["embed"] = (rows[keep].astype(np.int64), idx[keep].astype(np.int64))
        return out


def _device() -> str:
    try:
        import torch

        return "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def merge_channels(
    channels: dict[str, tuple[np.ndarray, np.ndarray]], n_right: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Union of channel pairs -> ``(a, b, channel_bits)`` (bit ``i`` = ``i``-th channel)."""
    keys, bits = [], []
    for pos, (a, b) in enumerate(channels.values()):
        keys.append(a.astype(np.int64) * np.int64(n_right) + b.astype(np.int64))
        bits.append(np.full(len(a), 1 << pos, dtype=np.int16))
    if not keys:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.int16)
    key = np.concatenate(keys)
    bit = np.concatenate(bits)
    uniq, inv = np.unique(key, return_inverse=True)
    merged = np.zeros(len(uniq), dtype=np.int16)
    np.bitwise_or.at(merged, inv, bit)
    return uniq // n_right, uniq % n_right, merged


def score_pairs(
    left: FeatureTable,
    right: FeatureTable,
    a: np.ndarray,
    b: np.ndarray,
    emb_left: np.ndarray | None,
    emb_right: np.ndarray | None,
) -> dict[str, np.ndarray]:
    """Distances for pairs ``(left[a], right[b])``; orientation minimises dHash distance."""
    rd = right["dhash"][b]
    dd = np.stack(
        [
            np.bitwise_count(left[f][a] ^ rd).astype(np.int16)
            for f in ("dhash", "dhash_h", "dhash_v")
        ]
    )
    orient = dd.argmin(axis=0)
    rp = right["phash"][b]
    dp = np.stack([np.bitwise_count(left[f][a] ^ rp).sum(axis=1) for f in PHASH_FIELDS])
    cos = (
        np.einsum("ij,ij->i", emb_left[a], emb_right[b]).astype(np.float32)
        if emb_left is not None and emb_right is not None
        else np.full(len(a), np.nan, np.float32)
    )
    return {
        "d_dhash": dd.min(axis=0),
        "orient": orient.astype(np.int8),
        "d_phash": dp.min(axis=0).astype(np.int16),
        "cos": cos,
        "exact": left["sha256"][a] == right["sha256"][b],
        "pixel": left["pixel_sha256"][a] == right["pixel_sha256"][b],
        "lowtex": left["lowtex"][a] | right["lowtex"][b],
    }


def confirm(scores: dict[str, np.ndarray], rules: ConfirmRules) -> tuple[np.ndarray, np.ndarray]:
    """``(confirmed, kind)``; kind is exact | pixel | copy | agree | lowtex | '' (rejected)."""
    cos = np.nan_to_num(scores["cos"], nan=-1.0)
    hash_agree = (scores["d_dhash"] <= rules.dhash_agree) | (scores["d_phash"] <= rules.phash_agree)
    lowtex = scores["lowtex"]
    kind = np.full(len(cos), "", dtype=object)
    kind[~lowtex & (cos >= rules.cos_agree) & hash_agree] = "agree"
    kind[~lowtex & (cos >= rules.cos_copy)] = "copy"
    kind[lowtex & (cos >= rules.cos_lowtex)] = "lowtex"
    kind[scores["pixel"]] = "pixel"
    kind[scores["exact"]] = "exact"
    return kind != "", kind
