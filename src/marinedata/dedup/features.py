"""Per-image dedup features: exact and decoded-pixel sha256, dHash-64, pHash-256, texture.

One decode yields everything the v2 dedup stage keys on:

* ``sha256`` of the encoded bytes (exact duplicates) and ``pixel_sha256`` of the
  decoded RGB pixels plus the size (re-encodes of identical pixels, e.g. PNG <-> BMP,
  metadata-only rewrites).
* ``dhash`` — bit-identical to v1's :func:`marinedata.neardup.dhash_file` (grayscale
  ``L``, LANCZOS 9x8, right > left, row-major, MSB first), so v1 rule-A/rule-B pairs
  can be replayed against v2 decisions. ``dhash_h``/``dhash_v`` are the dHash of the
  horizontally/vertically flipped image, derived from the same 9x8 thumbnail.
* ``phash`` — 256 bits: 2-D DCT-II of the 64x64 LANCZOS grayscale, the 16x16
  lowest-frequency block thresholded at its median (DC excluded from the median).
  ``phash64`` is the 8x8 lowest block thresholded the same way; it is the second
  multi-index key. Flip variants are exact sign flips of odd DCT columns/rows.
* ``margin`` — mean absolute difference of the 72 horizontally adjacent pixel pairs
  in the 9x8 dHash thumbnail, in grey levels. A small margin means the dHash bits are
  decided by noise; that is the mechanism behind the v1 low-texture false merges.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

FEATURES_VERSION = "dedup-v2.1"
LOWTEX_MARGIN = 3.0  # grey levels; below this a dHash bit flips on JPEG noise alone
_PHASH_SIDE = 64
_PHASH_BLOCK = 16
_P64_BLOCK = 8


class FeatureError(ValueError):
    """An image that cannot be decoded — a hole in the dedup guarantee, never skipped silently."""


@dataclass(frozen=True)
class ImageFeatures:
    sha256: str
    pixel_sha256: str
    width: int
    height: int
    dhash: int
    dhash_h: int
    dhash_v: int
    phash: bytes  # 32 bytes, MSB-first
    phash_h: bytes
    phash_v: bytes
    phash64: int
    phash64_h: int
    phash64_v: int
    margin: float
    gray_std: float

    @property
    def lowtex(self) -> bool:
        return self.margin < LOWTEX_MARGIN

    def as_row(self) -> dict[str, object]:
        row = {k: getattr(self, k) for k in self.__dataclass_fields__}
        row["lowtex"] = self.lowtex
        return row


@lru_cache(maxsize=4)
def _dct_matrix(n: int, keep: int) -> np.ndarray:
    """Rows ``0..keep-1`` of the orthonormal DCT-II matrix of size ``n``."""
    k = np.arange(keep)[:, None]
    x = np.arange(n)[None, :]
    mat = np.cos(np.pi * (2 * x + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    mat[0] /= np.sqrt(2.0)
    return mat


def _bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits.ravel().tolist():
        value = (value << 1) | int(bit)
    return value


def _threshold(coeffs: np.ndarray) -> np.ndarray:
    flat = coeffs.ravel()
    return coeffs > np.median(flat[1:])


def _dhash_bits(thumb: np.ndarray) -> int:
    return _bits_to_int(thumb[:, 1:] > thumb[:, :-1])


def _flip_signs(size: int) -> np.ndarray:
    return np.where(np.arange(size) % 2 == 1, -1.0, 1.0)


def features_from_image(img, sha256: str) -> ImageFeatures:  # type: ignore[no-untyped-def]
    from PIL import Image

    img.load()
    width, height = img.size
    rgb = img if img.mode == "RGB" else img.convert("RGB")
    pixel = hashlib.sha256(f"{width}x{height}:".encode())
    pixel.update(rgb.tobytes())
    gray = img.convert("L")  # v1 converts the decoded image straight to L
    thumb = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    small = np.asarray(
        gray.resize((_PHASH_SIDE, _PHASH_SIDE), Image.Resampling.LANCZOS), dtype=np.float64
    )
    dct = _dct_matrix(_PHASH_SIDE, _PHASH_BLOCK)
    coeffs = dct @ small @ dct.T
    col = _flip_signs(_PHASH_BLOCK)
    variants = {"": coeffs, "_h": coeffs * col[None, :], "_v": coeffs * col[:, None]}
    ph = {k: np.packbits(_threshold(c)).tobytes() for k, c in variants.items()}
    p64 = {k: _bits_to_int(_threshold(c[:_P64_BLOCK, :_P64_BLOCK])) for k, c in variants.items()}
    return ImageFeatures(
        sha256=sha256,
        pixel_sha256=pixel.hexdigest(),
        width=width,
        height=height,
        dhash=_dhash_bits(thumb),
        dhash_h=_dhash_bits(thumb[:, ::-1]),
        dhash_v=_dhash_bits(thumb[::-1, :]),
        phash=ph[""],
        phash_h=ph["_h"],
        phash_v=ph["_v"],
        phash64=p64[""],
        phash64_h=p64["_h"],
        phash64_v=p64["_v"],
        margin=float(np.abs(np.diff(thumb, axis=1)).mean()),
        gray_std=float(small.std()),
    )


def features_from_bytes(data: bytes, sha256: str | None = None) -> ImageFeatures:
    from PIL import Image

    digest = sha256 or hashlib.sha256(data).hexdigest()
    try:
        with Image.open(io.BytesIO(data)) as img:
            return features_from_image(img, digest)
    except Exception as exc:
        raise FeatureError(f"cannot decode image {digest[:12]}: {exc}") from exc


def features_from_path(path: str | Path) -> ImageFeatures:
    return features_from_bytes(Path(path).read_bytes())


def phash_distance(a: bytes, b: bytes) -> int:
    xa = np.frombuffer(a, dtype=np.uint8)
    xb = np.frombuffer(b, dtype=np.uint8)
    return int(np.bitwise_count(xa ^ xb).sum())


def phash_matrix_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Hamming distance of two ``(n, 32)`` uint8 arrays."""
    return np.bitwise_count(a ^ b).sum(axis=1).astype(np.int32)
