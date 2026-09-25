"""Copy-detection embeddings (SSCD) and the patch -> parent matcher.

SSCD (Pizzi et al., CVPR 2022, ``sscd_disc_mixup``, MIT licence) is a ResNet-50 trained
for image copy detection under crop, resize, re-encode, colour and flip augmentations;
it outputs a 512-d L2-normalised descriptor. Weights are a TorchScript file fetched once
over plain HTTPS (no login) — ``torch`` is an optional dependency imported lazily.

Embeddings are cached by the image ``sha256`` (+ variant tag) in a SQLite file, so a
re-run only embeds new images.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
SSCD_SHA256 = "9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56"
EMBED_DIM = 512
EMBED_SIDE = 288
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def prepare(img, side: int = EMBED_SIDE) -> np.ndarray:  # type: ignore[no-untyped-def]
    """PIL image -> normalised ``(3, side, side)`` float32 (square resize, no crop)."""
    from PIL import Image

    if getattr(img, "format", None) == "JPEG":
        img.draft("RGB", (side * 2, side * 2))
    rgb = img.convert("RGB").resize((side, side), Image.Resampling.BILINEAR)
    arr = (np.asarray(rgb, dtype=np.float32) / 255.0 - _MEAN) / _STD
    return arr.transpose(2, 0, 1)


class EmbeddingCache:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path))
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for lo in range(0, len(keys), 900):
            part = list(keys[lo : lo + 900])
            marks = ",".join("?" * len(part))
            for key, blob in self._db.execute(
                f"SELECT key, vec FROM emb WHERE key IN ({marks})", part
            ):
                out[key] = np.frombuffer(blob, dtype=np.float16)
        return out

    def put_many(self, items: Iterable[tuple[str, np.ndarray]]) -> None:
        self._db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?)",
            [(k, v.astype(np.float16).tobytes()) for k, v in items],
        )
        self._db.commit()


class SSCDEmbedder:
    def __init__(self, weights: str | Path, device: str | None = None, batch: int = 64) -> None:
        import torch

        self._torch = torch
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.batch = batch
        self.model = torch.jit.load(str(weights), map_location="cpu").eval().to(device)

    def embed_arrays(self, arrays: Sequence[np.ndarray]) -> np.ndarray:
        torch = self._torch
        out = []
        for lo in range(0, len(arrays), self.batch):
            x = torch.from_numpy(np.stack(arrays[lo : lo + self.batch])).to(self.device)
            with torch.no_grad():
                y = self.model(x)
            y = torch.nn.functional.normalize(y.float(), dim=1)
            out.append(y.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, EMBED_DIM), np.float32)

    def embed_keys(
        self,
        keys: Sequence[str],
        load: Callable[[str], np.ndarray],
        cache: EmbeddingCache | None = None,
        threads: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Embeddings for ``keys`` in order; ``load(key)`` returns a :func:`prepare` array."""
        have = cache.get_many(keys) if cache is not None else {}
        todo = [k for k in dict.fromkeys(keys) if k not in have]
        step = self.batch * 8
        with ThreadPoolExecutor(threads) as pool:
            for lo in range(0, len(todo), step):
                part = todo[lo : lo + step]
                vecs = self.embed_arrays(list(pool.map(load, part)))
                fresh = dict(zip(part, vecs, strict=True))
                have.update({k: v.astype(np.float16) for k, v in fresh.items()})
                if cache is not None:
                    cache.put_many(fresh.items())
                if progress is not None:
                    progress(lo + len(part), len(todo))
        return (
            np.stack([have[k].astype(np.float32) for k in keys])
            if keys
            else np.zeros((0, EMBED_DIM), np.float32)
        )


def knn_pairs(
    emb: np.ndarray, k: int, min_cos: float, device: str = "cpu", block: int = 4096
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact cosine kNN self-join in blocks: ``(i, j, cos)`` with ``i < j``, ``cos >= min_cos``.

    O(N^2 d) flops but O(block * N) memory; at 5M images swap in an IVF-PQ index.
    """
    import torch

    base = torch.from_numpy(emb.astype(np.float16 if device != "cpu" else np.float32)).to(device)
    ii, jj, cc = [], [], []
    for lo in range(0, len(emb), block):
        sims = base[lo : lo + block] @ base.T
        vals, idx = torch.topk(sims.float(), min(k + 1, len(emb)), dim=1)
        vals, idx = vals.cpu().numpy(), idx.cpu().numpy()
        rows = np.arange(lo, lo + len(vals))[:, None].repeat(idx.shape[1], axis=1)
        keep = (vals >= min_cos) & (idx != rows)
        a, b = rows[keep], idx[keep]
        ii.append(np.minimum(a, b))
        jj.append(np.maximum(a, b))
        cc.append(vals[keep])
    i, j, c = (np.concatenate(x) for x in (ii, jj, cc))
    _, first = np.unique(i * np.int64(len(emb)) + j, return_index=True)
    return i[first], j[first], c[first]


@dataclass(frozen=True)
class PatchMatch:
    score: float  # normalised cross-correlation, -1..1
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in parent pixels
    scale: float  # patch width / parent width


def _ncc_map(parent: np.ndarray, patch: np.ndarray) -> np.ndarray:
    ph, pw = patch.shape
    t = patch - patch.mean()
    tnorm = np.sqrt((t * t).sum()) or 1.0
    fshape = (parent.shape[0] + ph, parent.shape[1] + pw)
    corr = np.fft.irfft2(np.fft.rfft2(parent, fshape) * np.conj(np.fft.rfft2(t, fshape)), fshape)
    corr = corr[: parent.shape[0] - ph + 1, : parent.shape[1] - pw + 1]
    s1 = np.pad(parent, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    s2 = np.pad(parent * parent, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def box(s):  # type: ignore[no-untyped-def]
        return s[ph:, pw:] - s[:-ph, pw:] - s[ph:, :-pw] + s[:-ph, :-pw]

    n = ph * pw
    var = np.maximum(box(s2) - box(s1) ** 2 / n, 1e-6)
    return corr / (np.sqrt(var) * tnorm)


def match_patch(
    patch_img, parent_img, scales: Sequence[float] | None = None, side: int = 256
) -> PatchMatch:  # type: ignore[no-untyped-def]
    """Locate ``patch_img`` inside ``parent_img`` by multi-scale NCC on grayscale.

    ``scales`` are candidate patch-width / parent-width ratios; the default grid spans
    NOAA 224-px point crops (~0.1) to near-full-frame crops (0.95). The patch keeps its
    own aspect ratio; resampling is to a ``side``-px wide parent.
    """
    from PIL import Image

    scales = scales or [round(0.08 * 1.12**i, 4) for i in range(24) if 0.08 * 1.12**i <= 1.0]
    parent = parent_img.convert("L")
    pw0, ph0 = parent.size
    factor = side / pw0
    par = np.asarray(
        parent.resize((side, max(8, round(ph0 * factor))), Image.Resampling.BILINEAR), np.float64
    )
    patch = patch_img.convert("L")
    aspect = patch.size[1] / patch.size[0]
    best = PatchMatch(-1.0, (0, 0, 0, 0), 0.0)
    for s in scales:
        w = round(s * side)
        h = round(w * aspect)
        if w < 8 or h < 8 or w > par.shape[1] or h > par.shape[0]:
            continue
        tpl = np.asarray(patch.resize((w, h), Image.Resampling.BILINEAR), np.float64)
        ncc = _ncc_map(par, tpl)
        y, x = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
        score = float(ncc[y, x])
        if score > best.score:
            box = (int(x / factor), int(y / factor), int((x + w) / factor), int((y + h) / factor))
            best = PatchMatch(score, box, s)
    return best
