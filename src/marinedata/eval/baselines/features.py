"""Frozen backbone feature extraction (design §4.4.A).

Two permissive, anonymously-downloadable backbones (D-AA — never a token, never a
login):

- ``dinov2-small``  — ``facebook/dinov2-small`` (ViT-S/14, Apache-2.0), via
  ``transformers``.
- ``openclip-vitb16`` — ``laion/CLIP-ViT-B-16-laion2B-s34B-b88K`` (MIT), via
  ``open_clip``'s HF-hub loader.

Both are pinned by revision sha (no ``main`` floating ref). Input: resize the short
side to 256, centre-crop 224, ImageNet mean/std (design §4.4.A). Output feature is the
CLS token concatenated with the mean-pooled patch tokens, cached as float16 keyed by
``(image_sha256, model, revision)`` so a re-run never re-extracts an already-cached
image.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Backbone:
    """One pinned, anonymously-fetchable backbone."""

    id: str
    kind: str  # "hf_transformers" | "open_clip"
    hf_id: str
    revision: str
    feature_dim: int
    licence: str


BACKBONES: dict[str, Backbone] = {
    "dinov2-small": Backbone(
        id="dinov2-small",
        kind="hf_transformers",
        hf_id="facebook/dinov2-small",
        revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        feature_dim=768,  # 384 CLS + 384 mean-patch
        licence="Apache-2.0",
    ),
    "openclip-vitb16": Backbone(
        id="openclip-vitb16",
        kind="open_clip",
        hf_id="laion/CLIP-ViT-B-16-laion2B-s34B-b88K",
        revision="7288da5a0d6f0b51c4a2b27c624837a9236d0112",
        feature_dim=1536,  # 768 CLS + 768 mean-patch
        licence="MIT",
    ),
}


def _preprocess(image: Image.Image) -> np.ndarray:
    """Resize short side to 256, centre-crop 224, ImageNet-normalise, CHW float32."""
    image = image.convert("RGB")
    w, h = image.size
    scale = 256 / min(w, h)
    image = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)
    w, h = image.size
    left, top = (w - 224) // 2, (h - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
    return arr.transpose(2, 0, 1)  # CHW


def image_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FeatureExtractor:
    """Loads one pinned backbone once; extracts CLS ⊕ mean-patch features.

    ``device`` defaults to MPS when available (design: Mac does feature extraction),
    falling back to CPU so tests and the server Job still work.
    """

    def __init__(self, backbone_id: str, *, device: str | None = None) -> None:
        import torch

        self.backbone = BACKBONES[backbone_id]
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._torch = torch
        self._model = self._load_model()

    def _load_model(self):
        bb = self.backbone
        if bb.kind == "hf_transformers":
            from transformers import AutoModel

            model = AutoModel.from_pretrained(bb.hf_id, revision=bb.revision)
        elif bb.kind == "open_clip":
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                f"hf-hub:{bb.hf_id}", revision=bb.revision
            )
            model = model.visual
        else:  # pragma: no cover - guarded by BACKBONES literal set
            raise ValueError(f"unknown backbone kind {bb.kind!r}")
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    def _forward_batch(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        x = torch.from_numpy(batch).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            if self.backbone.kind == "hf_transformers":
                out = self._model(pixel_values=x)
                cls = out.last_hidden_state[:, 0, :]
                patches = out.last_hidden_state[:, 1:, :].mean(dim=1)
            else:  # open_clip visual tower — forward_features-style tap
                tokens = self._model.forward_intermediates(x)["image_intermediates"][-1]
                if tokens.dim() == 4:  # B, C, H, W -> B, N, C
                    tokens = tokens.flatten(2).transpose(1, 2)
                cls = tokens[:, 0, :] if tokens.shape[1] > 1 else tokens.mean(dim=1)
                patches = tokens[:, 1:, :].mean(dim=1) if tokens.shape[1] > 1 else cls
            feat = torch.cat([cls, patches], dim=-1)
        return feat.to("cpu").numpy().astype(np.float16)

    def extract(
        self, images: Sequence[tuple[str, bytes]], *, batch_size: int = 32
    ) -> tuple[pd.DataFrame, float]:
        """``images``: (image_sha256, raw_bytes) pairs. Returns (features_df, img_per_s).

        ``features_df`` columns: image_sha256, model, revision, feature (float16 list).
        """
        shas: list[str] = []
        arrays: list[np.ndarray] = []
        for sha, raw in images:
            arrays.append(_preprocess(Image.open(__import__("io").BytesIO(raw))))
            shas.append(sha)

        rows: list[dict] = []
        t0 = time.perf_counter()
        for start in range(0, len(arrays), batch_size):
            batch = np.stack(arrays[start : start + batch_size])
            feats = self._forward_batch(batch)
            for sha, feat in zip(shas[start : start + batch_size], feats, strict=True):
                rows.append(
                    {
                        "image_sha256": sha,
                        "model": self.backbone.id,
                        "revision": self.backbone.revision,
                        "feature": feat.tolist(),
                    }
                )
        elapsed = time.perf_counter() - t0
        img_per_s = len(arrays) / elapsed if elapsed > 0 else float("inf")
        return pd.DataFrame(rows), img_per_s


def cache_path(cache_dir: Path, backbone_id: str, revision: str) -> Path:
    return cache_dir / f"features-{backbone_id}-{revision[:12]}.parquet"


def extract_with_cache(
    extractor: FeatureExtractor,
    images: Iterable[tuple[str, bytes]],
    cache_dir: Path,
    *,
    batch_size: int = 32,
) -> tuple[pd.DataFrame, float]:
    """Extract only images missing from the on-disk cache; merge and rewrite it.

    Returns (features for the requested shas, img/s measured on the newly-extracted
    ones only — a full cache hit reports ``inf``).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, extractor.backbone.id, extractor.backbone.revision)
    cached = (
        pd.read_parquet(path)
        if path.exists()
        else pd.DataFrame(columns=["image_sha256", "model", "revision", "feature"])
    )
    cached_shas = set(cached["image_sha256"])

    images = list(images)
    todo = [(sha, raw) for sha, raw in images if sha not in cached_shas]
    if todo:
        new_df, img_per_s = extractor.extract(todo, batch_size=batch_size)
        combined = pd.concat([cached, new_df], ignore_index=True)
        combined.to_parquet(path, index=False)
    else:
        new_df, img_per_s = pd.DataFrame(columns=cached.columns), float("inf")
        combined = cached

    wanted_shas = {sha for sha, _ in images}
    result = combined[combined["image_sha256"].isin(wanted_shas)].drop_duplicates(
        "image_sha256", keep="last"
    )
    return result, img_per_s
