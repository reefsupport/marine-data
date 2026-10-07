"""DINOv2 feature cache over a release's images (WP-9, shared with eval P5).

Format and preprocessing follow design §4.4 (``eval-decontamination-split-v2.md``) so the
P5 frozen probe can reuse the cache unchanged:

* backbone ``facebook/dinov2-small`` (ViT-S/14), open weights, pinned by HF revision sha;
* input: RGB, resize the short side to 256 (bicubic, the long side scaled as the HF
  ``BitImageProcessor`` does), centre-crop 224, ``/255``, ImageNet mean/std;
* feature = the layer-normed CLS token ⊕ the mean of the layer-normed patch tokens
  (384 + 384 = 768-d), stored as float16;
* parquet keyed by ``(image_sha256, model, revision)``, one ``part-NNNNN.parquet`` per
  input shard (resumable: finished parts are skipped), plus ``FEATURES.json`` with the
  model revision, preprocessing, library versions, device and a per-part sha256.

torch/transformers are imported lazily; the preprocessing and the reader are pure numpy.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

MODEL_ID = "facebook/dinov2-small"
FEATURE_DIM = 768
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class Preprocess:
    resize_short: int = 256
    center_crop: int = 224
    interpolation: str = "bicubic"
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    feature: str = "cls_layernorm ⊕ mean(patch_layernorm), 384+384=768, float16"


PREPROCESS = Preprocess()


def resize_shape(width: int, height: int, short: int) -> tuple[int, int]:
    """(w, h) after scaling the short side to ``short`` — the HF ``shortest_edge`` rule."""
    if width <= height:
        return short, int(short * height / width)
    return int(short * width / height), short


def preprocess(image_bytes: bytes, spec: Preprocess = PREPROCESS) -> np.ndarray:
    """Decode and normalise one image to a ``(3, crop, crop)`` float32 array."""
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as im:
        rgb = im.convert("RGB")
    w, h = resize_shape(rgb.width, rgb.height, spec.resize_short)
    rgb = rgb.resize((w, h), Image.Resampling.BICUBIC)
    c = spec.center_crop
    left, top = round((w - c) / 2.0), round((h - c) / 2.0)
    arr = np.asarray(rgb.crop((left, top, left + c, top + c)), dtype=np.float32) / 255.0
    arr = (arr - np.asarray(spec.mean, np.float32)) / np.asarray(spec.std, np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


@dataclass
class FeatureRun:
    model: str
    revision: str
    device: str
    preprocess: dict[str, object]
    versions: dict[str, str]
    parts: dict[str, dict[str, object]] = field(default_factory=dict)
    count: int = 0
    seconds: float = 0.0


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_image_batches(
    shard: Path, batch_size: int, skip: set[str] | None = None
) -> Iterator[tuple[list[str], list[bytes]]]:
    """Yield ``(sha256s, image bytes)`` batches from one HF ``images`` parquet shard."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(shard)
    for rb in pf.iter_batches(batch_size=batch_size, columns=["image_sha256", "image"]):
        shas = rb.column(0).to_pylist()
        imgs = rb.column(1).field("bytes").to_pylist()
        keep = [(s, b) for s, b in zip(shas, imgs, strict=True) if not skip or s not in skip]
        if keep:
            yield [s for s, _ in keep], [b for _, b in keep]


def _load_model(model_id: str, revision: str | None, device: str):  # type: ignore[no-untyped-def]
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    path = Path(snapshot_download(model_id, revision=revision))
    model = AutoModel.from_pretrained(path).eval().to(device)
    torch.set_grad_enabled(False)
    return model, path.name  # snapshot dir name == the resolved commit sha


def embed(model, batch: np.ndarray, device: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    """CLS ⊕ mean patch token of the final (layer-normed) hidden state, float16."""
    import torch

    out = model(pixel_values=torch.from_numpy(batch).to(device)).last_hidden_state
    feat = torch.cat([out[:, 0], out[:, 1:].mean(dim=1)], dim=1)
    return feat.float().cpu().numpy().astype(np.float16)


def extract(
    shards: Sequence[Path],
    out_dir: Path,
    *,
    model_id: str = MODEL_ID,
    revision: str | None = None,
    device: str = "mps",
    batch_size: int = 64,
    workers: int = 8,
) -> FeatureRun:
    """Embed every image in ``shards``; one output part per shard, resumable."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import transformers

    out_dir.mkdir(parents=True, exist_ok=True)
    model, resolved = _load_model(model_id, revision, device)
    meta_path = out_dir / "FEATURES.json"
    run = FeatureRun(
        model=model_id,
        revision=resolved,
        device=device,
        preprocess=asdict(PREPROCESS),
        versions={
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pillow": __import__("PIL").__version__,
        },
    )
    if meta_path.exists():
        prior = json.loads(meta_path.read_text())
        if prior.get("revision") != resolved:
            raise ValueError(f"cache revision {prior.get('revision')} != model {resolved}")
        run.parts = prior.get("parts", {})
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, shard in enumerate(shards):
            part = out_dir / f"part-{i:05d}.parquet"
            if part.name in run.parts and part.exists():
                continue
            shas: list[str] = []
            feats: list[np.ndarray] = []
            for batch_shas, blobs in iter_image_batches(shard, batch_size):
                arr = np.stack(list(pool.map(preprocess, blobs)))
                feats.append(embed(model, arr, device))
                shas.extend(batch_shas)
            vecs = np.concatenate(feats) if feats else np.zeros((0, FEATURE_DIM), np.float16)
            table = pa.table(
                {
                    "image_sha256": pa.array(shas, pa.string()),
                    "model": pa.array([model_id] * len(shas), pa.string()),
                    "revision": pa.array([resolved] * len(shas), pa.string()),
                    "feature": pa.FixedSizeListArray.from_arrays(
                        pa.array(vecs.reshape(-1), pa.float16()), FEATURE_DIM
                    ),
                }
            )
            pq.write_table(table, part)
            run.parts[part.name] = {
                "input": shard.name,
                "rows": len(shas),
                "sha256": _file_sha256(part),
            }
            run.count = sum(int(p["rows"]) for p in run.parts.values())  # type: ignore[call-overload]
            run.seconds = round(run.seconds + time.time() - t0, 1)
            t0 = time.time()
            meta_path.write_text(json.dumps(asdict(run), indent=1, sort_keys=True))
            print(f"{part.name} rows={len(shas)} total={run.count} t={run.seconds}s", flush=True)
    return run


def load_features(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    """Read a feature cache → (sha256 list, ``(n, 768)`` float32 matrix), sha-sorted."""
    import pyarrow.parquet as pq

    shas: list[str] = []
    mats: list[np.ndarray] = []
    for part in sorted(Path(cache_dir).glob("part-*.parquet")):
        t = pq.read_table(part, columns=["image_sha256", "feature"])
        shas.extend(t.column(0).to_pylist())
        flat = t.column(1).combine_chunks().flatten().to_numpy(zero_copy_only=False)
        mats.append(flat.reshape(-1, FEATURE_DIM).astype(np.float32))
    mat = np.concatenate(mats) if mats else np.zeros((0, FEATURE_DIM), np.float32)
    order = np.argsort(np.asarray(shas))
    return [shas[i] for i in order], mat[order]
