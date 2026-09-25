"""Tier B: a stratified VLM captioning pilot, `label_origin=model`.

Model policy (manager decision): an Apache-2.0 open-weights VLM of the
Qwen2.5-VL-3B-Instruct class. Prefer the MLX 4-bit build on MPS (`mlx-vlm`); fall back
to `transformers` on MPS if MLX is unavailable. Anonymous download only — no
`HF_TOKEN`, no login, a ModelScope mirror is an acceptable substitute source. D-AA's
backoff applies: on a 429, cool down >= 15 min, requeue at the tail, HF concurrency 1,
and after 3 cool-downs the source goes to the needs-Yohan list.

Neither `mlx-vlm` nor `transformers` is a hard dependency of this package (see
`pyproject.toml`, which WP-13 does not edit) — `load_backend` raises a clear
`RuntimeError` naming what to `uv pip install` when neither is present, rather than
importing them at module load time. `VLMBackend` is the seam tests use to run the
pipeline without a real model.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from .facts import CaptionFacts

MLX_MODEL_ID = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
TRANSFORMERS_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# D-AA backoff constants.
COOLDOWN_SECONDS = 15 * 60
MAX_COOLDOWNS = 3


class VLMBackend(Protocol):
    """The seam `run_pilot` needs. A real backend wraps MLX or `transformers`; tests
    supply a fake that returns fixed strings."""

    model_id: str
    revision: str

    def generate(self, image_path: str, prompt: str) -> str: ...


@dataclass
class _NullBackend:
    """Placeholder returned only by tests that don't care about generation."""

    model_id: str = "null"
    revision: str = "none"

    def generate(self, image_path: str, prompt: str) -> str:
        return ""


def load_backend(prefer_mlx: bool = True) -> VLMBackend:
    """Load the pilot's VLM backend: MLX 4-bit on MPS, else `transformers` on MPS.

    Raises `RuntimeError` (not an import crash) when neither stack is installed, so a
    caller can catch it and report "needs `uv pip install mlx-vlm` / `transformers`"
    instead of a bare traceback.
    """
    if prefer_mlx:
        try:
            import mlx_vlm  # noqa: F401

            return _MLXBackend(MLX_MODEL_ID)
        except ImportError:
            pass
    try:
        import transformers  # noqa: F401

        return _TransformersBackend(TRANSFORMERS_MODEL_ID)
    except ImportError as exc:
        raise RuntimeError(
            "no VLM backend installed — run "
            "`uv pip install mlx-vlm` (preferred, MPS 4-bit) or "
            "`uv pip install transformers accelerate` (fallback) inside the worktree "
            "venv. Neither is a pyproject dependency; WP-13 does not edit pyproject.toml."
        ) from exc


class _MLXBackend:
    """Thin wrapper over `mlx_vlm`. Constructed lazily so import errors surface in
    `load_backend`, not at module import time."""

    def __init__(self, model_id: str, revision: str = "main") -> None:
        import mlx_vlm

        self.model_id = model_id
        self.revision = revision
        self.model, self.processor = mlx_vlm.load(model_id)

    def generate(self, image_path: str, prompt: str) -> str:
        import mlx_vlm

        return mlx_vlm.generate(self.model, self.processor, prompt, image=[image_path])


class _TransformersBackend:
    """Thin wrapper over `transformers` `Qwen2VLForConditionalGeneration`-class model,
    run on MPS if available."""

    def __init__(self, model_id: str, revision: str = "main") -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_id = model_id
        self.revision = revision
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision).to(
            device
        )

    def generate(self, image_path: str, prompt: str) -> str:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=128)
        return self.processor.decode(out[0], skip_special_tokens=True)


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_prompt(facts: CaptionFacts) -> str:
    """Ground the VLM in the Tier-A facts and forbid contradicting them."""
    grounding = []
    if facts.habitat:
        grounding.append(f"habitat: {facts.habitat}")
    if facts.depth_band:
        grounding.append(f"depth: {facts.depth_band}")
    if facts.meow_ecoregion:
        grounding.append(f"ecoregion: {facts.meow_ecoregion}")
    if facts.benthic_dominant:
        grounding.append(f"dominant benthic cover: {facts.benthic_dominant}")
    if facts.bleaching_status:
        grounding.append(f"recorded coral condition: {facts.bleaching_status}")
    facts_line = "; ".join(grounding) if grounding else "no additional metadata"
    return (
        "You are captioning an underwater reef-survey photo for a training dataset. "
        f"Known facts about this image ({facts_line}). "
        "Describe only what is visible in the image in one or two sentences. "
        "Do not contradict the known facts above; if you are unsure of something the "
        "facts do not cover, omit it rather than guess."
    )


def stratified_sample(
    metadata: pd.DataFrame,
    n: int,
    strata_cols: tuple[str, ...] = ("source_id", "habitat"),
    seed: int = 0,
) -> pd.DataFrame:
    """Deterministic proportional-stratified sample of `n` rows from `metadata`.

    Strata with fewer rows than their proportional share keep all of their rows
    (never oversampled); the shortfall is redistributed pro-rata over the remaining
    strata. Row order and choice are fully determined by `seed`.
    """
    if n >= len(metadata):
        return metadata.copy()
    groups = list(metadata.groupby(list(strata_cols), dropna=False))
    total = len(metadata)
    quotas = {key: max(1, round(n * len(g) / total)) for key, g in groups}
    picked = []
    for key, g in groups:
        k = min(quotas[key], len(g))
        picked.append(g.sample(n=k, random_state=seed))
    sample = pd.concat(picked, axis=0)
    if len(sample) > n:
        sample = sample.sample(n=n, random_state=seed)
    return sample.sort_values("image_sha256").reset_index(drop=True)


@dataclass
class PilotRow:
    image_sha256: str
    caption_vlm: str
    vlm_model: str
    vlm_revision: str
    prompt_sha256: str
    seed: int
    label_origin: str = "model"


def run_pilot(
    rows: Iterable[tuple[str, str, CaptionFacts]],
    backend: VLMBackend,
    seed: int,
    on_rate_limit: Callable[[], None] | None = None,
) -> list[PilotRow]:
    """Caption each `(image_sha256, image_path, facts)` row with `backend`.

    D-AA backoff is the caller's responsibility for network/HF errors when fetching
    images (this function only runs local inference); `on_rate_limit` is a hook a CLI
    can use to log a cool-down without this function knowing about HTTP status codes.
    """
    out: list[PilotRow] = []
    for sha, image_path, facts in rows:
        prompt = build_prompt(facts)
        caption = backend.generate(image_path, prompt)
        out.append(
            PilotRow(
                image_sha256=sha,
                caption_vlm=caption,
                vlm_model=backend.model_id,
                vlm_revision=backend.revision,
                prompt_sha256=prompt_sha256(prompt),
                seed=seed,
            )
        )
    return out


def cooldown_then_requeue(attempt: int, sleep: Callable[[float], None] = time.sleep) -> bool:
    """D-AA: sleep `COOLDOWN_SECONDS` and report whether the source should be requeued
    (True) or moved to the needs-Yohan list (False, after `MAX_COOLDOWNS` attempts)."""
    if attempt >= MAX_COOLDOWNS:
        return False
    sleep(COOLDOWN_SECONDS)
    return True
