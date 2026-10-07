"""Small fine-tune, same backbone as the probe (design §4.4.B).

Last 4 transformer blocks + a linear head are trainable; everything else stays frozen.
AdamW, lr 5e-5 (backbone) / 1e-3 (head), weight decay 0.05, cosine schedule with a
1-epoch warm-up. fp32 on MPS (bf16 is CUDA-only per the design; MPS bf16 support is
still partial). P5 runs only the smallest task, 1 epoch, on the 2k fixture — the
manager decision that keeps this package inside its compute budget; the full 10-epoch
/ 3-seed protocol is the v2 build's job, not P5's.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FinetuneResult:
    train_loss: list[float]
    val_macro_f1: float
    epochs_run: int
    seed: int


def _unfreeze_last_blocks(model, n_blocks: int) -> list:
    """Best-effort: unfreeze the last ``n_blocks`` transformer blocks of a HF ViT-style
    backbone (``model.encoder.layer`` for DINOv2/ViT) plus every top-level parameter
    that is not part of ``encoder.layer`` (embeddings stay frozen)."""
    trainable = []
    layers = getattr(getattr(model, "encoder", None), "layer", None)
    if layers is not None:
        for block in list(layers)[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad_(True)
                trainable.append(p)
    return trainable


def finetune_head(
    model,
    head,
    x_train_images,
    y_train,
    x_val_images,
    y_val,
    *,
    device: str,
    epochs: int = 1,
    seed: int = 0,
    n_unfrozen_blocks: int = 4,
    batch_size: int = 32,
) -> FinetuneResult:
    """Train ``head`` (a small ``nn.Module``) plus the last ``n_unfrozen_blocks`` of
    ``model``. ``x_train_images``/``x_val_images`` are already-preprocessed float32
    CHW tensors (``np.ndarray``, N×3×224×224); labels are integer class ids.
    """
    import torch
    from torch import nn, optim

    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad_(False)
    backbone_params = _unfreeze_last_blocks(model, n_unfrozen_blocks)
    head_params = list(head.parameters())

    opt = optim.AdamW(
        [
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 1e-3},
        ],
        weight_decay=0.05,
    )
    steps_per_epoch = max(1, len(y_train) // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch  # 1 warm-up epoch
    sched = optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: _cosine_warmup(s, warmup_steps, total_steps)
    )
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    head.to(device)
    x_train = torch.from_numpy(np.asarray(x_train_images, dtype=np.float32))
    y_train_t = torch.as_tensor(np.asarray(y_train), dtype=torch.long)

    losses: list[float] = []
    rng = np.random.default_rng(seed)
    n = len(y_train_t)
    for _epoch in range(epochs):
        order = rng.permutation(n)
        epoch_losses = []
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            xb = x_train[idx].to(device)
            yb = y_train_t[idx].to(device)
            out = model(pixel_values=xb)
            cls = out.last_hidden_state[:, 0, :]
            patches = out.last_hidden_state[:, 1:, :].mean(dim=1)
            feat = torch.cat([cls, patches], dim=-1)
            logits = head(feat)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    val_f1 = _eval_macro_f1(model, head, x_val_images, y_val, device=device, batch_size=batch_size)
    return FinetuneResult(train_loss=losses, val_macro_f1=val_f1, epochs_run=epochs, seed=seed)


def _cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    import math

    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _eval_macro_f1(model, head, x_images, y, *, device: str, batch_size: int) -> float:
    import torch
    from sklearn.metrics import f1_score

    model.eval()
    head.eval()
    preds = []
    x = torch.from_numpy(np.asarray(x_images, dtype=np.float32))
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start : start + batch_size].to(device)
            out = model(pixel_values=xb)
            cls = out.last_hidden_state[:, 0, :]
            patches = out.last_hidden_state[:, 1:, :].mean(dim=1)
            feat = torch.cat([cls, patches], dim=-1)
            preds.append(head(feat).argmax(dim=-1).cpu().numpy())
    y_pred = np.concatenate(preds) if preds else np.array([])
    labels = sorted(set(np.asarray(y).tolist()))
    return float(f1_score(y, y_pred, average="macro", labels=labels, zero_division=0))
