"""The 500-sample expert-audit sheet (WP-9, D-I: filled by Yohan's team later).

Strata are ``source × given health label × CL flag``. Allocation: half the budget is
spread equally across strata (so small strata and the flags are well covered), half is
proportional to stratum size; every row carries its inclusion weight ``N_h / n_h`` so
population noise is estimated with a weighted (Horvitz-Thompson) mean, not a raw one.
"""

from __future__ import annotations

import numpy as np

EXPERT_COLUMNS = (
    "expert1_condition",
    "expert1_confidence",
    "expert2_condition",
    "expert2_confidence",
    "adjudicated_condition",
    "adjudicator",
    "notes",
)
CONDITIONS = ("HEALTHY", "PALE", "BLEACHED", "OTHER_UNHEALTHY", "DEAD", "NOT_CORAL", "UNSURE")


def allocate(sizes: dict[tuple[str, ...], int], n: int) -> dict[tuple[str, ...], int]:
    """Half equal, half proportional, capped at the stratum size, topped up to ``n``."""
    keys = sorted(sizes)
    total = sum(sizes.values())
    alloc = {
        k: min(sizes[k], int(n / 2 / len(keys)) + round(n / 2 * sizes[k] / total)) for k in keys
    }
    while sum(alloc.values()) < min(n, total):
        room = [k for k in keys if alloc[k] < sizes[k]]
        best = max(room, key=lambda k: (sizes[k] - alloc[k], k))
        alloc[best] += 1
    while sum(alloc.values()) > n:
        best = max(keys, key=lambda k: (alloc[k], k))
        alloc[best] -= 1
    return alloc


def expert_sheet(labels, flagged_keys: set[tuple[str, str]], n: int = 500, seed: int = 20260925):  # type: ignore[no-untyped-def]
    """Sample ``n`` rows of ``labels`` (image_sha256, source_id, label, split, ...).

    ``flagged_keys`` holds ``(image_sha256, source_id)`` flagged by confident learning.
    Returns the sheet with ``stratum``, ``weight`` and blank expert columns, shuffled so
    the order leaks neither the stratum nor the flag.
    """
    rng = np.random.default_rng(seed)
    df = labels.drop_duplicates(["image_sha256", "source_id"]).reset_index(drop=True)
    df["cl_flag"] = [
        (s, src) in flagged_keys for s, src in zip(df.image_sha256, df.source_id, strict=True)
    ]
    df["stratum"] = df.source_id + "|" + df.label + "|" + np.where(df.cl_flag, "flag", "noflag")
    strata = df.groupby("stratum").indices
    alloc = allocate({(k,): len(v) for k, v in strata.items()}, n)
    picks, weights = [], []
    for (k,), m in sorted(alloc.items()):
        idx = strata[k]
        chosen = rng.choice(idx, size=m, replace=False)
        picks.extend(chosen.tolist())
        weights.extend([len(idx) / m] * m)
    sheet = df.iloc[picks].copy()
    sheet["weight"] = weights
    sheet = sheet.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sheet.insert(0, "audit_id", [f"A{i:03d}" for i in range(1, len(sheet) + 1)])
    for col in EXPERT_COLUMNS:
        sheet[col] = ""
    return sheet
