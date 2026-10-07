"""Dedup v2 (WP-10): exact + perceptual + SSCD-confirmed duplicates, split groups, leak gate.

Modules: :mod:`.features` (per-image hashes), :mod:`.mih` (multi-index Hamming search),
:mod:`.embed` (SSCD + patch -> parent matcher), :mod:`.confirm` (candidates + rules),
:mod:`.groups` (clusters, split groups, gate), :mod:`.corpus` (runner),
:mod:`.evaluate` / :mod:`.audit` (acceptance evidence).
"""

from .groups import DedupGateError, GateResult, gate_release, run_gate

__all__ = ["DedupGateError", "GateResult", "gate_release", "run_gate"]
