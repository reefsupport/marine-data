"""Run release gates by name. A gate that raises is a failed gate: never a crash, never a pass."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import GateResult, finish

ALL = ("licence", "dedup", "splits", "masks", "privacy", "load", "card", "croissant")


@dataclass(frozen=True)
class GateOptions:
    flavour: str | None = None
    release_dir: Path | None = None
    phash_bits: int = 8
    split_tolerance: float = 5.0
    empty_threshold: float = 0.05
    detector: str | None = None
    privacy_threshold: float = 0.85
    sample_rows: int = 50
    v1_dir: Path | None = None


def _licence(b: Path, o: GateOptions) -> GateResult:
    from .licence import run_licence_gate

    return run_licence_gate(b, flavour=o.flavour, release_dir=o.release_dir)


def _dedup(b: Path, o: GateOptions) -> GateResult:
    from ..licence_class import source_families
    from .dedup import run_dedup_gate

    return run_dedup_gate(
        b, release_dir=o.release_dir, phash_bits=o.phash_bits, families=source_families()
    )


def _splits(b: Path, o: GateOptions) -> GateResult:
    from .splits import run_splits_gate

    return run_splits_gate(
        b, release_dir=o.release_dir, phash_bits=o.phash_bits, tolerance=o.split_tolerance
    )


def _masks(b: Path, o: GateOptions) -> GateResult:
    from .masks import run_masks_gate

    return run_masks_gate(b, empty_threshold=o.empty_threshold, v1_dir=o.v1_dir)


def _privacy(b: Path, o: GateOptions) -> GateResult:
    from .privacy import run_privacy_gate
    from .privacy_detectors import load_detector

    return run_privacy_gate(b, detector=load_detector(o.detector), threshold=o.privacy_threshold)


def _load(b: Path, o: GateOptions) -> GateResult:
    from .load import run_load_gate

    return run_load_gate(b, release_dir=o.release_dir, sample_rows=o.sample_rows, v1_dir=o.v1_dir)


def _card(b: Path, o: GateOptions) -> GateResult:
    from .card import run_card_gate

    return run_card_gate(b, release_dir=o.release_dir, v1_dir=o.v1_dir)


def _croissant(b: Path, o: GateOptions) -> GateResult:
    from .croissant import run_croissant_gate

    return run_croissant_gate(b)


RUNNERS: dict[str, Callable[[Path, GateOptions], GateResult]] = {
    "licence": _licence,
    "dedup": _dedup,
    "splits": _splits,
    "masks": _masks,
    "privacy": _privacy,
    "load": _load,
    "card": _card,
    "croissant": _croissant,
}


def run_gate(name: str, build_dir: Path, options: GateOptions | None = None) -> GateResult:
    try:
        return RUNNERS[name](Path(build_dir), options or GateOptions())
    except Exception as exc:
        return finish(
            name,
            {"failed_gate_error": 1},
            [f"{name} gate_error: {type(exc).__name__}: {str(exc)[:200]}"],
        )
