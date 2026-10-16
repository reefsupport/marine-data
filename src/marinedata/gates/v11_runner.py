"""Orchestrator for the v1.1.0 release gates (``marinedata release gate --v11``).

Runs G1 join, G2 frozen, G3 leak, G4 bytes_once, G5 masks, G6 splits_v11, G7 decon, G8 privacy
and G9 labelmap on a build directory. A gate that raises is a failed gate, never a crash and never
a pass. ``rehearsal`` reaches only the gates that define it (``splits_v11``, ``decon``): it turns
an unsplit-only or input-missing failure into ``skipped``, which ``publish_verdict`` still refuses.
The v1.0 gates (licence, dedup, splits, load, card, croissant) stay with ``release gate --all``."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from . import GateResult, finish

V11_ALL = (
    "join",
    "frozen",
    "leak",
    "bytes_once",
    "masks",
    "splits_v11",
    "decon",
    "privacy",
    "labelmap",
)
"""G1 .. G9, in the order they run."""


@dataclass(frozen=True)
class V11Options:
    flavour: str = "open"
    release_dir: Path | None = None
    v1_dir: Path | None = None
    hidden_test: tuple[Path, ...] = ()
    benchmarks: tuple[Path, ...] = ()
    rehearsal: bool = False
    extra_nc_ids: frozenset[str] = frozenset()
    empty_threshold: float = 0.05
    detector: str | None = None
    privacy_threshold: float = 0.85
    taxonomy_version: str | None = None


@cache
def nc_source_ids(registry_root: Path | None = None) -> frozenset[str]:
    """Every registry source that does not ship in the open flavour."""
    from ..flavours import ships_in
    from ..registry import Registry

    registry = Registry.load(registry_root)
    return frozenset(s.id for s in registry.sources if not ships_in(registry, s.id, "open"))


def _nc(o: V11Options) -> frozenset[str]:
    return nc_source_ids() | o.extra_nc_ids


def _v1_sets(o: V11Options) -> tuple[frozenset[str] | None, frozenset[str]]:
    if o.v1_dir is None:
        return None, frozenset()
    from ..hf_export_v11 import v1_sha256s

    data = Path(o.v1_dir) / "data"
    configs = (
        frozenset(p.name for p in data.iterdir() if p.is_dir()) if data.is_dir() else frozenset()
    )
    return v1_sha256s(o.v1_dir), configs


def _join(b: Path, o: V11Options) -> GateResult:
    from .join import run_join_gate

    shas, configs = _v1_sets(o)
    return run_join_gate(b, flavour=o.flavour, nc_ids=_nc(o), v1_sha256=shas, v1_configs=configs)


def _frozen(b: Path, o: V11Options) -> GateResult:
    from .frozen import run_frozen_gate

    return run_frozen_gate(b, v1_dir=o.v1_dir)


def _leak(b: Path, o: V11Options) -> GateResult:
    from .leak import run_leak_gate

    return run_leak_gate(b, nc_ids=_nc(o), flavour=o.flavour)


def _bytes_once(b: Path, o: V11Options) -> GateResult:
    from .bytes_once import run_bytes_once_gate

    return run_bytes_once_gate(b, v1_dir=o.v1_dir)


def _masks(b: Path, o: V11Options) -> GateResult:
    from .masks import run_masks_gate

    return run_masks_gate(b, empty_threshold=o.empty_threshold, v1_dir=o.v1_dir)


def _splits(b: Path, o: V11Options) -> GateResult:
    from .splits_v11 import run_splits_v11_gate

    return run_splits_v11_gate(b, release_dir=o.release_dir, v1_dir=o.v1_dir, rehearsal=o.rehearsal)


def _decon(b: Path, o: V11Options) -> GateResult:
    from .decon_gate import run_decon_gate

    return run_decon_gate(
        b, hidden_test=o.hidden_test, benchmarks=o.benchmarks, rehearsal=o.rehearsal
    )


def _privacy(b: Path, o: V11Options) -> GateResult:
    from .privacy_detectors import load_detector
    from .privacy_v11 import run_privacy_v11_gate

    detector = load_detector(o.detector) if o.detector else None  # None: pinned YuNet
    return run_privacy_v11_gate(b, detector=detector, threshold=o.privacy_threshold)


def _labelmap(b: Path, o: V11Options) -> GateResult:
    from .labelmap import run_labelmap_gate

    return run_labelmap_gate(b, version=o.taxonomy_version)


RUNNERS: dict[str, Callable[[Path, V11Options], GateResult]] = {
    "join": _join, "frozen": _frozen, "leak": _leak, "bytes_once": _bytes_once, "masks": _masks,
    "splits_v11": _splits, "decon": _decon, "privacy": _privacy, "labelmap": _labelmap,
}  # fmt: skip


def run_v11_gate(name: str, build_dir: Path, options: V11Options | None = None) -> GateResult:
    try:
        return RUNNERS[name](Path(build_dir), options or V11Options())
    except Exception as exc:
        return finish(
            name,
            {"failed_gate_error": 1},
            [f"{name} gate_error: {type(exc).__name__}: {str(exc)[:200]}"],
        )
