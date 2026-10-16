"""``release split-map`` glue for the benchmark pins (design eval-decontamination-split-v2, 2.2)."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .benchmark_pins import PIN_TEST, PIN_VAL, load_benchmark_pins
from .benchmarks import BenchmarkRegistry, BenchmarksError
from .registry import Registry


def add_pin_args(parser: argparse.ArgumentParser) -> None:
    """Register ``--no-benchmark-pins`` (the manifest pins are on by default)."""
    parser.add_argument(
        "--no-benchmark-pins",
        action="store_true",
        help=(
            "do not pin the benchmark manifests' upstream eval images to test and their second "
            "held-out split to val (default: pin them; a pin wins over a v1.0 assignment under "
            "--inherit and the override is recorded in the map)"
        ),
    )


def pin_kwargs(args: argparse.Namespace, registry: Registry | None) -> dict[str, Any]:
    """``generate_split_map`` kwargs for the pins: ``{}`` under ``--no-benchmark-pins``.

    The benchmark registry and manifests come from the ``--registry`` directory when it holds a
    ``benchmarks.yaml``, else from the packaged registry."""
    if getattr(args, "no_benchmark_pins", False):
        return {}
    found = getattr(registry, "root", None)
    root = Path(found) if found else None
    listing = root / "benchmarks.yaml" if root else None
    try:
        benchmarks = BenchmarkRegistry.load(listing if listing and listing.is_file() else None)
        return {"benchmark_pins": load_benchmark_pins(benchmarks, root)}
    except BenchmarksError as exc:
        raise ValueError(f"benchmark pins: {exc}") from exc


def pins_summary(header: Mapping[str, Any]) -> str:
    """The CLI lines for a ``pins`` header: counts per benchmark, then the v1.0 overrides."""
    per = "; ".join(
        f"{label} test={c.get(PIN_TEST, 0)} validation={c.get(PIN_VAL, 0)}"
        for label, c in (header.get("benchmarks") or {}).items()
    )
    components, skipped = header.get("components") or {}, header.get("skipped_components") or {}
    lines = [
        f"release: benchmark pins {per or 'none'} | components test={components.get(PIN_TEST, 0)} "
        f"validation={components.get(PIN_VAL, 0)} | yielded ood={skipped.get('ood', 0)} "
        f"earlier-rule={skipped.get('earlier_rule', 0)}"
    ]
    over = header.get("v1_continuity_overridden")
    if over is not None:
        line = f"release: benchmark pins override v1 continuity for {over['images']} image(s)"
        if over["images"]:
            line += f" in {over['groups']} group(s): {over['by_source']}"
        lines.append(line)
    return "\n".join(lines)
