"""Eval harness core (WP-12, design doc §4): metric registry, cluster bootstrap CIs,
and prediction readers. See ``docs/design/eval-decontamination-split-v2.md``.

This package intentionally does not depend on the split-v2 registry (P3) or the
benchmark manifests (P1) — see the P4 row of §5 ("needs merged first: none"). Metric
and bootstrap functions operate on plain arrays; ``eval score`` (this package's CLI,
wired into ``marinedata`` via :func:`marinedata.eval.cli.add_eval_subparser`) reads
ground truth and predictions from files supplied on the command line.
"""

from __future__ import annotations

from .bootstrap import BootstrapResult, cluster_bootstrap
from .metrics import METRICS, Metric

__all__ = [
    "METRICS",
    "BootstrapResult",
    "Metric",
    "cluster_bootstrap",
]
