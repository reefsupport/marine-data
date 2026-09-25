"""WP-8 task layers: points, VQA, semantic segmentation, and the benthic-coarse rollup.

This package holds logic that is new for WP-8 and does not belong inside the
restricted `hf_card.py` / `hf_export.py` / `hf_parquet.py` trio (D-M): only the
:mod:`.rollup` module has landed so far — see ``docs/task-layers.md`` for the
per-source inventory, the real gaps found, and what is still open.
"""

from __future__ import annotations

from .rollup import RollupResult, rollup_counts

__all__ = ["RollupResult", "rollup_counts"]
