"""Task-label producers (WP-8c, charter D-Z2).

Each producer reads one already-staged/cached source and writes
``data/_tasklabels/<source_id>/<task>.parquet`` — the fixed D-Z2 interface: ``sha256``,
``source_id``, ``label_origin``, plus the task's own payload columns. Producers never
touch the network and never decide canonical/coarse classes — that projection happens in
:mod:`marinedata.task_layers.configs`, which is the one place the WP-7 crosswalk and the
D-Y rollup are applied.

This package is WP-8c's own producers for the three sources with usable data today
(Reefolution points, Coralscapes masks, CoralVQA train Q/A). A parallel worker (WP-8d)
adds more under ``marinedata.task_layers.sources`` for other sources — a separate module,
so the two never touch the same files.
"""

from __future__ import annotations
