#!/usr/bin/env python3
"""Thin CLI for :mod:`marinedata.ingest_coralscapes_v2` (docs/ingest-howto.md, coralscapes-v2).

    uv run python scripts/stage_coralscapes_v2.py --download --hf-cache <dir>
    uv run python scripts/stage_coralscapes_v2.py --stage --hf-cache <dir> --out <dir>

Runs on VM 900 (D6), never on the Mac. The package has subcommands in ``cli.py`` (not owned by
WP2), so this stays a script beside ``registry_sync_bucket.py`` and friends.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marinedata.ingest_coralscapes_v2 import main

if __name__ == "__main__":
    raise SystemExit(main())
