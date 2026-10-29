#!/usr/bin/env python3
"""Verify that every source's native labels resolve into rs-benthic-v1 (registry only).

    uv run python scripts/verify_label_matching.py            # run the checks, print a summary
    uv run python scripts/verify_label_matching.py --check    # also fail when the results are stale
    uv run python scripts/verify_label_matching.py --write    # refresh results/ and docs/

Writes ``results/label-matching-v1.2.0.json`` and the generated blocks of
``docs/label-matching.md``. ``make labels-check`` runs ``--check``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marinedata.gates import verify_matching_cross as cross
from marinedata.gates.verify_matching_render import blocks, splice

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "label-matching-v1.2.0.json"
DOC = ROOT / "docs" / "label-matching.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="fail on a check failure or stale output"
    )
    mode.add_argument("--write", action="store_true", help="rewrite the JSON and the doc blocks")
    args = parser.parse_args(argv)

    report = cross.run()
    text = cross.dumps(report)
    doc = splice(DOC.read_text(), blocks(report))
    problems = [f"FAIL {f}" for f in report["failures"]]
    if args.write:
        RESULT.parent.mkdir(parents=True, exist_ok=True)
        RESULT.write_text(text)
        DOC.write_text(doc)
    elif args.check:
        if not RESULT.exists() or RESULT.read_text() != text:
            problems.append(
                f"STALE {RESULT.relative_to(ROOT)}: run scripts/verify_label_matching.py --write"
            )
        if DOC.read_text() != doc:
            problems.append(
                f"STALE {DOC.relative_to(ROOT)}: run scripts/verify_label_matching.py --write"
            )

    s = report["summary"]
    status = ", ".join(f"{k} {v}" for k, v in s["by_status"].items())
    print(
        f"label matching: {s['units']} units ({status}); {s['labels_verified']} labels, "
        f"{s['crosswalk_edges']} edges in {s['crosswalks']} crosswalks; "
        f"unresolved {s['unresolved']}, "
        f"ambiguous {s['ambiguous']}, missing {s['missing']}, duplicate {s['duplicate']}"
    )
    for line in problems[:40]:
        print(line)
    if len(problems) > 40:
        print(f"... and {len(problems) - 40} more")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
