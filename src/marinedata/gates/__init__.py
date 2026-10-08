"""Release gates. Each gate returns a :class:`GateResult`; ``marinedata release gate`` merges it
into ``GATES.json`` next to the build it checked."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MAX_FAILURES = 50
GATES_FILE = "GATES.json"


@dataclass(frozen=True)
class GateResult:
    """``failures`` is capped at :data:`MAX_FAILURES`; ``counts`` is always complete."""

    name: str
    passed: bool
    counts: dict
    failures: list

    def to_dict(self) -> dict:
        return {"passed": self.passed, "counts": self.counts, "failures": list(self.failures)}


def write_gates(directory: Path, result: GateResult) -> Path:
    """Merge ``result`` into ``<directory>/GATES.json``; other gates' entries are kept."""
    path = Path(directory) / GATES_FILE
    existing = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(existing, dict):
        raise ValueError(f"{path} is not a JSON object")
    merged = {**existing, result.name: result.to_dict()}
    path.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    return path
