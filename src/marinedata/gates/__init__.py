"""Release gates. A gate returns a :class:`GateResult`; ``GATES.json`` is one top-level dict keyed
by gate name, each value ``{"passed", "counts", "failures"}``, merged one gate at a time."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_FAILURES = 50


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    counts: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", list(self.failures)[:MAX_FAILURES])

    def to_json(self) -> dict[str, Any]:
        return {"passed": self.passed, "counts": self.counts, "failures": self.failures}


def merge_gates_json(path: str | Path, results: Iterable[GateResult]) -> dict[str, Any]:
    """Merge ``results`` into the ``GATES.json`` at ``path`` (created if absent); returns the
    dict."""
    target = Path(path)
    data: dict[str, Any] = json.loads(target.read_text()) if target.is_file() else {}
    for result in results:
        data[result.name] = result.to_json()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


__all__ = ["MAX_FAILURES", "GateResult", "merge_gates_json"]
