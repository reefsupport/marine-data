"""Release gates. Each gate returns a :class:`GateResult`; ``marinedata release gate`` merges it
into ``GATES.json`` next to the build it checked and recomputes the ``summary`` entry.
``GATES.json`` is one top-level dict keyed by gate name, each value ``{"passed", "counts",
"failures"}`` (plus ``skipped`` when the gate did not run), and one ``summary`` entry.

A gate that could not run says why (``skipped``) and is never counted as passed; a gate that
checked nothing fails instead of passing. ``release publish`` (P1.8) calls :func:`verdict`."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_FAILURES = 50
GATES_FILE = "GATES.json"
SUMMARY_KEY = "summary"
REQUIRED = ("licence", "dedup", "splits", "masks", "load", "card")
"""Gates every publish needs. Any other gate (privacy, croissant, metadata) is required once it
ran, i.e. unless it reported ``skipped``."""


@dataclass(frozen=True)
class GateResult:
    """``failures`` is capped at :data:`MAX_FAILURES`; ``counts`` is always complete.
    ``skipped`` (a reason) marks a gate that did not run: it is never ``passed``."""

    name: str
    passed: bool
    counts: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    skipped: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "failures", list(self.failures)[:MAX_FAILURES])

    @property
    def status(self) -> str:
        return gate_status(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "passed": self.passed and self.skipped is None,
            "counts": self.counts,
            "failures": list(self.failures),
        }
        if self.skipped is not None:
            out["skipped"] = self.skipped
        return out

    to_json = to_dict


def finish(name: str, counts: Mapping, failures: list[str]) -> GateResult:
    """Result of a gate that ran: passes only without a failure line."""
    out = {**dict(sorted(counts.items())), "failures_total": len(failures)}
    return GateResult(name, not failures, out, failures[:MAX_FAILURES])


def skip(name: str, reason: str, counts: Mapping | None = None) -> GateResult:
    return GateResult(name, False, dict(counts or {}), [], reason)


def gate_status(entry: object) -> str:
    """``passed`` | ``failed`` | ``skipped`` of one GATES.json entry (tolerates stub entries)."""
    if not isinstance(entry, dict):
        return "failed"
    if entry.get("skipped") is not None:
        return "skipped"
    return "passed" if entry.get("passed") is True else "failed"


def summarize(gates: Mapping) -> dict:
    entries = {k: v for k, v in gates.items() if k != SUMMARY_KEY}
    status = {k: gate_status(v) for k, v in entries.items()}
    extra = sorted(k for k in entries if k not in REQUIRED and status[k] != "skipped")
    required = [*REQUIRED, *extra]
    return {
        "passed": sum(s == "passed" for s in status.values()),
        "failed": sum(s == "failed" for s in status.values()),
        "skipped": sum(s == "skipped" for s in status.values()),
        "required_for_publish": required,
        "blocking": [n for n in required if status.get(n) != "passed"],
    }


def read_gates(directory: Path) -> dict:
    return read_gates_file(Path(directory) / GATES_FILE)


def read_gates_file(path: str | Path) -> dict:
    target = Path(path)
    if not target.exists():
        return {}
    data = json.loads(target.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{target} is not a JSON object")
    return data


def merge_gates_json(path: str | Path, results: Iterable[GateResult]) -> dict[str, Any]:
    """Merge ``results`` into the ``GATES.json`` at ``path`` (created if absent): other gates'
    entries are kept and the ``summary`` entry is recomputed. Returns the merged dict."""
    target = Path(path)
    merged = read_gates_file(target)
    for result in results:
        merged[result.name] = result.to_dict()
    merged[SUMMARY_KEY] = summarize(merged)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(merged, indent=1, sort_keys=True) + "\n")
    return merged


def write_gates(directory: Path, result: GateResult) -> Path:
    """Merge ``result`` into ``<directory>/GATES.json`` (see :func:`merge_gates_json`)."""
    path = Path(directory) / GATES_FILE
    merge_gates_json(path, [result])
    return path


def verdict(directory: Path) -> tuple[bool, list[str]]:
    """``(ok, problems)``: ok when every required gate in GATES.json is ``passed``. A missing
    GATES.json, a missing required gate and a skipped required gate are all problems."""
    gates = read_gates(directory)
    problems = []
    for name in summarize(gates)["blocking"]:
        status = gate_status(gates[name]) if name in gates else "missing"
        why = gates[name].get("skipped") if status == "skipped" else None
        problems.append(f"{name}: {status}" + (f" ({why})" if why else ""))
    return not problems, problems


__all__ = [
    "GATES_FILE",
    "MAX_FAILURES",
    "REQUIRED",
    "SUMMARY_KEY",
    "GateResult",
    "finish",
    "gate_status",
    "merge_gates_json",
    "read_gates",
    "read_gates_file",
    "skip",
    "summarize",
    "verdict",
    "write_gates",
]
