"""Pluggable face / plate detectors for the privacy gate. The default is none: the gate then
reports ``skipped`` and counts the people-label rows only.

A detector is any object with ``name``, ``version`` and ``detect(image_bytes) -> [Detection]``.
``--detector pkg.module:attr`` names it; ``attr`` is such an object or a zero-argument factory."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    kind: str  # "face" | "plate"
    score: float


class Detector(Protocol):
    name: str
    version: str

    def detect(self, image: bytes) -> list[Detection]: ...


def load_detector(spec: str | None) -> Detector | None:
    if not spec:
        return None
    module, _, attr = spec.partition(":")
    if not module or not attr:
        raise ValueError(f"detector must be 'package.module:attribute', got {spec!r}")
    obj = getattr(importlib.import_module(module), attr)
    return obj() if isinstance(obj, type) or (callable(obj) and not hasattr(obj, "detect")) else obj
