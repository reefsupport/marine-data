# ruff: noqa: E501
"""Croissant gate (P1.6): ``croissant.json`` is well-formed and validates with ``mlcroissant``.

``skipped`` (never passed) when the file is absent or ``mlcroissant`` is not installed
(``uv sync --extra croissant``). A file that is not JSON or lacks ``@context``/``@type``/``name``
fails even without ``mlcroissant``. Required for publish only when the gate ran."""

from __future__ import annotations

import json
from pathlib import Path

from . import GateResult, finish, skip

NAME = "croissant"
FILE = "croissant.json"
_KEYS = ("@context", "@type", "name")


def run_croissant_gate(build_dir: Path) -> GateResult:
    path = Path(build_dir) / FILE
    if not path.is_file():
        return skip(NAME, f"no {FILE} in the build")
    try:
        doc = json.loads(path.read_text())
    except ValueError as exc:
        return finish(NAME, {}, [f"{FILE} unparseable_json: {str(exc)[:120]}"])
    missing = [k for k in _KEYS if not isinstance(doc, dict) or k not in doc]
    if missing:
        return finish(NAME, {}, [f"{FILE} missing_key: {', '.join(missing)}"])
    try:
        import mlcroissant as mlc
    except ImportError:
        return skip(
            NAME,
            "mlcroissant is not installed (uv sync --extra croissant); the file parses, unvalidated",
        )
    try:
        dataset = mlc.Dataset(jsonld=path)
    except Exception as exc:
        lines = [ln.strip() for ln in str(exc).splitlines() if ln.strip()]
        return finish(
            NAME,
            {},
            [f"{FILE} invalid_croissant: {ln[:200]}" for ln in lines[:10]]
            or [f"{FILE} invalid_croissant: {exc!r}"],
        )
    warnings = getattr(getattr(dataset.metadata, "issues", None), "warnings", ()) or ()
    return finish(NAME, {"warnings": len(warnings)}, [])
