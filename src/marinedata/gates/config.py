"""Release-gate configuration: ``registry/gates.yaml`` (mirrors ``registry/split_ratio.yaml``)."""

from __future__ import annotations

from pathlib import Path

import yaml

from ..licence_class import _registry_root

GATES_FILE = "gates.yaml"


def load_gates_config(root: Path | None = None) -> dict:
    """``registry/gates.yaml`` as a dict; ``{}`` when the file is absent."""
    path = _registry_root(root) / GATES_FILE
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a YAML mapping")
    return data


def exif_gps_allowed(root: Path | None = None) -> frozenset[str]:
    """Source ids set to ``true`` under ``privacy.allow_exif_gps``; anything else is refused."""
    allow = (load_gates_config(root).get("privacy") or {}).get("allow_exif_gps") or {}
    if not isinstance(allow, dict):
        raise ValueError("privacy.allow_exif_gps must map a source id to true or false")
    bad = {str(k): v for k, v in allow.items() if not isinstance(v, bool)}
    if bad:
        raise ValueError(f"privacy.allow_exif_gps values must be true or false, got {bad}")
    return frozenset(str(source) for source, flag in allow.items() if flag)
