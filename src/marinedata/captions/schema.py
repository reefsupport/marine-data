"""Loads and validates against `registry/captions/schema.yaml`.

The YAML is the single source of truth for the captions parquet's columns; this module
only reads it (there is no hardcoded duplicate list to drift out of sync — a test
round-trips every column name through `validate_frame`).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

_PANDAS_DTYPE_OK: dict[str, tuple[str, ...]] = {
    # "str" is pandas >= 3's default inferred dtype for a column of Python strings;
    # "object"/"string" cover pandas 2.x and the nullable StringDtype.
    "string": ("object", "string", "str"),
    "int64": ("int64", "Int64", "float64"),  # float64 covers an all-null nullable int col
}


def _default_schema_path() -> Path:
    """Packaged registry, falling back to the repo layout for editable installs —
    same lookup `Registry._default_root` uses."""
    packaged = Path(__file__).resolve().parents[1] / "_registry" / "captions" / "schema.yaml"
    if packaged.is_file():
        return packaged
    repo = Path(__file__).resolve().parents[3] / "registry" / "captions" / "schema.yaml"
    if repo.is_file():
        return repo
    raise FileNotFoundError(
        "registry/captions/schema.yaml not found (packaged or repo layout). "
        "Pass an explicit path to load_schema()."
    )


def load_schema(path: str | Path | None = None) -> list[dict]:
    """Read `registry/captions/schema.yaml`'s `columns` list."""
    with open(path or _default_schema_path(), encoding="utf-8") as fh:
        return yaml.safe_load(fh)["columns"]


def required_columns(schema: list[dict]) -> list[str]:
    return [c["name"] for c in schema]


def validate_frame(df: pd.DataFrame, schema: list[dict]) -> list[str]:
    """Return a list of human-readable problems; empty means the frame is valid.

    Checks: every schema column is present, no unexpected extra columns, dtype is
    plausible, and non-nullable columns have no nulls.
    """
    problems: list[str] = []
    expected = {c["name"] for c in schema}
    actual = set(df.columns)

    missing = expected - actual
    if missing:
        problems.append(f"missing columns: {sorted(missing)}")
    extra = actual - expected
    if extra:
        problems.append(f"unexpected columns: {sorted(extra)}")

    for col in schema:
        name = col["name"]
        if name not in df.columns:
            continue
        if not col.get("nullable", True) and df[name].isna().any():
            problems.append(f"{name}: non-nullable but has nulls")
        dtype = col.get("dtype")
        ok_kinds = _PANDAS_DTYPE_OK.get(dtype)
        if ok_kinds and df[name].dtype.name not in ok_kinds:
            problems.append(f"{name}: dtype {df[name].dtype.name} not in {ok_kinds}")

    return problems
