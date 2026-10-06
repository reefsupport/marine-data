"""Column fill rates of a normalised table."""

from __future__ import annotations


def fill_rates(table) -> dict[str, float]:
    """``{column: % non-null}`` (0-100, 1 dp); an empty table fills nothing."""
    n = table.num_rows
    return {
        name: round(100.0 * (n - table[name].null_count) / n, 1) if n else 0.0
        for name in table.column_names
    }
