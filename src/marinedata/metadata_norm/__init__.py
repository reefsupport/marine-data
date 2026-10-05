"""``marinedata.metadata_norm``: per-source metadata normalisers (WP-U14a)."""

from __future__ import annotations

from .base import EXTRA_COLUMNS, OUTPUT_COLUMNS, Normaliser, NormContext, Staged, to_table
from .default import default_normalise
from .per_row import fathomnet, inat_marine, mermaid_aws, planktonzilla, qut_fish

NORMALISERS: dict[str, Normaliser] = {
    "fathomnet": fathomnet,
    "inat-marine": inat_marine,
    "mermaid-aws": mermaid_aws,
    "planktonzilla": planktonzilla,
    "qut-fish": qut_fish,
}


def normalise(source_id: str, version: str, staged: Staged, ctx: NormContext):
    """Dispatch to the source's normaliser (default when it has none)."""
    return NORMALISERS.get(source_id, default_normalise)(source_id, version, staged, ctx)


__all__ = [
    "EXTRA_COLUMNS",
    "NORMALISERS",
    "OUTPUT_COLUMNS",
    "NormContext",
    "Normaliser",
    "default_normalise",
    "normalise",
    "to_table",
]
