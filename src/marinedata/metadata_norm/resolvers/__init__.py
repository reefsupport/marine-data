"""Per-row metadata resolvers (PA-P2.1): geo, date, depth, site, quadrat a source carries."""

from .registry import RESOLVERS, resolver_for
from .run import resolve_source
from .schema import OUTPUT_COLUMNS, RESOLVED_COLUMNS, ResolvedRow, Unresolved

__all__ = [
    "OUTPUT_COLUMNS",
    "RESOLVED_COLUMNS",
    "RESOLVERS",
    "ResolvedRow",
    "Unresolved",
    "resolve_source",
    "resolver_for",
]
