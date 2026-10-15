"""Shared fetch types: the error hierarchy, ``FetchResult`` and the cache location."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class FetchError(Exception):
    """A sample could not be retrieved."""


class IntegrityError(FetchError):
    """Fetched bytes do not hash to the digest the manifest / CHECKSUMS / registry pins.

    Names the key (or URL) and both digests. Never retried: the transfer completed, so a
    second download would only fetch the same wrong bytes again."""


class FetchNotSupported(FetchError):
    """No fetcher exists for this access method.

    Raised rather than silently returning nothing, so "we cannot automate this" is
    distinguishable from "this dataset is empty". Gated and request-only sources land
    here by design — they require a human to accept terms, which is exactly the
    ``contract_gated`` situation the licence model already tracks.
    """


def cache_root() -> Path:
    """Cache location. ``MARINEDATA_CACHE`` overrides."""
    env = os.environ.get("MARINEDATA_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "marinedata"


@dataclass(frozen=True)
class FetchResult:
    source_id: str
    root: Path
    items: int
    method: str
    truncated: bool
    """True when the source has more items than were fetched — always, for a sample."""

    def manifest(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "items": self.items,
            "method": self.method,
            "truncated": self.truncated,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "note": (
                "Bounded verification sample. NOT the full dataset."
                if self.truncated
                else "Complete fetch of the source, not a bounded sample."
            ),
        }


def facade_cap(name: str) -> int:
    """Current value of a cap constant, read from the public ``marinedata.fetch`` facade.

    The caps were module globals of the single ``fetch.py`` and tests (or an operator) override
    them as ``marinedata.fetch.MAX_MEMBER_BYTES = ...``. Resolving the facade attribute on every
    call keeps that override working after the split instead of binding the value at import."""
    import sys

    return getattr(sys.modules["marinedata.fetch"], name)
