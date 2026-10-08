"""Thin adapters onto the remote fetchers, and the stub for registered-but-unbuilt api clients."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..models import Source
from .core import FetchResult


def _fetch_s3(source: Source, root: Path, limit: int) -> FetchResult:
    from ..fetchers_remote import fetch_s3

    return fetch_s3(source, root, limit)


def _fetch_fathomnet_client(source: Source, root: Path, limit: int) -> FetchResult:
    from ..fetchers_remote import fetch_fathomnet

    return fetch_fathomnet(source, root, limit)


def _unimplemented_client(client: str) -> Callable[[Source, Path, int], FetchResult]:
    """One stub factory shared by every registered-but-unbuilt api client.

    Each of the bespoke clients below (obis, allen-coral-atlas, copernicus-marine,
    coralnet, atlantis) is a separate workstream (master brief WS-D: 7 non-bulk
    sources) — this raises rather than silently reusing another source's client.
    """

    def _fetch(source: Source, root: Path, limit: int) -> FetchResult:
        raise NotImplementedError(
            f"{source.id}: api client '{client}' is known but not yet implemented — "
            "building it is a separate workstream (master brief WS-D: 7 non-bulk sources)."
        )

    return _fetch
