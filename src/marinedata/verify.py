"""Layout verification — does the declared layout match the real dataset?

The gap this closes: synthetic fixtures prove a *reader* works; only real data proves a
*declaration* is right. Both matter, and they fail differently. A reader bug throws; a
wrong declaration quietly finds nothing, or finds the wrong thing.

Verification is deliberately cheap — around 100 items per source. Enough to confirm the
directories exist, the columns are named what we claim, and the loader yields sane
samples. Not enough to be a download.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .fetch import FetchError, FetchNotSupported, fetch_sample, sample_digest
from .loaders import LoaderError, build_loader
from .models import Source
from .registry import Registry


@dataclass(frozen=True)
class VerifyResult:
    """Outcome of verifying one source's declared layout against real data."""

    source_id: str
    status: str
    """``verified`` | ``failed`` | ``unfetchable`` | ``skipped``"""

    items: int = 0
    detail: str = ""
    digest: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "verified"

    def line(self) -> str:
        mark = {"verified": "✓", "failed": "✗", "unfetchable": "·", "skipped": "-"}[self.status]
        suffix = f"  {self.detail}" if self.detail else ""
        return f"{mark} {self.source_id:<34} {self.status:<12} {self.items or '':>4}{suffix}"


def verify_source(
    source: Source,
    *,
    registry: Registry | None = None,
    limit: int = 100,
    root: Path | None = None,
    force: bool = False,
) -> VerifyResult:
    """Fetch a bounded sample and run the declared loader over it.

    ``unfetchable`` is not a failure. Gated, request-only, S3 and scrape sources
    legitimately need a human, and reporting that distinctly keeps genuine breakage
    visible instead of drowning it in expected noise.
    """
    if source.loader is None:
        return VerifyResult(source.id, "skipped", detail="no loader declared")
    if source.loader.layout == "metadata-only":
        return VerifyResult(source.id, "skipped", detail="metadata-only by design")

    try:
        fetched = fetch_sample(source, limit=limit, root=root, force=force)
    except FetchNotSupported as exc:
        return VerifyResult(source.id, "unfetchable", detail=str(exc).split("—")[-1].strip()[:80])
    except FetchError as exc:
        return VerifyResult(source.id, "failed", detail=f"fetch: {exc}"[:160])

    try:
        loader = build_loader(source, fetched.root, partial=True)
        if registry is not None:
            harmonizer = registry.harmonizer_for(source.id)
            if harmonizer is not None and hasattr(loader, "bind_harmonizer"):
                loader.bind_harmonizer(harmonizer)
        samples = list(loader)
    except LoaderError as exc:
        return VerifyResult(
            source.id,
            "failed",
            items=fetched.items,
            detail=f"layout mismatch: {exc}"[:200],
        )

    if not samples:
        return VerifyResult(source.id, "failed", detail="loader yielded nothing")

    return VerifyResult(
        source.id,
        "verified",
        items=len(samples),
        digest=sample_digest(fetched.root),
        detail=f"layout '{source.loader.layout}' confirmed",
    )


def verify_all(
    registry: Registry,
    *,
    limit: int = 100,
    only: list[str] | None = None,
    root: Path | None = None,
) -> list[VerifyResult]:
    """Verify every source, or a named subset."""
    sources = [registry.source(sid) for sid in only] if only else list(registry)
    return [
        verify_source(
            src,
            registry=registry,
            limit=limit,
            root=(root / src.id) if root else None,
        )
        for src in sources
    ]


def unverified(registry: Registry) -> tuple[Source, ...]:
    """Sources whose layout has never been checked against real data.

    These are declarations transcribed from documentation. Treat them as hypotheses.
    """
    return tuple(
        s
        for s in registry
        if s.loader is not None and s.loader.layout != "metadata-only" and not s.loader.is_verified
    )


def summarise(results: list[VerifyResult]) -> str:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    parts = [f"{status}={count}" for status, count in sorted(counts.items())]
    return "  ".join(parts)
