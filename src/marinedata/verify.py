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

from .fetch import FetchError, FetchNotSupported, auto_fetchable, fetch_sample, sample_digest
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


_STATUS_MARK = {"ok": "✓", "n/a": "·", "missing": "✗"}


@dataclass(frozen=True)
class DoctorRow:
    """What is and is not done for one source, at a glance.

    Every status is one of ``"ok"`` | ``"missing"`` | ``"n/a"`` — the same three-way
    shape throughout, so nothing needs its own boolean-vs-string special case. ``n/a``
    means the question genuinely does not apply (a ``metadata-only`` layout has nothing
    to verify; a source with no declared supervision has nothing to cross), which is
    different from ``missing`` and must not read as equally incomplete.
    """

    source_id: str
    layout_status: str
    licence_status: str
    crosswalk_status: str
    fetchable_status: str

    @property
    def complete(self) -> bool:
        return "missing" not in (
            self.layout_status,
            self.licence_status,
            self.crosswalk_status,
            self.fetchable_status,
        )

    def line(self) -> str:
        return (
            f"{'✓' if self.complete else ' '} {self.source_id:<34} "
            f"layout={_STATUS_MARK[self.layout_status]}  "
            f"licence={_STATUS_MARK[self.licence_status]}  "
            f"crosswalk={_STATUS_MARK[self.crosswalk_status]}  "
            f"fetchable={_STATUS_MARK[self.fetchable_status]}"
        )


def doctor(registry: Registry) -> tuple[DoctorRow, ...]:
    """One row per source: exactly what is incomplete, no network required.

    Complements ``verify_all`` (which actually fetches) with the questions answerable
    from the registry alone: is the layout verified, is the licence primary-sourced,
    does a source that claims real supervision have a crosswalk to use it, and is a
    sample even reachable without a human. Run before ``verify --unverified-only`` to
    see the whole gap in one table instead of one source at a time.
    """
    rows = []
    for source in registry:
        supervises = source.declared_supervision()
        has_crosswalk = source.loader is not None and bool(source.loader.crosswalk_id)
        is_metadata_only = source.loader is not None and source.loader.layout == "metadata-only"

        if source.loader is None:
            layout_status = "missing"
        elif is_metadata_only:
            layout_status = "n/a"  # nothing to fetch or verify against
        else:
            layout_status = "ok" if source.loader.is_verified else "missing"

        rows.append(
            DoctorRow(
                source_id=source.id,
                layout_status=layout_status,
                licence_status="ok" if source.verification.is_primary else "missing",
                crosswalk_status=(
                    "n/a"
                    if not supervises or is_metadata_only
                    else ("ok" if has_crosswalk else "missing")
                ),
                fetchable_status="n/a"
                if is_metadata_only
                else ("ok" if auto_fetchable(source) else "missing"),
            )
        )
    return tuple(sorted(rows, key=lambda r: r.source_id))


def doctor_totals(rows: tuple[DoctorRow, ...]) -> str:
    """The trailing counts block: how many rows, and by which column they fall short."""
    complete = sum(1 for r in rows if r.complete)
    return "\n".join(
        [
            f"{complete}/{len(rows)} sources fully complete",
            f"layout unverified:      {sum(1 for r in rows if r.layout_status == 'missing')}",
            f"licence not primary:    {sum(1 for r in rows if r.licence_status == 'missing')}",
            f"crosswalk missing:      {sum(1 for r in rows if r.crosswalk_status == 'missing')}",
            f"not auto-fetchable:     {sum(1 for r in rows if r.fetchable_status == 'missing')}",
        ]
    )
