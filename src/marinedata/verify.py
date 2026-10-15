"""Layout verification — does the declared layout match the real dataset?

The gap this closes: synthetic fixtures prove a *reader* works; only real data proves a
*declaration* is right. Both matter, and they fail differently. A reader bug throws; a
wrong declaration quietly finds nothing, or finds the wrong thing.

Verification is deliberately cheap — around 100 items per source. Enough to confirm the
directories exist, the columns are named what we claim, and the loader yields sane
samples. Not enough to be a download.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

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


def write_verification(registry: Registry, result: VerifyResult) -> Path | None:
    """Persist a successful layout check back into the source's YAML.

    Only ever called for ``result.status == "verified"`` — a failed or unfetchable
    result records nothing, because there is nothing honest to write. This is a
    targeted text edit of the two ``loader:`` fields, not a full YAML round-trip:
    these files carry hand-written prose (module comments, folded descriptions) that
    a load-then-dump cycle would silently reformat across every entry in the file,
    not just the one being verified.

    Returns the path written, or ``None`` if there was nothing to write to (no
    filesystem root — e.g. a registry constructed directly rather than loaded — or
    the source could not be found on disk).
    """
    if registry.root is None or not result.ok:
        return None
    sources_dir = registry.root / "sources"
    if not sources_dir.is_dir():
        return None

    entry_pattern = re.compile(rf"^\s*- id:\s*{re.escape(result.source_id)}\s*$", re.MULTILINE)
    note = f"marinedata verify — {result.items} item(s), {result.digest or 'no digest'}"
    for path in sorted(sources_dir.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if not entry_pattern.search(text):
            continue
        patched = _patch_loader_verification(text, result.source_id, date.today(), note)
        if patched != text:
            path.write_text(patched, encoding="utf-8")
        return path
    return None


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _remove_key_block(
    lines: list[str], lo: int, hi: int, indent: int, key: str
) -> tuple[bool, int, int]:
    """Remove ``key:`` — and any folded/continuation lines under it — at exactly
    ``indent`` spaces within ``lines[lo:hi]``.

    Returns ``(found, insertion_index, new_hi)``: where a replacement line should go,
    and the block's new end once the old lines are gone.
    """
    pad = " " * indent
    for i in range(lo, hi):
        if lines[i][: len(pad)] == pad and lines[i][len(pad) :].startswith(f"{key}:"):
            j = i + 1
            while j < hi and (lines[j].strip() == "" or _line_indent(lines[j]) > indent):
                j += 1
            del lines[i:j]
            return True, i, hi - (j - i)
    return False, hi, hi


def _patch_loader_verification(text: str, source_id: str, verified_on: date, note: str) -> str:
    """Set ``loader.verified_on``/``loader.verified_note`` for one source entry.

    Handles both shapes a ``loader:`` mapping takes in this registry: multi-line
    block style, and single-line flow style (``loader: { layout: ..., ... }``).
    """
    lines = text.split("\n")
    entry_re = re.compile(r"^(\s*)- id:\s*" + re.escape(source_id) + r"\s*$")
    start = next((i for i, ln in enumerate(lines) if entry_re.match(ln)), None)
    if start is None:
        return text
    entry_indent = _line_indent(lines[start])

    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() and _line_indent(lines[i]) <= entry_indent:
            end = i
            break

    loader_idx = loader_indent = None
    flow_value: str | None = None
    for i in range(start, end):
        stripped = lines[i].strip()
        if stripped.startswith("loader:"):
            loader_idx = i
            loader_indent = _line_indent(lines[i])
            flow_value = stripped[len("loader:") :].strip() or None
            break
    if loader_idx is None or loader_indent is None:
        return text  # no loader block to write into — nothing this function can do

    if flow_value is not None:
        parsed = yaml.safe_load(flow_value)
        if not isinstance(parsed, dict):
            return text  # not a mapping — unrecognised shape, leave it untouched
        parsed = dict(parsed)
        parsed["verified_on"] = verified_on
        parsed["verified_note"] = note
        rendered = yaml.safe_dump(
            parsed, default_flow_style=True, sort_keys=False, width=1000, allow_unicode=True
        ).strip()
        lines[loader_idx] = f"{' ' * loader_indent}loader: {rendered}"
        return "\n".join(lines)

    block_end = end
    for i in range(loader_idx + 1, end):
        if lines[i].strip() and _line_indent(lines[i]) <= loader_indent:
            block_end = i
            break
    child_indent = loader_indent + 2
    for i in range(loader_idx + 1, block_end):
        if lines[i].strip():
            child_indent = _line_indent(lines[i])
            break

    new_vo = f"{' ' * child_indent}verified_on: {verified_on.isoformat()}"
    _, ins, block_end = _remove_key_block(
        lines, loader_idx + 1, block_end, child_indent, "verified_on"
    )
    lines.insert(ins, new_vo)
    block_end += 1

    new_vn = f"{' ' * child_indent}verified_note: {json.dumps(note)}"
    _, ins, block_end = _remove_key_block(
        lines, loader_idx + 1, block_end, child_indent, "verified_note"
    )
    lines.insert(ins, new_vn)

    return "\n".join(lines)


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


_OPEN_VOCABULARY_SCHEMAS = frozenset({"worms-species", "sonotype"})
"""Schemas declaring zero nodes *by design*, not because nobody has populated them yet
— see their descriptions in registry/schemas/open-vocabs.yaml. A static crosswalk_id
is the wrong mechanism for these (worms-species is resolved via WoRMS at genus/family
rank through worms-genus or a fauna schema instead; sonotype explicitly has no
taxonomic mapping to supply). ``dataset-native`` is deliberately excluded: its own
description says "no published crosswalk *yet*" — that is a real, closeable gap, not
an open vocabulary."""


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
        is_open_vocabulary = (
            source.loader is not None and source.loader.schema_id in _OPEN_VOCABULARY_SCHEMAS
        )

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
                    if not supervises or is_metadata_only or is_open_vocabulary
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
