"""D-AJ: a PANGAEA tape-recall 503 is deferred, not failed (see ``adapters._http.TapeRecall``).

A deferred item never counts toward the D-AF abort threshold (:class:`.ingest_missing.
MissingLedger`) and never lands in ``MISSING.tsv`` — PANGAEA's own hint (the file comes
back once the recall completes) makes it *pending*, not dead. Each source keeps a small
local checkpoint of its currently-pending items (``key``, ``url``) in the run's work dir;
the next pass over the same work dir reads it and retries those items first, and a
successful fetch clears the entry.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .adapters import RemoteItem

COLUMNS = ("key", "url")


class DeferredLedger:
    """Tracks items deferred by a tape recall across passes over one work dir."""

    def __init__(self, journal: Path) -> None:
        self.journal = journal
        self.pending: dict[str, str] = {}
        if journal.exists():
            lines = journal.read_text().splitlines()
            for line in lines[1:] if lines else []:
                parts = line.split("\t")
                if len(parts) == len(COLUMNS):
                    self.pending[parts[0]] = parts[1]

    def defer(self, key: str, url: str) -> None:
        """Record ``key`` as pending (a tape recall in progress); idempotent."""
        if key in self.pending:
            return
        self.pending[key] = url
        self._persist()

    def resolve(self, key: str) -> None:
        """Clear ``key`` once it has fetched successfully."""
        if self.pending.pop(key, None) is not None:
            self._persist()

    def order(self, items: list[RemoteItem]) -> list[RemoteItem]:
        """``items`` with any still-pending keys moved first, pending order preserved."""
        if not self.pending:
            return items
        by_key = {item.key: item for item in items}
        head = [by_key[key] for key in self.pending if key in by_key]
        head_keys = {item.key for item in head}
        return head + [item for item in items if item.key not in head_keys]

    def _persist(self) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        lines = ["\t".join(COLUMNS)] + [f"{k}\t{v}" for k, v in self.pending.items()]
        self.journal.write_text("\n".join(lines) + "\n")

    @property
    def count(self) -> int:
        return len(self.pending)
