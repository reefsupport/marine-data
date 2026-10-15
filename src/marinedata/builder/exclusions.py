"""What a build drops: the sources-dropped error and the partial-abstain exclusion record."""

from __future__ import annotations

from dataclasses import dataclass


class SourcesDroppedError(RuntimeError):
    """Raised when the build completed but one or more permitted sources contributed
    nothing.

    The full pass still runs — every source is attempted, so this lists every drop
    rather than stopping at the first — but a build that silently lost a source and
    still returned as if nothing had happened would be the same failure class as
    reporting a denied source as included. Letting this propagate is the point: an
    uncaught exception exits the process non-zero, so a caller cannot read "no error"
    as "every source was included."

    The build that *did* complete is attached as ``.dataset`` (a :class:`Dataset` or
    :class:`StreamingDataset`) for a caller who deliberately wants the partial result
    anyway — catch this, read ``.dropped`` to decide, and pull ``.dataset`` back out.
    """

    def __init__(self, dropped: dict[str, str], dataset: object) -> None:
        self.dropped = dict(dropped)
        self.dataset = dataset
        detail = "; ".join(f"{k}: {v}" for k, v in sorted(dropped.items()))
        super().__init__(
            f"{len(dropped)} permitted source(s) contributed nothing and were dropped: {detail}"
        )


@dataclass(frozen=True)
class PartialAbstainExclusion:
    """A source dropped from one task's axis entirely (WS-D S26).

    Not a per-row abstention — this is what happens when a source's OWN native labels
    on the task's axis are a mix of resolved and coarser-abstaining (see
    ``SourceFit.partial_abstain``). Keeping the resolved subset would confound the label
    with the source itself — a Roboflow set whose "Unhealthy" rows abstain from
    `bleaching-condition` while its "Healthy" rows are kept would teach a model
    "looks like Roboflow" rather than "looks healthy". So the source contributes no
    rows to this task at all, and that is reported here rather than happening silently.
    """

    source_id: str
    task_id: str
    abstaining_labels: tuple[str, ...]
    """The native labels responsible — the coarser ones, e.g. ``("Unhealthy",)``."""
    rows_dropped: int
    """The source's total item count (``Source.primary_count``), since none of it reaches
    this task once excluded."""
