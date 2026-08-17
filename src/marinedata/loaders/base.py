"""Loader contract.

Forty bespoke loader classes would be untestable — you cannot download 400 GB in CI, and
a loader that is never exercised is a loader that is wrong. So sources *declare* their
on-disk layout, and a small set of generic readers implement each layout once.

Consequences worth stating:

- Adding a source is usually a YAML change, not a code change.
- Every layout is tested against a synthetic fixture, so all sources sharing that layout
  inherit real coverage.
- Where a source genuinely needs bespoke logic, it registers a custom loader — and that
  loader still has to satisfy this contract.

Loaders never download. They read what is already on disk and fail loudly if it is not
there, because silent emptiness is indistinguishable from a filter that matched nothing.
"""

from __future__ import annotations

import abc
from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar

from ..models import Source
from ..sample import Sample
from ..schema import Crosswalk


class LoaderError(Exception):
    """Data is missing, malformed, or does not match the declared layout."""


class DataNotAvailable(LoaderError):
    """The source root does not exist or is empty.

    Distinct from a malformed layout: this means "you have not fetched this yet",
    which is a different problem with a different fix.
    """


class SourceLoader(abc.ABC):
    """Reads one source from disk and yields canonical :class:`Sample` objects."""

    layout: ClassVar[str]

    def __init__(
        self,
        source: Source,
        root: str | Path,
        *,
        crosswalk: Crosswalk | None = None,
        split: str | None = None,
        partial: bool = False,
    ) -> None:
        self.source = source
        self.root = Path(root)
        self.crosswalk = crosswalk
        self.split = split
        self.partial = partial
        """Tolerate incompleteness. Set only when reading a *sample*: a sampled
        image/mask set is legitimately missing pairs, whereas a full dataset with a
        missing mask has a real defect that should not be swallowed."""
        self.params = dict(source.loader.params) if source.loader else {}

    # ── contract ──────────────────────────────────────────────────────────

    @abc.abstractmethod
    def _iter_samples(self) -> Iterator[Sample]:
        """Yield samples. Implementations may assume :meth:`validate` has passed."""

    def validate(self) -> None:
        """Check the root exists and looks like the declared layout.

        Raise :class:`DataNotAvailable` when the data is simply absent, and
        :class:`LoaderError` when it is present but wrong. Subclasses should extend this
        with layout-specific checks rather than discovering problems mid-iteration.
        """
        if not self.root.exists():
            raise DataNotAvailable(
                f"{self.source.id}: root does not exist: {self.root}. "
                f"Fetch it from {self.source.access.uri or self.source.access.method.value} first."
            )
        if self.root.is_dir() and not any(self.root.iterdir()):
            raise DataNotAvailable(f"{self.source.id}: root is empty: {self.root}")

    def partitions(self) -> list[Path]:
        """Sub-roots to iterate, when a dataset is split by site, survey or region.

        Many field datasets are partitioned this way — ours has eight site directories,
        each a self-contained ``images/`` + ``masks/`` + export bundle. Declaring
        ``partition_glob`` lets one layout serve the whole set instead of forcing
        callers to loop, and it is a common enough shape to belong in the base class.
        """
        glob = self._param("partition_glob")
        if not glob:
            return [self.root]
        found = sorted(p for p in self.root.glob(str(glob)) if p.is_dir())
        if not found:
            raise LoaderError(
                f"{self.source.id}: partition_glob '{glob}' matched no directories "
                f"under {self.root}."
            )
        return found

    def __iter__(self) -> Iterator[Sample]:
        # Root-level existence first; layout checks run per partition, since a
        # partitioned dataset satisfies its layout inside each sub-root, not at the top.
        SourceLoader.validate(self)
        count = 0
        base = self.root
        try:
            for partition in self.partitions():
                self.root = partition  # subclasses read self.root
                self.validate()
                for sample in self._iter_samples():
                    count += 1
                    if partition != base:
                        sample.meta.setdefault("partition", partition.name)
                    yield sample
        finally:
            self.root = base

        if count == 0:
            raise LoaderError(
                f"{self.source.id}: layout '{self.layout}' matched no items under "
                f"{base}. The layout parameters are probably wrong — an empty "
                f"result is never treated as success."
            )

    # ── helpers for subclasses ────────────────────────────────────────────

    def _param(self, name: str, default: object = None, *, required: bool = False) -> object:
        value = self.params.get(name, default)
        if required and value is None:
            raise LoaderError(
                f"{self.source.id}: layout '{self.layout}' requires loader param '{name}'"
            )
        return value

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)


_REGISTRY: dict[str, type[SourceLoader]] = {}


def register_loader(cls: type[SourceLoader]) -> type[SourceLoader]:
    """Register a loader implementation against its declared layout."""
    layout = getattr(cls, "layout", None)
    if not layout:
        raise ValueError(f"{cls.__name__} does not declare a `layout`")
    if layout in _REGISTRY:
        raise ValueError(f"layout '{layout}' already registered by {_REGISTRY[layout].__name__}")
    _REGISTRY[layout] = cls
    return cls


def loader_for(layout: str) -> type[SourceLoader]:
    try:
        return _REGISTRY[layout]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise LoaderError(f"No loader registered for layout '{layout}'. Known: {known}") from None


def registered_layouts() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_loader(
    source: Source,
    root: str | Path,
    *,
    crosswalk: Crosswalk | None = None,
    split: str | None = None,
    partial: bool = False,
) -> SourceLoader:
    """Construct the loader a source declares.

    Raises:
        LoaderError: if the source declares no loader, or an unknown layout.
    """
    if source.loader is None:
        raise LoaderError(
            f"{source.id}: no loader declared. Add a `loader:` block to its registry entry, "
            f"or use layout 'metadata-only' if it is a reference rather than a training set."
        )
    cls = loader_for(source.loader.layout)
    return cls(source, root, crosswalk=crosswalk, split=split, partial=partial)
