"""Task annotations built from the staged trees a ``--v11`` release build is given.

``release build --v11`` read task annotations only from
``<tasklabels_root>/_annotations/<table>/<src>/*.parquet`` (legacy ``_tasklabels/`` as the
fallback) and never from the staged trees it was handed. A build whose tasklabels root was the
image's ``/app/data`` therefore wrote ``points`` / ``boxes`` / ``instances`` with zero rows.

:func:`staged_overlay` closes that gap with a temporary directory shaped like a tasklabels root.
Everything of the resolved root stays reachable (``_tasklabels/``, ``_labelquality/`` and every
annotation file already staged there, which wins over a producer; the root itself is never
written). Each registered producer (:data:`PRODUCERS`, one per ``(table, source)``) whose source
is among the admitted staged roots writes
``<overlay>/_annotations/<table>/<source>/<version>.parquet``. ``build_all_configs`` then takes
the overlay as its ``base_dir``. A producer that yields no rows raises: the build must not go on
with an empty table.

Producers read a tree through :class:`StagedFetch`, a local stand-in for
:func:`marinedata.task_layers.s3_keyed.fetch_small`: the pinned key
``sources/<id>/<version>/<rel>`` is ``<root>/<rel>``, and ``CHECKSUMS.sha256`` is built from the
files on disk (the cached trees carry none). The digest is the build's memoised one, so the image
bytes the metadata spine already hashed are not hashed again and task rows key on the same sha256
as the spine. No network, no image is ever read.

Adding a producer (part B: ``boxes`` through ``staged_boxes``, ``instances`` through
``staged_instance_rows``): write ``run(ctx) -> ProducedLayer`` that writes its table with the
table's own writer (``write_boxes`` / ``write_masks``) into ``ctx.annotations_root``, and pass a
:class:`LayerProducer` to :func:`register_producer`.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from ..annotation_schema import annotation_path
from ..checksums import CHECKSUM_FILE, _relative_paths, file_digest, iter_lines
from ..registry import Registry
from .s3_keyed import IMAGE_PREFIX, FetchFailed, ImageFetchRefused

ANNOTATIONS_DIR = "_annotations"
SEAVIEW = "seaview-survey-imagery"
_PENDING = ".pending.parquet"
_UNREPRESENTABLE = ("\n", "\r", "\\")  # cannot be written into a sha256sum manifest


class StagedLayerError(RuntimeError):
    """A staged tree could not produce its table; the v1.1.0 build must not go on without it."""


class StagedFetch:
    """``fetch(key) -> bytes`` over admitted staged roots, for the S3-keyed producers.

    ``sources/<id>/<version>/<rel>`` is ``<root of id>/<rel>`` (the staged root *is* the pinned
    version; the version segment is the producer's pin and is not compared). ``CHECKSUMS.sha256``
    is rendered from the files on disk with ``digest``. Like ``fetch_small`` it refuses
    ``images/`` keys, and a key it cannot serve fails as a ``FetchFailed`` 404.
    """

    def __init__(
        self, roots: Mapping[str, str | Path], digest: Callable[[Path], str] = file_digest
    ) -> None:
        self.roots = {source_id: Path(root) for source_id, root in roots.items()}
        self._digest = digest
        self._sums: dict[str, dict[str, str]] = {}

    def checksums(self, source_id: str) -> dict[str, str]:
        """``{posix rel: sha256}`` of every file under the root of ``source_id`` (computed once)."""
        if source_id not in self._sums:
            self._sums[source_id] = {
                rel: self._digest(path)
                for rel, path in _relative_paths(self.roots[source_id])
                if not any(bad in rel for bad in _UNREPRESENTABLE)
            }
        return self._sums[source_id]

    def __call__(self, key: str) -> bytes:
        parts = key.split("/", 3)
        if len(parts) != 4 or parts[0] != "sources" or parts[1] not in self.roots:
            raise FetchFailed(key, 1, "404")
        source_id, rel = parts[1], parts[3]
        if f"/{IMAGE_PREFIX}" in f"/{rel}":
            raise ImageFetchRefused(key)
        if rel == CHECKSUM_FILE:
            return "".join(iter_lines(self.checksums(source_id))).encode()
        pure = PurePosixPath(rel)
        path = self.roots[source_id].joinpath(*pure.parts)
        if pure.is_absolute() or ".." in pure.parts or not path.is_file():
            raise FetchFailed(key, 1, "404")
        return path.read_bytes()


@dataclass(frozen=True)
class ProducerContext:
    """What a producer's ``run`` is given."""

    registry: Registry
    source_id: str
    version: str
    root: Path
    """The admitted staged root of ``source_id``."""
    fetch: StagedFetch
    annotations_root: Path
    """The overlay root: write ``annotation_path(annotations_root, table, source_id, version)``."""
    scratch: Path
    """A private writable directory for side outputs; deleted with the overlay."""


@dataclass(frozen=True)
class ProducedLayer:
    """A producer's own tally (the row count that counts is read back from the parquet)."""

    rows: int
    images: int = 0
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LayerProducer:
    """The producer of one annotation ``table`` for one staged ``source_id``."""

    table: str
    source_id: str
    version: Callable[[Registry], str]
    """The staged version the producer is pinned to (names the parquet file)."""
    run: Callable[[ProducerContext], ProducedLayer]


@dataclass(frozen=True)
class StagedLayer:
    """One table of one source in the overlay."""

    table: str
    source_id: str
    version: str
    rows: int
    images: int = 0
    reused: bool = False
    """True when the base root already held the table for this source (it wins)."""
    detail: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StagedOverlay:
    root: Path
    """Pass as ``base_dir`` to ``build_all_configs``."""
    layers: tuple[StagedLayer, ...]


PRODUCERS: dict[tuple[str, str], LayerProducer] = {}
"""Registered producers by ``(table, source_id)``."""


def register_producer(producer: LayerProducer) -> LayerProducer:
    key = (producer.table, producer.source_id)
    if key in PRODUCERS:
        raise ValueError(f"{key}: a staged-annotation producer is already registered")
    PRODUCERS[key] = producer
    return producer


def _head(path: Path, n: int) -> bytes:
    with path.open("rb") as handle:
        return handle.read(n)


def _seaview_version(registry: Registry) -> str:
    from ..metadata_norm.resolvers.run import default_version

    return default_version(SEAVIEW, registry.root)


def _seaview_points(ctx: ProducerContext) -> ProducedLayer:
    """Seaview ``points`` from the staged tree: the code path of ``marinedata tasks build-points``
    over a local fetch. WoRMS is off (names come from the shipped cache only), the cache is a
    scratch copy (the registry is never written), the master CSV head is read from disk."""
    from .seaview_points import run_build_points

    cache = ctx.scratch / "worms-cache.json"
    shipped = ctx.registry.root / "taxonomy" / "worms-cache.json"
    if shipped.is_file():
        shutil.copyfile(shipped, cache)
    summary = run_build_points(
        ctx.scratch / "points",
        version=ctx.version,
        registry_root=ctx.registry.root,
        worms_cache_path=cache,
        worms=False,
        annotations_root=ctx.annotations_root,
        fetch=ctx.fetch,
        range_fetch=lambda rel, n: _head(ctx.root / rel, n),
    )
    totals = summary["totals"]
    detail = {"totals": totals, "crosswalked_pct": summary.get("crosswalked_pct")}
    return ProducedLayer(totals["points"], totals["images_with_points"], detail)


register_producer(LayerProducer("points", SEAVIEW, _seaview_version, _seaview_points))


def _mirror(base: Path, overlay: Path) -> None:
    """``overlay`` reaches everything in ``base`` and can be written without touching it: the
    top-level entries are symlinked, ``_annotations/`` is a real tree of file symlinks."""
    overlay.mkdir(parents=True)
    for entry in sorted(base.iterdir()):
        if entry.name != ANNOTATIONS_DIR:
            (overlay / entry.name).symlink_to(entry.resolve())
    (overlay / ANNOTATIONS_DIR).mkdir()
    source = base / ANNOTATIONS_DIR
    walk = os.walk(source, followlinks=True) if source.is_dir() else ()
    for dirpath, _dirs, files in walk:
        target = overlay / ANNOTATIONS_DIR / Path(dirpath).relative_to(source)
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            (target / name).symlink_to((Path(dirpath) / name).resolve())


def _layer_files(root: Path, table: str, source_id: str) -> list[Path]:
    directory = root / ANNOTATIONS_DIR / table / source_id
    return sorted(p for p in directory.glob("*.parquet") if not p.name.endswith(_PENDING))


def _rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.read_metadata(path).num_rows if path.is_file() else 0


def _run_producer(
    producer: LayerProducer,
    registry: Registry,
    root: Path,
    fetch: StagedFetch,
    overlay: Path,
    scratch_root: Path,
) -> StagedLayer:
    where = f"{producer.table}/{producer.source_id}"
    version = producer.version(registry)
    if existing := _layer_files(overlay, producer.table, producer.source_id):
        return StagedLayer(
            producer.table, producer.source_id, version, _rows(existing[-1]), 0, True
        )
    scratch = scratch_root / f"{producer.table}-{producer.source_id}"
    scratch.mkdir(parents=True)
    ctx = ProducerContext(registry, producer.source_id, version, root, fetch, overlay, scratch)
    try:
        produced = producer.run(ctx)
    except Exception as exc:
        raise StagedLayerError(f"{where} ({root}): {type(exc).__name__}: {exc}") from exc
    path = annotation_path(overlay, producer.table, producer.source_id, version)
    rows = _rows(path)
    if rows == 0:
        raise StagedLayerError(
            f"{where}: the staged tree {root} yielded 0 rows (expected {path.name}); "
            "refusing to build the v1.1.0 table empty"
        )
    return StagedLayer(
        producer.table, producer.source_id, version, rows, produced.images, False, produced.detail
    )


@contextlib.contextmanager
def staged_overlay(
    registry: Registry,
    base_dir: str | Path,
    roots: Mapping[str, str | Path],
    *,
    digest: Callable[[Path], str] = file_digest,
    producers: Sequence[LayerProducer] | None = None,
    tmp_dir: str | Path | None = None,
) -> Iterator[StagedOverlay]:
    """A tasklabels root = ``base_dir`` + the annotations the admitted staged ``roots`` yield.

    ``digest`` is the build's memoised digest. A root with no registered producer is ignored; a
    table the base already holds for a source is kept (``reused``). The directory is deleted on
    exit, so build the configs inside the ``with``.
    """
    base = Path(base_dir)
    admitted = {source_id: Path(root) for source_id, root in roots.items()}
    chosen = list(PRODUCERS.values()) if producers is None else list(producers)
    with tempfile.TemporaryDirectory(prefix="marinedata-staged-", dir=tmp_dir) as tmp:
        overlay = Path(tmp) / "root"
        _mirror(base, overlay)
        fetch = StagedFetch(admitted, digest)
        layers = tuple(
            _run_producer(p, registry, admitted[p.source_id], fetch, overlay, Path(tmp) / "scratch")
            for p in sorted(chosen, key=lambda p: (p.table, p.source_id))
            if p.source_id in admitted
        )
        yield StagedOverlay(overlay, layers)


__all__ = [  # noqa: RUF022
    "ANNOTATIONS_DIR",
    "LayerProducer",
    "PRODUCERS",
    "ProducedLayer",
    "ProducerContext",
    "SEAVIEW",
    "StagedFetch",
    "StagedLayer",
    "StagedLayerError",
    "StagedOverlay",
    "register_producer",
    "staged_overlay",
]
