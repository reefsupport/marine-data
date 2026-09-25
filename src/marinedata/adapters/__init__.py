"""Source adapters: ``enumerate -> fetch -> decode -> samples`` for any open dataset (WP-6).

An adapter knows one *access pattern* (an HF repo, an HTTP/Zenodo archive, an anonymous
bucket listing, a GitHub release), never one dataset. Dataset specifics live in the
spec yaml (``params``), so a new catalog source is a yaml file, not code.

* :meth:`SourceAdapter.enumerate` lists :class:`RemoteItem` s in a deterministic order
  (sorted keys) against a *pinned* upstream version (:meth:`resolve_version`).
* :meth:`fetch` opens one item: tars and single images stream straight through (no temp
  copy); zips and parquet files need random access and are spooled to the temp dir.
* :meth:`decode` turns a fetched item into :class:`Decoded` images (+ inline labels and
  label files); the format dispatch lives in :mod:`marinedata.adapters.decode`.
* :meth:`samples` composes the three; :mod:`marinedata.ingest_source` turns each
  :class:`Decoded` into a :class:`marinedata.sample_schema.SampleRow` via the writer.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, runtime_checkable

from ._http import AccessRefused, ConcatReader, HashingReader, download, open_url

__all__ = [
    "AccessRefused",
    "BaseAdapter",
    "Decoded",
    "Fetched",
    "NoStageableItems",
    "RemoteItem",
    "SourceAdapter",
    "make_adapter",
]

# WP-6d-B: payload kinds that resolve access fine but need a real decoder, not just an
# extension the base filter recognises (see decode.py). ``.json``/``.jsonl`` are
# STREAMABLE (small text, read once); video/rosbag/rar need random access -> SPOOLED
# (fetch() already spools any SPOOLED suffix to a temp file, no adapter change needed).
VIDEO = (".mp4", ".mov", ".avi", ".mkv", ".webm")
ROSBAG = (".bag",)
RAR = (".rar",)
CAPTION_JSON = (".json", ".jsonl")

STREAMABLE = (
    ".tar",
    ".tar.gz",
    ".tgz",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
    *CAPTION_JSON,
)
SPOOLED = (".zip", ".parquet", *RAR, *VIDEO, *ROSBAG)

# ``<name>.tar.gz.aa``/``.ab``/... (D-D multipart-tar, e.g. seamapd21): grouped into one
# synthetic ``.tar.gz`` item by ``_group_multipart`` before the suffix filter runs.
_MULTIPART_RE = re.compile(r"^(.+\.tar\.gz)\.([a-z]{2,3})$", re.IGNORECASE)


def suffix_of(key: str) -> str:
    name = PurePosixPath(key).name.lower()
    for multi in (".tar.gz",):
        if name.endswith(multi):
            return multi
    return PurePosixPath(name).suffix


@dataclass(frozen=True)
class RemoteItem:
    key: str
    """Upstream-relative identifier (repo path, record file key, bucket key)."""
    url: str
    size: int | None = None
    md5: str | None = None
    """Upstream-declared md5 (Zenodo checksum, S3 single-part ETag) — verified on fetch."""
    sha256: str | None = None
    """Upstream-declared sha256 (HF LFS oid, GitHub asset digest) — verified on fetch."""
    parts: tuple[str, ...] = ()
    """Multipart-tar (D-D): ordered part URLs; ``fetch()`` concatenates them with
    :class:`~marinedata.adapters._http.ConcatReader` instead of using ``url`` alone."""


class NoStageableItems(RuntimeError):
    """D-R4 root cause fix: a dry-run that resolves access but enumerates 0 stageable
    items is a decoder gap, never a real "ok" — this makes that structurally impossible
    to miss (non-zero exit with a reason), instead of relying on a hand-maintained table
    of ids already diagnosed by a human (the old ``scripts/spec_dryrun.py`` approach)."""

    def __init__(self, source: str, reason: str, sample_keys: tuple[str, ...] = ()) -> None:
        detail = f" (first upstream keys: {', '.join(sample_keys[:5])})" if sample_keys else ""
        super().__init__(f"{source}: 0 stageable items -> needs_adapter:{reason}{detail}")
        self.reason = reason


def _group_multipart(raw: list[RemoteItem]) -> list[RemoteItem]:
    """Collapse ``<name>.tar.gz.aa``/``.ab``/... runs into one synthetic ``.tar.gz`` item
    per group, parts ordered by their letter suffix. Everything else passes through."""
    groups: dict[str, list[tuple[str, RemoteItem]]] = {}
    passthrough: list[RemoteItem] = []
    for item in raw:
        m = _MULTIPART_RE.match(item.key)
        if m:
            groups.setdefault(m.group(1), []).append((m.group(2).lower(), item))
        else:
            passthrough.append(item)
    merged = list(passthrough)
    for base, parts in groups.items():
        ordered = [p for _, p in sorted(parts, key=lambda t: t[0])]
        merged.append(
            RemoteItem(
                key=base,
                url=ordered[0].url,
                size=sum(p.size or 0 for p in ordered) or None,
                parts=tuple(p.url for p in ordered),
            )
        )
    return merged


def _classify_empty(raw: list[RemoteItem]) -> str:
    """Best-effort reason for a 0-item enumeration, from the RAW (pre-filter) upstream
    listing — used only once every known kind (video/rar/rosbag/json/multipart-tar) has
    already been given a real decoder, so this is the fallback for the next unknown one."""
    if not raw:
        return "unknown-empty"
    if any(_MULTIPART_RE.match(i.key) for i in raw):
        return "multipart-tar"
    suffixes = {suffix_of(i.key) for i in raw}
    if suffixes & set(VIDEO):
        return "video"
    if suffixes & set(RAR):
        return "rar"
    if suffixes & set(ROSBAG):
        return "rosbag"
    if suffixes and suffixes <= set(CAPTION_JSON):
        return "json-captions"
    return "unknown-container"


@dataclass(frozen=True)
class Decoded:
    upstream_id: str
    data: bytes
    suffix: str
    upstream_url: str | None = None
    split_hint: str | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    """Per-sample schema fields stated upstream (lat, lon, depth_m, capture_datetime …)."""
    labels: Mapping[str, str] = field(default_factory=dict)
    """Inline labels (key -> value), written to ``labels/image_labels.parquet``."""
    label_files: Mapping[str, bytes] = field(default_factory=dict)
    """Label files (name -> bytes), written under ``labels/<name>``."""


@dataclass
class Fetched:
    item: RemoteItem
    path: Path | None = None
    stream: HashingReader | None = None
    sha256: str | None = None
    size: int = 0

    def close(self) -> None:
        """Drain + close a stream so the digest covers every upstream byte."""
        if self.stream is not None:
            self.stream.drain()
            self.stream.raw.close()
            self.sha256 = self.stream.sha.hexdigest()
            self.size = self.stream.nbytes
            _check_declared(self.item, self.sha256, self.stream.md5.hexdigest(), self.size)
            self.stream = None


def _check_declared(item: RemoteItem, sha: str, md5: str, size: int) -> None:
    if item.sha256 and item.sha256 != sha:
        raise ValueError(f"{item.key}: sha256 {sha} != declared {item.sha256}")
    if item.md5 and item.md5 != md5:
        raise ValueError(f"{item.key}: md5 {md5} != declared {item.md5}")
    if item.size is not None and item.size != size:
        raise ValueError(f"{item.key}: {size} bytes != declared {item.size}")


@runtime_checkable
class SourceAdapter(Protocol):
    name: str

    def resolve_version(self) -> str: ...

    def enumerate(self) -> Iterator[RemoteItem]: ...

    def is_label(self, key: str) -> bool:
        """A loose file matching ``label_patterns`` (e.g. YOLO ``labels/*.txt``)."""
        return any(fnmatch.fnmatch(key, g) for g in self.params.get("label_patterns") or [])

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched: ...

    def decode(self, fetched: Fetched) -> Iterator[Decoded]: ...

    def samples(self, tmp_dir: Path) -> Iterator[tuple[RemoteItem, Fetched, Decoded]]: ...


class BaseAdapter:
    """Shared fetch/decode/samples; subclasses implement ``resolve_version``/``enumerate``.

    Common ``params``: ``include``/``exclude`` (fnmatch globs on the item key),
    ``format`` (``auto``|``imagefolder``|``webdataset``|``parquet``), ``label_patterns``
    (archive member globs staged as label files), ``columns`` (schema field -> parquet
    column), ``label_columns`` (parquet columns kept as inline labels),
    ``image_column``, ``max_items``.
    """

    name = "base"

    def __init__(self, params: Mapping[str, Any]) -> None:
        self.params = dict(params)

    def resolve_version(self) -> str:  # pragma: no cover - abstract
        raise NotImplementedError

    def list_items(self) -> Iterator[RemoteItem]:  # pragma: no cover - abstract
        raise NotImplementedError

    def enumerate(self) -> Iterator[RemoteItem]:
        inc = list(self.params.get("include") or ["*"])
        exc = list(self.params.get("exclude") or [])
        cap = self.params.get("max_items")
        labels = list(self.params.get("label_patterns") or [])
        raw = _group_multipart(list(self.list_items()))
        items = sorted(
            (
                i
                for i in raw
                if any(fnmatch.fnmatch(i.key, g) for g in inc)
                and not any(fnmatch.fnmatch(i.key, g) for g in exc)
                and (
                    suffix_of(i.key) in STREAMABLE + SPOOLED
                    or any(fnmatch.fnmatch(i.key, g) for g in labels)  # loose label files
                )
            ),
            key=lambda i: i.key,
        )
        if not items:
            # D-R4 root cause: never a silent/false "ok" — see NoStageableItems.
            raise NoStageableItems(self.name, _classify_empty(raw), tuple(i.key for i in raw))
        yield from items[: int(cap)] if cap else items

    def is_label(self, key: str) -> bool:
        """A loose file matching ``label_patterns`` (e.g. YOLO ``labels/*.txt``)."""
        return any(fnmatch.fnmatch(key, g) for g in self.params.get("label_patterns") or [])

    def fetch(self, item: RemoteItem, tmp_dir: Path) -> Fetched:
        if item.parts:
            return Fetched(item, stream=HashingReader(ConcatReader(item.parts)))
        if suffix_of(item.key) in SPOOLED and not self.is_label(item.key):
            dest = tmp_dir / "fetch" / PurePosixPath(item.key).name
            sha, md5, size = download(item.url, dest)
            _check_declared(item, sha, md5, size)
            return Fetched(item, path=dest, sha256=sha, size=size)
        return Fetched(item, stream=HashingReader(open_url(item.url)))

    def decode(self, fetched: Fetched) -> Iterator[Decoded]:
        from .decode import decode_item

        yield from decode_item(fetched, self.params)

    def samples(self, tmp_dir: Path) -> Iterator[tuple[RemoteItem, Fetched, Decoded]]:
        for item in self.enumerate():
            fetched = self.fetch(item, tmp_dir)
            try:
                for decoded in self.decode(fetched):
                    yield item, fetched, decoded
                fetched.close()
            finally:
                if fetched.path is not None:
                    fetched.path.unlink(missing_ok=True)


def make_adapter(name: str, params: Mapping[str, Any]) -> BaseAdapter:
    from .bucket import BucketAdapter
    from .fathomnet import FathomNetAdapter
    from .figshare import FigshareAdapter
    from .gdrive import GDriveAdapter
    from .girder import GirderAdapter
    from .github import GitHubAdapter
    from .hf import HFAdapter
    from .http_index import HttpIndexAdapter
    from .inat import INatOpenDataAdapter
    from .pangaea import PangaeaAdapter
    from .pawsey import FrdrHttpsAdapter, PawseyPortalAdapter
    from .seafile import SeafileAdapter
    from .treeoflife import HFMemberFilterAdapter
    from .web import HttpAdapter
    from .wikimedia import CommonsAdapter

    table: dict[str, type[BaseAdapter]] = {
        "hf": HFAdapter,
        "http": HttpAdapter,
        "zenodo": HttpAdapter,
        "bucket": BucketAdapter,
        "github": GitHubAdapter,
        "fathomnet": FathomNetAdapter,
        "figshare": FigshareAdapter,
        "pangaea": PangaeaAdapter,
        "http-index": HttpIndexAdapter,
        "gdrive-public": GDriveAdapter,
        "seafile-share": SeafileAdapter,
        "girder": GirderAdapter,
        "pawsey-portal": PawseyPortalAdapter,
        "frdr-https": FrdrHttpsAdapter,
        "inat-open-data": INatOpenDataAdapter,
        "hf-member-filter": HFMemberFilterAdapter,
        "commons-api": CommonsAdapter,
    }
    if name not in table:
        raise KeyError(f"unknown adapter {name!r}; known: {sorted(table)}")
    return table[name](params)
