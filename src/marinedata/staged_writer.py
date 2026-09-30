"""The D-D staged-tree writer: per-object images, or WebDataset shards + ``index.parquet``.

``layout="objects"`` writes ``images/<stem><suffix>``; ``layout="shards"`` appends to
``images/shard-NNNNN.tar`` (deterministic tar headers: mtime 0, uid/gid 0, no names, so
a re-run produces byte-identical shards) and closes each after it crosses
``shard_bytes``. :mod:`marinedata.ingest_source` picks the layout from the image count
against ``threshold`` (200k, D-D) *before* writing; an ``objects`` run that crosses the
threshold raises :class:`LayoutError` rather than silently producing 200k+ objects.

Files are handed to the uploader as they close (:meth:`drain_closed`) so local temp stays
bounded; the writer remembers each file's sha256/size, so ``CHECKSUMS.sha256`` can be
written at the end even though most bytes have already left the machine.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import re
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any

from .adapters import Decoded, RemoteItem
from .models import SplitGroupRule
from .sample_schema import SampleRow, depth_zone_for

DEFAULT_THRESHOLD = 200_000
DEFAULT_SHARD_BYTES = 1 << 30
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
_FLOAT_FIELDS = ("lat", "lon", "gps_precision_m", "depth_m")
_STR_FIELDS = ("depth_source", "platform", "camera", "meow_realm", "depth_zone", "habitat")


class LayoutError(RuntimeError):
    """An ``objects`` layout run crossed the shard threshold (D-D)."""


class _HashingFile(io.RawIOBase):
    def __init__(self, path: Path) -> None:
        self.fh = path.open("wb")
        self.sha = hashlib.sha256()
        self.pos = 0

    def writable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def write(self, b) -> int:  # type: ignore[override]
        self.fh.write(b)
        self.sha.update(b)
        self.pos += len(b)
        return len(b)

    def close(self) -> None:
        if not self.fh.closed:
            self.fh.close()
        super().close()


class DiskSink:
    """Default sink: files land under ``root`` (disk mode). WP-6h's stream mode swaps in
    a sink that PUTs each file from memory instead — same bytes, same keys."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, rel: str, data: bytes) -> None:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def open_stream(self, rel: str) -> Any:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        return _HashingFile(path)


def _coerce_datetime(value: Any, naive_is_utc: bool) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        try:
            parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=dt.timezone.utc) if naive_is_utc else None
    return parsed.astimezone(dt.timezone.utc)


def _image_facts(data: bytes, suffix: str) -> tuple[int | None, int | None, str]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return im.width, im.height, (im.format or suffix.lstrip(".")).lower()
    except Exception:
        return None, None, suffix.lstrip(".").lower() or "bin"


@dataclass
class WriterConfig:
    source_id: str
    version: str
    license: str
    attribution: str
    fetch_date: dt.date
    layout: str = "objects"
    threshold: int = DEFAULT_THRESHOLD
    shard_bytes: int = DEFAULT_SHARD_BYTES
    defaults: Mapping[str, Any] | None = None
    naive_datetime_is_utc: bool = False
    label_stem_suffix: str = ""
    lineage_root_digest: str | None = None
    split_group: SplitGroupRule | None = None
    """The source's registry rule; ``None`` = unknown here, so rows carry a null
    ``split_group`` (D-AI2). Resolved against the flat D-K stem, partition ``""``."""


class StagedWriter:
    def __init__(self, root: Path, cfg: WriterConfig, sink: Any = None) -> None:
        if cfg.layout not in {"objects", "shards"}:
            raise ValueError(f"layout must be objects|shards, not {cfg.layout!r}")
        self.root, self.cfg = root, cfg
        self.sink = sink if sink is not None else DiskSink(root)
        self.rows: list[SampleRow] = []
        self.files: dict[str, tuple[str, int]] = {}
        self._closed: list[str] = []
        self._stems: set[str] = set()
        self._basename: dict[str, list[int]] = {}
        self._pending_labels: dict[str, list[str]] = {}
        self._inline: list[tuple[str, str, str]] = []
        self._index: list[dict[str, Any]] = []
        self._item_start = 0
        self._shard_no = 0
        self._tar: tarfile.TarFile | None = None
        self._tar_fh: Any = None
        self._tar_rel = ""

    # -- stems ---------------------------------------------------------------------
    def _stem(self, upstream_id: str) -> str:
        base = (
            _UNSAFE.sub(
                "_", PurePosixPath(upstream_id.replace("#", "/")).with_suffix("").as_posix()
            ).strip("_")
            or "item"
        )
        if len(base) > 90:  # USTAR member names cap at 100 bytes
            tag = hashlib.sha1(upstream_id.encode(), usedforsecurity=False).hexdigest()[:10]
            base = f"{base[:79]}_{tag}"
        stem = base
        if stem in self._stems:
            tag = hashlib.sha1(upstream_id.encode(), usedforsecurity=False).hexdigest()[:10]
            stem = f"{base}_{tag}"
        self._stems.add(stem)
        return stem

    # -- file bookkeeping ----------------------------------------------------------
    def _write_file(self, rel: str, data: bytes) -> str:
        self.sink.write(rel, data)
        sha = hashlib.sha256(data).hexdigest()
        self.files[rel] = (sha, len(data))
        self._closed.append(rel)
        return sha

    def drain_closed(self) -> list[str]:
        """Relative paths of files closed since the last call (ready to upload)."""
        out, self._closed = self._closed, []
        return out

    # -- shards --------------------------------------------------------------------
    def _open_shard(self) -> None:
        self._tar_rel = f"images/shard-{self._shard_no:05d}.tar"
        self._tar_fh = self.sink.open_stream(self._tar_rel)
        self._tar = tarfile.open(  # noqa: SIM115 — closed in _close_shard
            fileobj=self._tar_fh, mode="w", format=tarfile.USTAR_FORMAT
        )

    def _close_shard(self) -> None:
        if self._tar is None or self._tar_fh is None:
            return
        self._tar.close()
        self.files[self._tar_rel] = (self._tar_fh.sha.hexdigest(), self._tar_fh.pos)
        self._tar_fh.close()
        self._closed.append(self._tar_rel)
        self._tar, self._tar_fh = None, None
        self._shard_no += 1

    def _add_to_shard(self, member: str, data: bytes, sha: str, stem: str) -> str:
        if self._tar is None:
            self._open_shard()
        assert self._tar is not None and self._tar_fh is not None
        info = tarfile.TarInfo(member)
        info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
        info.uname = info.gname = ""
        header = len(info.tobuf(self._tar.format, self._tar.encoding, self._tar.errors))
        offset = self._tar_fh.pos + header
        self._tar.addfile(info, io.BytesIO(data))
        rel = self._tar_rel
        self._index.append(
            {
                "shard": rel,
                "member": member,
                "stem": stem,
                "sha256": sha,
                "size": len(data),
                "offset": offset,
            }
        )
        if self._tar_fh.pos >= self.cfg.shard_bytes:
            self._close_shard()
        return rel

    # -- samples -------------------------------------------------------------------
    def add(self, item: RemoteItem, decoded: Decoded) -> SampleRow | None:
        if not decoded.data:  # label-only record: pair with images by basename stem
            for name, blob in decoded.label_files.items():
                safe = _UNSAFE.sub("_", str(PurePosixPath(name).with_suffix(""))).strip("_")
                rel = f"labels/files/{safe}{PurePosixPath(name).suffix}"
                self._write_file(rel, blob)
                key = PurePosixPath(name).stem
                if self.cfg.label_stem_suffix and key.endswith(self.cfg.label_stem_suffix):
                    key = key[: -len(self.cfg.label_stem_suffix)]
                self._pending_labels.setdefault(key, []).append(rel)
            return None
        if self.cfg.layout == "objects" and len(self.rows) >= self.cfg.threshold:
            raise LayoutError(
                f"{self.cfg.source_id}: more than {self.cfg.threshold} images in objects "
                "layout; re-run with layout: shards (or set expected_images)"
            )
        stem = self._stem(decoded.upstream_id)
        sha = hashlib.sha256(decoded.data).hexdigest()
        width, height, fmt = _image_facts(decoded.data, decoded.suffix)
        member = f"{stem}{decoded.suffix}"
        if self.cfg.layout == "shards":
            image_path, image_member = self._add_to_shard(member, decoded.data, sha, stem), member
        else:
            image_path, image_member = f"images/{member}", None
            self._write_file(image_path, decoded.data)
        refs = []
        for name, blob in decoded.label_files.items():
            rel = f"labels/files/{stem}{PurePosixPath(name).suffix}"
            self._write_file(rel, blob)
            refs.append(rel)
        for k, v in sorted(decoded.labels.items()):
            self._inline.append((stem, k, v))
        if decoded.labels:
            refs.append("labels/image_labels.parquet")
        vals = {**(self.cfg.defaults or {}), **decoded.fields}
        floats = {
            f: (float(vals[f]) if vals.get(f) not in (None, "") else None) for f in _FLOAT_FIELDS
        }
        strs = {f: (str(vals[f]) if vals.get(f) not in (None, "") else None) for f in _STR_FIELDS}
        if floats["depth_m"] is not None and strs["depth_zone"] is None:
            strs["depth_zone"] = depth_zone_for(floats["depth_m"])
        raw_split = decoded.fields.get("upstream_split") or decoded.split_hint
        upstream_split = str(raw_split) if raw_split not in (None, "") else None
        split_group = (
            self.cfg.split_group.resolve(
                source_id=self.cfg.source_id,
                stem=stem,
                upstream_path=decoded.upstream_id,
                partition="",
            )
            if self.cfg.split_group is not None
            else None
        )
        row = SampleRow(
            sample_id=f"{self.cfg.source_id}/{stem}",
            source_id=self.cfg.source_id,
            source_version=self.cfg.version,
            stem=stem,
            image_path=image_path,
            image_sha256=sha,
            image_bytes=len(decoded.data),
            image_format=fmt,
            license=str(vals.get("license") or self.cfg.license),
            attribution=str(vals.get("attribution") or self.cfg.attribution),
            fetch_date=self.cfg.fetch_date,
            image_member=image_member,
            width=width,
            height=height,
            upstream_id=decoded.upstream_id,
            upstream_url=decoded.upstream_url,
            lineage_root_digest=self.cfg.lineage_root_digest,
            capture_datetime=_coerce_datetime(
                vals.get("capture_datetime"), self.cfg.naive_datetime_is_utc
            ),
            split_hint=decoded.split_hint,
            label_refs=tuple(refs),
            split_group=split_group,
            upstream_split=upstream_split,
            upstream_path=decoded.upstream_id,
            **floats,
            **strs,
        )
        self._basename.setdefault(
            PurePosixPath(decoded.upstream_id.rsplit("#", 1)[-1]).stem, []
        ).append(len(self.rows))
        self.rows.append(row)
        return row

    def finish_item(self, upstream_digest: str | None) -> None:
        """Stamp the finished item's container sha256 onto its rows (immutable replace)."""
        for i in range(self._item_start, len(self.rows)):
            self.rows[i] = replace(self.rows[i], upstream_digest=upstream_digest)
        self._item_start = len(self.rows)

    def finalize(self) -> None:
        """Close the open shard; write index.parquet, labels, and pair label files."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        self._close_shard()
        for key, rels in self._pending_labels.items():
            for i in self._basename.get(key, []):
                r = self.rows[i]
                self.rows[i] = replace(r, label_refs=tuple(sorted({*r.label_refs, *rels})))
        kw = {"compression": "zstd", "write_statistics": False}
        if self._index:
            buf = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(self._index), buf, **kw)
            self.add_bytes("images/index.parquet", buf.getvalue())
        if self._inline:
            buf = io.BytesIO()
            cols = list(zip(*self._inline, strict=True))
            pq.write_table(pa.table({"stem": cols[0], "key": cols[1], "value": cols[2]}), buf, **kw)
            self.add_bytes("labels/image_labels.parquet", buf.getvalue())

    def add_bytes(self, rel: str, data: bytes) -> None:
        """A whole staged file known in memory (index/metadata/INGEST.json/...)."""
        self._write_file(rel, data)

    # -- WP-6h stream-mode resume: checkpoint deltas and restore -----------------
    @property
    def shard_open(self) -> bool:
        return self._tar is not None

    @property
    def shard_no(self) -> int:
        return self._shard_no

    def mark(self) -> tuple[int, int, int, int]:
        return (len(self.rows), len(self.files), len(self._inline), len(self._index))

    def delta(self, mark: tuple[int, int, int, int]) -> dict[str, Any]:
        """Everything staged since ``mark`` (call at an item boundary, no shard open)."""
        rows_n, files_n, inline_n, index_n = mark
        return {
            "rows": self.rows[rows_n:],
            "files": {k: list(v) for k, v in list(self.files.items())[files_n:]},
            "inline": [list(t) for t in self._inline[inline_n:]],
            "index": self._index[index_n:],
            "pending_labels": {k: list(v) for k, v in self._pending_labels.items()},
            "shard_no": self._shard_no,
        }

    def restore(self, rows: list[SampleRow], state: Mapping[str, Any]) -> None:
        """Re-apply one checkpoint delta (the inverse of :meth:`delta`)."""
        for row in rows:
            self._basename.setdefault(
                PurePosixPath(str(row.upstream_id).rsplit("#", 1)[-1]).stem, []
            ).append(len(self.rows))
            self._stems.add(row.stem)
            self.rows.append(row)
        self._item_start = len(self.rows)
        self.files.update({k: (str(v[0]), int(v[1])) for k, v in state["files"].items()})
        self._inline.extend(tuple(t) for t in state["inline"])
        # JSON checkpoints sort keys; index.parquet column order follows the first dict
        order = ("shard", "member", "stem", "sha256", "size", "offset")
        self._index.extend({k: e[k] for k in order} for e in state["index"])
        self._pending_labels = {k: list(v) for k, v in state["pending_labels"].items()}
        self._shard_no = int(state["shard_no"])

    def register(self, path: Path) -> None:
        rel = path.relative_to(self.root).as_posix()
        data = path.read_bytes()
        self.files[rel] = (hashlib.sha256(data).hexdigest(), len(data))
        self._closed.append(rel)
