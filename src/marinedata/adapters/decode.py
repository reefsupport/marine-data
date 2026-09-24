"""Format dispatch for :meth:`BaseAdapter.decode`: image, tar/WebDataset, zip, parquet.

Bounded memory throughout: tar members are read one at a time off the stream, zip
members one at a time off the spooled file, parquet in 64-row batches. Only one image's
bytes (plus its sidecars) are ever held at once.
"""

from __future__ import annotations

import fnmatch
import json
import tarfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import PurePosixPath
from typing import Any

from ..sample_schema import FIELD_NAMES, normalise_split
from . import Decoded, Fetched, suffix_of

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
_SIDE_FIELDS = frozenset(FIELD_NAMES) & {
    "capture_datetime",
    "lat",
    "lon",
    "gps_precision_m",
    "depth_m",
    "depth_source",
    "platform",
    "camera",
    "meow_realm",
    "depth_zone",
    "habitat",
}


def split_from_path(key: str) -> str | None:
    """``train/x.jpg``, ``data/validation-0001.parquet`` -> ``train``/``val``."""
    parts = PurePosixPath(key).parts
    for part in (*parts[:-1], parts[-1].split("-")[0].split("_")[0], PurePosixPath(key).stem):
        if (hint := normalise_split(part)) is not None:
            return hint
    return None


def _wds_key(name: str) -> str:
    """WebDataset key: directory + basename up to the FIRST dot."""
    p = PurePosixPath(name)
    return str(p.parent / p.name.split(".", 1)[0])


def _sidecar(data: bytes, suffix: str) -> tuple[dict[str, Any], dict[str, str]]:
    fields: dict[str, Any] = {}
    labels: dict[str, str] = {}
    if suffix == ".json":
        try:
            obj = json.loads(data)
        except ValueError:
            return fields, labels
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _SIDE_FIELDS:
                    fields[k] = v
                elif isinstance(v, (str, int, float, bool)):
                    labels[k] = str(v)
    elif suffix in {".cls", ".txt", ".label"}:
        labels[suffix.lstrip(".")] = data.decode("utf-8", "replace").strip()
    return fields, labels


def _group(
    members: Iterable[tuple[str, Callable[[], bytes]]],
    fetched: Fetched,
    params: Mapping[str, Any],
) -> Iterator[Decoded]:
    """Group consecutive same-key members (WebDataset), emit one Decoded per image.

    Members matching ``label_patterns`` become label-only records (``data == b""``);
    the writer pairs them with images by basename stem.
    """
    label_globs = list(params.get("label_patterns") or [])
    imagefolder = params.get("format") == "imagefolder"
    current: str | None = None
    bucket: list[tuple[str, bytes]] = []

    def flush() -> Iterator[Decoded]:
        images = [(n, b) for n, b in bucket if suffix_of(n) in IMAGE_SUFFIXES]
        side_f: dict[str, Any] = {}
        side_l: dict[str, str] = {}
        for n, b in bucket:
            f, lab = _sidecar(b, suffix_of(n))
            side_f.update(f)
            side_l.update(lab)
        for n, b in images[:1]:
            labels = dict(side_l)
            parent = PurePosixPath(n).parent.name
            if imagefolder and parent and normalise_split(parent) is None:
                labels.setdefault("label", parent)
            yield Decoded(
                upstream_id=f"{fetched.item.key}#{n}" if n != fetched.item.key else n,
                data=b,
                suffix=suffix_of(n),
                split_hint=split_from_path(n),
                fields=side_f,
                labels=labels,
            )

    for name, read in members:
        if any(fnmatch.fnmatch(name, g) for g in label_globs):
            yield Decoded(
                upstream_id=f"{fetched.item.key}#{name}",
                data=b"",
                suffix=suffix_of(name),
                label_files={name: read()},
            )
            continue
        key = _wds_key(name)
        if key != current:
            yield from flush()
            bucket, current = [], key
        if suffix_of(name) in IMAGE_SUFFIXES | {".json", ".cls", ".txt", ".label"}:
            bucket.append((name, read()))
    yield from flush()


def _tar_members(fetched: Fetched) -> Iterator[tuple[str, Callable[[], bytes]]]:
    assert fetched.stream is not None
    with tarfile.open(fileobj=fetched.stream, mode="r|*") as tar:  # type: ignore[call-overload]
        for m in tar:
            if not m.isfile():
                continue
            handle = tar.extractfile(m)
            data = handle.read() if handle else b""
            yield m.name, (lambda d=data: d)


def _zip_members(fetched: Fetched) -> Iterator[tuple[str, Callable[[], bytes]]]:
    assert fetched.path is not None
    with zipfile.ZipFile(fetched.path) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir() or "__MACOSX" in info.filename:
                continue
            yield info.filename, (lambda i=info: zf.read(i))


def _class_names(pf) -> dict[str, list[str]]:
    """ClassLabel names from HF's parquet ``huggingface`` schema metadata, if present."""
    meta = (pf.schema_arrow.metadata or {}).get(b"huggingface")
    if not meta:
        return {}
    try:
        feats = json.loads(meta).get("info", {}).get("features", {})
    except ValueError:
        return {}
    return {
        k: v["names"]
        for k, v in feats.items()
        if isinstance(v, dict) and v.get("_type") == "ClassLabel" and "names" in v
    }


def _parquet(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    import pyarrow.parquet as pq

    assert fetched.path is not None
    pf = pq.ParquetFile(fetched.path)
    names = [f.name for f in pf.schema_arrow]
    image_col = params.get("image_column") or next(
        (n for n in names if n in {"image", "img", "jpg", "png"}), None
    )
    if image_col is None:
        raise ValueError(f"{fetched.item.key}: no image column in {names}; set image_column")
    columns: Mapping[str, str] = params.get("columns") or {}
    label_cols = list(params.get("label_columns") or [])
    classes = _class_names(pf)
    split = split_from_path(fetched.item.key)
    index = 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            cell = row[image_col]
            data, inner = (
                (cell.get("bytes"), cell.get("path")) if isinstance(cell, dict) else (cell, None)
            )
            if not data:
                index += 1
                continue
            suffix = (
                suffix_of(inner) if inner and suffix_of(inner) in IMAGE_SUFFIXES else _sniff(data)
            )
            labels = {}
            for c in label_cols:
                v = row.get(c)
                if v is None:
                    continue
                labels[c] = (
                    classes[c][v]
                    if c in classes and isinstance(v, int) and 0 <= v < len(classes[c])
                    else str(v)
                )
            yield Decoded(
                upstream_id=f"{fetched.item.key}#{index}",
                data=data,
                suffix=suffix,
                split_hint=split,
                labels=labels,
                fields={f: row.get(c) for f, c in columns.items() if row.get(c) is not None},
            )
            index += 1


def _sniff(data: bytes) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return ".tif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".bin"


def decode_item(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    key = fetched.item.key
    suffix = suffix_of(key)
    if suffix in IMAGE_SUFFIXES:
        assert fetched.stream is not None
        members: Iterable[tuple[str, Callable[[], bytes]]] = [
            (key, lambda: fetched.stream.read())  # type: ignore[union-attr]
        ]
        for d in _group(members, fetched, params):
            yield Decoded(
                d.upstream_id,
                d.data,
                d.suffix,
                fetched.item.url,
                d.split_hint,
                d.fields,
                d.labels,
                d.label_files,
            )
    elif suffix in {".tar", ".tar.gz", ".tgz"}:
        yield from _group(_tar_members(fetched), fetched, params)
    elif suffix == ".zip":
        yield from _group(_zip_members(fetched), fetched, params)
    elif suffix == ".parquet":
        yield from _parquet(fetched, params)
    else:
        raise ValueError(f"{key}: no decoder for {suffix!r}")
