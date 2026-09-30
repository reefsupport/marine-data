"""Format dispatch for :meth:`BaseAdapter.decode`: image, tar/WebDataset, zip, parquet.

Bounded memory throughout: tar members are read one at a time off the stream, zip
members one at a time off the spooled file, parquet in 64-row batches. Only one image's
bytes (plus its sidecars) are ever held at once.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ..sample_schema import FIELD_NAMES, normalise_split
from . import RAR, ROSBAG, VIDEO, Decoded, Fetched, suffix_of
from .zipread import read_member

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"})
CAPTION_JSON_SUFFIXES = frozenset({".json", ".jsonl"})
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
            yield info.filename, (lambda i=info: read_member(zf, i))


def _rar_members(fetched: Fetched) -> Iterator[tuple[str, Callable[[], bytes]]]:
    """WP-6d-B: extract member-by-member via ``bsdtar`` (libarchive), never a bulk
    unpack — one member's bytes are ever held at once, same as ``_zip_members``.
    libarchive detects the RAR format from content, not the ``.rar`` extension."""
    assert fetched.path is not None
    listing = subprocess.run(
        ["bsdtar", "-tf", str(fetched.path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    names = sorted(n for n in listing if n and not n.endswith("/") and "__MACOSX" not in n)

    def _read(path, name: str) -> bytes:
        return subprocess.run(
            ["bsdtar", "-xOf", str(path), name],
            capture_output=True,
            check=True,
        ).stdout

    for name in names:
        yield name, (lambda n=name: _read(fetched.path, n))


def _dhash(data: bytes, hash_size: int = 8) -> int | None:
    """8x8 difference hash (Hamming-comparable) — near-duplicate consecutive video/rosbag
    frames are dropped when two hashes differ by <= 6 bits. Returns ``None`` if the frame
    can't be decoded as an image (kept rather than dropped)."""
    try:
        import io as _io

        from PIL import Image

        img = Image.open(_io.BytesIO(data)).convert("L").resize((hash_size + 1, hash_size))
        pixels = list(img.getdata())
    except Exception:
        return None
    bits = 0
    for row in range(hash_size):
        off = row * (hash_size + 1)
        for col in range(hash_size):
            bits = (bits << 1) | int(pixels[off + col] > pixels[off + col + 1])
    return bits


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    from imageio_ffmpeg import get_ffmpeg_exe  # optional decoders-extra fallback

    return get_ffmpeg_exe()


def _video_frames(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    """WP-6d-B: sample at ``video_fps`` (default 1), drop near-duplicate consecutive
    frames (dHash <= 6). ``video_id``/``frame_ts`` go in ``labels`` (D-D: parquet-
    compatible sample metadata); ``upstream_id`` keeps the existing ``key#member``
    convention so a registry ``split_group`` pattern can group frames by video (leak-safe)
    the same way it already groups tar/zip members."""
    assert fetched.path is not None
    fps = float(params.get("video_fps", 1))
    video_id = PurePosixPath(fetched.item.key).stem
    with tempfile.TemporaryDirectory(prefix="vidframes-") as td:
        pattern = str(Path(td) / "f-%08d.jpg")
        subprocess.run(
            [
                _ffmpeg_bin(),
                "-nostdin",
                "-loglevel",
                "error",
                "-i",
                str(fetched.path),
                "-vf",
                f"fps={fps}",
                "-qscale:v",
                "2",
                pattern,
            ],
            check=True,
        )
        last_hash: int | None = None
        for i, frame_path in enumerate(sorted(Path(td).glob("f-*.jpg"))):
            data = frame_path.read_bytes()
            h = _dhash(data)
            if h is not None and last_hash is not None and _hamming(h, last_hash) <= 6:
                continue
            last_hash = h if h is not None else last_hash
            yield Decoded(
                upstream_id=f"{fetched.item.key}#frame_{i:06d}",
                data=data,
                suffix=".jpg",
                upstream_url=fetched.item.url,
                labels={"video_id": video_id, "frame_ts": f"{i / fps:.3f}"},
            )


def _rosbag_frames(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    """WP-6d-B: sensor_msgs/Image + CompressedImage topics via the pure-Python
    ``rosbags`` reader (no ROS install needed), same 1 fps + dHash sampling/grouping as
    video. ``bag_id``/``topic``/``stamp`` go in ``labels``; ``upstream_id`` groups by
    ``key#topic`` the same leak-safe way ``_video_frames`` groups by video."""
    from rosbags.highlevel import AnyReader

    assert fetched.path is not None
    fps = float(params.get("video_fps", 1))
    bag_id = PurePosixPath(fetched.item.key).stem
    min_gap_ns = int(1e9 / fps) if fps > 0 else 0
    topics = params.get("rosbag_topics")
    with AnyReader([fetched.path]) as reader:
        connections = [
            c
            for c in reader.connections
            if c.msgtype in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
            and (not topics or c.topic in topics)
        ]
        last_ts: dict[str, int] = {}
        last_hash: dict[str, int] = {}
        for i, (connection, timestamp, rawdata) in enumerate(reader.messages(connections)):
            if timestamp - last_ts.get(connection.topic, -min_gap_ns) < min_gap_ns:
                continue
            msg = reader.deserialize(rawdata, connection.msgtype)
            data, suffix = _rosbag_image_bytes(msg, connection.msgtype)
            if data is None:
                continue
            h = _dhash(data)
            prev = last_hash.get(connection.topic)
            if h is not None and prev is not None and _hamming(h, prev) <= 6:
                continue
            last_ts[connection.topic] = timestamp
            if h is not None:
                last_hash[connection.topic] = h
            yield Decoded(
                upstream_id=f"{fetched.item.key}#{connection.topic}#{i:08d}",
                data=data,
                suffix=suffix,
                upstream_url=fetched.item.url,
                labels={
                    "bag_id": bag_id,
                    "topic": connection.topic,
                    "stamp": f"{timestamp / 1e9:.6f}",
                },
            )


def _rosbag_image_bytes(msg: Any, msgtype: str) -> tuple[bytes | None, str]:
    if msgtype == "sensor_msgs/msg/CompressedImage":
        fmt = str(getattr(msg, "format", "jpeg")).split(";")[0].strip().lower()
        return bytes(msg.data), f".{fmt}" if fmt else ".jpg"
    encoding = str(getattr(msg, "encoding", "")).lower()
    try:
        import io as _io

        from PIL import Image

        if encoding in {"rgb8", "bgr8"}:
            img = Image.frombytes("RGB", (msg.width, msg.height), bytes(msg.data))
            if encoding == "bgr8":
                b, g, r = img.split()
                img = Image.merge("RGB", (r, g, b))
        elif encoding in {"mono8", "8uc1"}:
            img = Image.frombytes("L", (msg.width, msg.height), bytes(msg.data))
        else:
            return None, ".jpg"  # unsupported raw encoding: skip, don't guess
        buf = _io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue(), ".jpg"
    except Exception:
        return None, ".jpg"


def _caption_json(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    """WP-6d-B: caption/VQA JSON (json or jsonl) staged as label-only records under
    ``labels/`` (D-D). Each record's image reference resolves against
    ``params.image_index`` (an already-staged upstream key/sha -> url map built by the
    caller) or, failing that, is fetched directly if it is an absolute URL. Unresolvable
    references are counted (``unresolved`` label) rather than silently dropped."""
    assert fetched.stream is not None
    raw = fetched.stream.read()
    text = raw.decode("utf-8", "replace")
    if suffix_of(fetched.item.key) == ".jsonl":
        records: list[Any] = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        obj = json.loads(text) if text.strip() else []
        # WP-BENCH-fix3 B: a COCO-detection file (``{images, annotations, categories}``,
        # e.g. FathomNet's coco_*.json) has no ``data``/``records`` key, so it used to
        # fall through to ``[obj]`` — the WHOLE dict treated as one caption record with
        # no ``image_field``, silently producing one always-unresolved (empty-bytes)
        # Decoded per file. ``images`` is COCO's own per-sample list; try it too.
        records = (
            obj
            if isinstance(obj, list)
            else obj.get("data") or obj.get("records") or obj.get("images") or [obj]
        )

    image_index: Mapping[str, str] = params.get("image_index") or {}
    image_field = params.get("caption_image_field", "image")
    from . import open_url as _open_url

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue
        ref = rec.get(image_field)
        labels = {k: str(v) for k, v in rec.items() if k != image_field and v is not None}
        data = b""
        resolved = False
        url = None
        if isinstance(ref, str):
            url = image_index.get(ref)
            if url is None and ref.startswith(("http://", "https://")):
                url = ref
        if url:
            try:
                with _open_url(url) as resp:
                    data = resp.read()
                resolved = True
            except Exception:
                resolved = False
        labels["unresolved"] = "false" if resolved else "true"
        split_val = rec.get("split")
        yield Decoded(
            upstream_id=f"{fetched.item.key}#{i}",
            data=data,
            suffix=_sniff(data) if resolved else ".bin",
            upstream_url=fetched.item.url,
            split_hint=normalise_split(str(split_val)) if isinstance(split_val, str) else None,
            labels=labels,
            label_files={} if resolved else {f"unresolved-{i}.json": json.dumps(rec).encode()},
        )


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


def _image_columns(names: list[str], params: Mapping[str, Any]) -> list[str]:
    """``image_column`` may name one column or a list (BENCH-fix3 A: a paired-image
    benchmark like ``euvp`` — ``input_image``/``edited_image`` — must hash both; both
    are benchmark eval images, so decontamination has to see either one as contact)."""
    configured = params.get("image_column")
    if configured:
        return [configured] if isinstance(configured, str) else list(configured)
    auto = next((n for n in names if n in {"image", "img", "jpg", "png"}), None)
    return [auto] if auto else []


def _parquet(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    import pyarrow.parquet as pq

    assert fetched.path is not None
    pf = pq.ParquetFile(fetched.path)
    names = [f.name for f in pf.schema_arrow]
    image_cols = _image_columns(names, params)
    if not image_cols:
        raise ValueError(f"{fetched.item.key}: no image column in {names}; set image_column")
    columns: Mapping[str, str] = params.get("columns") or {}
    label_cols = list(params.get("label_columns") or [])
    classes = _class_names(pf)
    split = split_from_path(fetched.item.key)
    index = 0
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
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
            fields = {f: row.get(c) for f, c in columns.items() if row.get(c) is not None}
            for col in image_cols:
                cell = row[col]
                data, inner = (
                    (cell.get("bytes"), cell.get("path"))
                    if isinstance(cell, dict)
                    else (cell, None)
                )
                if not data:
                    continue
                suffix = (
                    suffix_of(inner)
                    if inner and suffix_of(inner) in IMAGE_SUFFIXES
                    else _sniff(data)
                )
                tag = f"_{col}" if len(image_cols) > 1 else ""
                yield Decoded(
                    upstream_id=f"{fetched.item.key}#{index}{tag}",
                    data=data,
                    suffix=suffix,
                    split_hint=split,
                    labels=labels,
                    fields=fields,
                )
            index += 1


def _video_step_frames(fetched: Fetched, params: Mapping[str, Any]) -> Iterator[Decoded]:
    """WP-6e-B (D-AB): see :mod:`marinedata.adapters.video_frames`."""
    from .video_frames import sample_video

    assert fetched.path is not None
    split = split_from_path(fetched.item.key)
    for idx, data, labels in sample_video(fetched.path, fetched.item.key, params):
        yield Decoded(
            upstream_id=f"{fetched.item.key}#frame_{idx:06d}",
            data=data,
            suffix=".jpg",
            upstream_url=fetched.item.url,
            split_hint=split,
            labels=labels,
        )


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
    bare = key.rpartition("#")[2] or key  # WP-6k: strip a remote-zip "container#member" prefix
    suffix = suffix_of(key)
    globs = params.get("label_patterns") or []
    if any(fnmatch.fnmatch(k, g) for k in (key, bare) for g in globs):
        assert fetched.stream is not None  # loose label file: pair by basename stem
        yield Decoded(key, b"", suffix, fetched.item.url, label_files={bare: fetched.stream.read()})
    elif suffix in IMAGE_SUFFIXES:
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
    elif suffix in RAR:
        yield from _group(_rar_members(fetched), fetched, params)
    elif suffix in VIDEO and params.get("frame_step"):
        yield from _video_step_frames(fetched, params)
    elif suffix in VIDEO:
        yield from _video_frames(fetched, params)
    elif suffix in ROSBAG:
        yield from _rosbag_frames(fetched, params)
    elif suffix in CAPTION_JSON_SUFFIXES:
        yield from _caption_json(fetched, params)
    else:
        raise ValueError(f"{key}: no decoder for {suffix!r}")
