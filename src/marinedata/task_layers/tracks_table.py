"""MOT tracks into the unified ``tracks`` table (WP-U9).

One row per track box: ``video_id``, 0-based ``frame_idx``, ``track_id``, the normalised ``xyxy``
box and the ``boxes`` label / provenance columns (:func:`boxes_table.box_row`). The reader is
:mod:`sources.tracks_mot`; this module binds two staged sources to it (:data:`TRACK_SOURCES`):

* ``brackishmot``: MOT ``<seq>_gt_gt.txt`` + ``<seq>_seqinfo.ini``; frame ``n`` is the image
  ``<seq>_img1_<n:06d>``; class ids 1-6 resolve through ``brackishmot-class-id``;
* ``muot3m``: single-object ``<video>_groundtruth.txt`` (one ``x,y,w,h`` per frame, no class:
  ``__unlabelled``, unmapped); the image size is read from the JPEG header of a staged frame and the
  tree has no CHECKSUMS yet, so every staged frame is *pending*.

A track frame is one of three things: its image is staged with a known sha256 (main table, sha
set); its image is not staged (main table, ``image_sha256`` null: ``video_id`` + ``frame_idx``
locate it); its image is staged but the sha256 is not resolvable (*pending*: ``image_key`` = the
upstream image stem, no sha, as for pending boxes).

Licence class: ``brackishmot`` internal-only (licence unknown, lic-A); ``muot3m`` restricted-nd
(CC-BY-NC-ND-4.0): internal training only, never released to HF.
"""

# ruff: noqa: E501

from __future__ import annotations

import io
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..annotation_schema import annotation_path, arrow_schema, validate_row, write_annotations
from ..registry import Registry
from .boxes_table import ND, BoxSource, box_row, bucket_lister, resolver_for_source
from .image_labels_table import _stride
from .masks_table import INTERNAL_ONLY
from .s3_keyed import StagedTree
from .sources.boxes_common import BoxCounts, BoxFormatError
from .sources.tracks_mot import MOT_CHALLENGE, SOT_GROUNDTRUTH, TrackBox, parse_seqinfo, read_mot

TRACK_SOURCES: dict[str, BoxSource] = {
    s.source_id: s
    for s in (
        BoxSource("brackishmot", "zip-e717dc1438aa", "mot-challenge", "human", "NOASSERTION",
                  INTERNAL_ONLY, crosswalk_id="brackishmot-class-id",
                  label_set="brackishmot-class-id"),
        BoxSource("muot3m", "rev-4294cb6a2537", "sot-groundtruth", "human", "CC-BY-NC-ND-4.0",
                  ND),
    )
}  # fmt: skip
PENDING_COLUMNS = ("image_key",)
_BRACKISH_SEQ = re.compile(r"^(?P<seq>.+)_gt_gt\.txt$")
_BRACKISH_SPLIT = re.compile(r"^BrackishMOT_(?P<split>train|val|test)_")
_MUOT_FRAME = re.compile(r"^(?P<video>(?:train|test)_Video_\d+)_Video_\d+_mp4_frame_(?P<n>\d+)$")
_MUOT_GT = re.compile(r"^(?P<video>(?:train|test)_Video_\d+)_groundtruth\.txt$")


@dataclass(frozen=True)
class TrackResult:
    rows: tuple[dict, ...]
    pending: tuple[dict, ...]  # staged frames whose sha256 is not resolvable
    counts: BoxCounts
    videos: int  # videos with at least one row
    unparsable: tuple[str, ...]


def track_row(
    *, spec: BoxSource, registry: Registry, resolve, ordinal: int, video_id: str, tb: TrackBox,
    sha: str | None, split: str | None, staged: bool,
) -> dict:  # fmt: skip
    row = box_row(
        spec=spec, registry=registry, resolve=resolve, ordinal=ordinal, image_sha256=sha,
        box=tb.box, split=split, extra_attrs={"frame_staged": staged},
    )  # fmt: skip
    row.update(video_id=video_id, frame_idx=tb.frame_idx, track_id=tb.track_id, is_crowd=None)
    return row


def _split(video: str, rx: re.Pattern[str]) -> str | None:
    m = rx.match(video)
    return m["split"] if m else None


def _emit(spec, registry, videos, *, sha_of, staged_of, split_of) -> TrackResult:
    """``videos``: ``[(video_id, (TrackBox, ...), BoxCounts)]``; ``staged_of(video, native)`` is the
    image stem when that frame is staged; ``sha_of(stem)`` its sha256 or None."""
    resolve = resolver_for_source(registry, spec)
    rows: list[dict] = []
    pend: list[dict] = []
    counts, nvid = BoxCounts(), 0
    for video, boxes, c in videos:
        counts, nvid = counts + c, nvid + bool(boxes)
        for tb in boxes:
            stem = staged_of(video, tb.native_frame)
            sha = sha_of(stem) if stem else None
            row = track_row(
                spec=spec, registry=registry, resolve=resolve, ordinal=len(rows) + len(pend),
                video_id=video, tb=tb, sha=sha, split=split_of(video), staged=stem is not None,
            )  # fmt: skip
            if stem and sha is None:
                pend.append({**row, "image_key": stem})
            else:
                rows.append(row)
    return TrackResult(tuple(rows), tuple(pend), counts, nvid, ())


def _brackishmot(spec, registry, limit, fetch) -> TrackResult:
    tree = StagedTree(spec.tree, fetch)
    seqs = sorted(
        m["seq"] for rel in tree.checksums if (m := _BRACKISH_SEQ.match(rel.rsplit("/", 1)[-1]))
    )
    bad: list[str] = []
    videos = []
    for seq in _stride(seqs, limit):
        gt = tree.get_or_skip(f"labels/files/{seq}_gt_gt.txt")
        info = tree.get_or_skip(f"labels/files/{seq}_seqinfo.ini")
        w, h = parse_seqinfo(info.decode()) if info else (None, None)
        try:
            if gt is None:
                raise BoxFormatError("gt file not readable")
            boxes, counts = read_mot(gt.decode(), MOT_CHALLENGE, img_w=w, img_h=h)
        except (BoxFormatError, UnicodeDecodeError) as exc:
            bad.append(f"{seq}: {exc}"[:160])
            continue
        videos.append((seq, boxes, counts))

    def stem(v: str, n: int) -> str | None:
        name = f"{v}_img1_{n:06d}"
        return name if ("default", name) in tree.shas else None

    got = _emit(
        spec, registry, videos, staged_of=stem,
        sha_of=lambda s: tree.shas.get(("default", s)), split_of=lambda v: _split(v, _BRACKISH_SPLIT),
    )  # fmt: skip
    return TrackResult(got.rows, got.pending, got.counts, got.videos, tuple(bad[:10]))


def jpeg_size_reader(bucket: str = "rs-storage-open") -> Callable[[str], tuple[int, int] | None]:
    """``key -> (width, height)`` from the first 128 KiB of a JPEG (a ranged, read-only GET)."""
    import configparser
    import os

    import boto3
    from PIL import Image

    sec = configparser.ConfigParser()
    sec.read(os.path.expanduser("~/.config/rclone/rclone.conf"))
    sec = sec["rs-hel1"]
    ep = sec["endpoint"]
    client = boto3.client(
        "s3", aws_access_key_id=sec["access_key_id"], aws_secret_access_key=sec["secret_access_key"],
        endpoint_url=ep if ep.startswith("http") else f"https://{ep}",
    )  # fmt: skip

    def size(key: str) -> tuple[int, int] | None:
        head = client.get_object(Bucket=bucket, Key=key, Range="bytes=0-131071")["Body"].read()
        try:
            return Image.open(io.BytesIO(head)).size
        except Exception:
            return None

    return size


def _muot3m(spec, registry, limit, fetch, lister, image_size) -> TrackResult:
    base = f"sources/{spec.tree}/"
    staged: dict[str, dict[int, str]] = {}
    for key in lister(base + "images/"):
        if m := _MUOT_FRAME.match(key.rsplit("/", 1)[-1].rsplit(".", 1)[0]):
            staged.setdefault(m["video"], {})[int(m["n"])] = key
    gts = sorted(
        (m["video"], k)
        for k in lister(base + "labels/files/")
        if (m := _MUOT_GT.match(k.rsplit("/", 1)[-1]))
    )
    bad: list[str] = []
    videos = []
    for video, key in _stride(gts, limit):
        frames = staged.get(video, {})
        size = image_size(frames[min(frames)]) if frames else None
        if size is None:
            bad.append(f"{video}: no staged frame to read the image size from")
            continue
        try:
            boxes, counts = read_mot(
                fetch(key).decode(), SOT_GROUNDTRUTH, img_w=size[0], img_h=size[1]
            )
        except (BoxFormatError, UnicodeDecodeError) as exc:
            bad.append(f"{video}: {exc}"[:160])
            continue
        videos.append((video, boxes, counts))

    def stem(v: str, n: int) -> str | None:
        key = staged.get(v, {}).get(n)
        return key.rsplit("/", 1)[-1].rsplit(".", 1)[0] if key else None

    got = _emit(
        spec, registry, videos, staged_of=stem, sha_of=lambda _s: None,
        split_of=lambda v: v.split("_", 1)[0],
    )  # fmt: skip
    return TrackResult(got.rows, got.pending, got.counts, got.videos, tuple(bad[:10]))


def staged_tracks(
    spec: BoxSource,
    registry: Registry,
    *,
    limit: int | None = None,
    fetch: Callable[[str], bytes],
    lister: Callable[[str], list[str]] | None = None,
    image_size: Callable[[str], tuple[int, int] | None] | None = None,
) -> TrackResult:
    """Read one source's staged tracks, at most ``limit`` videos (evenly spaced, deterministic)."""
    if spec.reader == "mot-challenge":
        return _brackishmot(spec, registry, limit, fetch)
    return _muot3m(spec, registry, limit, fetch, lister or bucket_lister(),
                   image_size or jpeg_size_reader())  # fmt: skip


def write_tracks(data_dir: Path, source_id: str, source_version: str, rows: list[dict]) -> Path:
    path = annotation_path(data_dir, "tracks", source_id, source_version)
    write_annotations(path, "tracks", rows)
    return path


def validate_pending(rows: Iterable[Mapping[str, object]]) -> list[str]:
    errs: list[str] = []
    for row in rows:
        core = {k: v for k, v in row.items() if k not in PENDING_COLUMNS}
        errs += [f"{row['ann_id']}: {e}" for e in validate_row("tracks", core)]
        if core.get("image_sha256") is not None or not row.get("image_key"):
            errs.append(f"{row['ann_id']}: a pending row has image_key and no image_sha256")
    return errs


def write_pending_tracks(path: Path, rows: list[dict]) -> int:
    """Staged frames with no resolvable sha256: ``tracks`` columns plus ``image_key``."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if errs := validate_pending(rows):
        raise ValueError("; ".join(errs[:10]))
    base = arrow_schema("tracks")
    fields = [*base] + [pa.field(c, pa.string()) for c in PENDING_COLUMNS]
    table = pa.table(
        {f.name: [r.get(f.name) for r in rows] for f in fields},
        schema=pa.schema(fields, metadata=base.metadata),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return len(rows)


__all__ = [
    "TRACK_SOURCES", "TrackResult", "jpeg_size_reader", "staged_tracks", "track_row",
    "write_pending_tracks", "write_tracks",
]  # fmt: skip
