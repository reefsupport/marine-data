"""S3-keyed task-label producers (WP-8e): full-scale ``data/_tasklabels/<src>/<task>.parquet``.

WP-8c's producers key ``sha256`` by hashing images in a local cache, so a partial cache
gives a tiny build (5-55 images). These producers key every row from the staged tree's
own ``CHECKSUMS.sha256`` on ``rs-storage-open`` (``images/<partition>/<stem>.<ext>`` ->
sha256), joined to ``metadata.parquet`` and the tree's ``labels/`` files by
``(partition, stem)``. Nothing under ``images/`` is ever fetched: :func:`fetch_small`
refuses those keys, and every read is an anonymous HTTPS GET held in memory (no disk).

Mask-derived payloads (Coralscapes semseg, Reef Support bleaching presence) decode the
``labels/masks/`` PNGs in memory — those are label files, never the imagery.

The D-Z2 columns are unchanged (``sha256``, ``source_id``, ``label_origin`` + payload), so
:mod:`marinedata.task_layers.configs` consumes these files exactly as it does WP-8c's.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

from ..checksums import parse_checksums
from ..tables import _require_pyarrow

PUBLIC_HOST = "https://rs-storage-open.hel1.your-objectstorage.com"
SMALL_FILE_CAP = 64 * (1 << 20)
IMAGE_PREFIX = "images/"

Fetch = Callable[[str], bytes]


class ImageFetchRefused(ValueError):
    """A producer asked for a key under ``images/`` — imagery is never downloaded."""


class FetchFailed(OSError):
    """A small-file GET still failing after every retry; ``status`` is the last HTTP code
    (or the transport error's name) so ``missing_keys.tsv`` can say why."""

    def __init__(self, key: str, tries: int, status: str) -> None:
        super().__init__(f"{key}: GET failed after {tries} tries (last: {status})")
        self.key, self.tries, self.status = key, tries, status


def _status(exc: BaseException) -> str:
    code = getattr(exc, "code", None)  # urllib.error.HTTPError
    if code is not None:
        return str(code)
    reason = getattr(exc, "reason", None)  # urllib.error.URLError
    return type(reason if isinstance(reason, BaseException) else exc).__name__


def fetch_small(
    key: str,
    *,
    tries: int = 8,
    timeout: float = 30.0,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    sleep: Callable[[float], None] = time.sleep,
    opener: Callable = urllib.request.urlopen,
) -> bytes:
    """Anonymous GET of one small non-image object on ``rs-storage-open`` (in memory).

    Hetzner returns intermittent 403/5xx on this bucket, so every failure is retried with
    exponential backoff and full jitter (``uniform(0, min(max_delay, base_delay * 2**i))``)
    up to ``tries`` attempts, then :class:`FetchFailed` carries the last status.
    """
    if f"/{IMAGE_PREFIX}" in f"/{key}":
        raise ImageFetchRefused(key)
    url = f"{PUBLIC_HOST}/{urllib.parse.quote(key)}"
    status = "none"
    for attempt in range(tries):
        if attempt:
            sleep(random.uniform(0.0, min(max_delay, base_delay * 2 ** (attempt - 1))))
        try:
            with opener(url, timeout=timeout) as resp:
                data = resp.read(SMALL_FILE_CAP + 1)
        except (OSError, TimeoutError) as exc:  # HTTPError/URLError are OSErrors
            status = _status(exc)
            continue
        if len(data) > SMALL_FILE_CAP:
            raise ValueError(f"{key}: larger than the {SMALL_FILE_CAP} B small-file cap")
        return data
    raise FetchFailed(key, tries, status)


def sha_index(checksums_text: str) -> dict[tuple[str, str], str]:
    """``(partition, stem) -> sha256`` for every ``images/`` entry of a CHECKSUMS file.

    ``images/<partition>/<file>`` keys the partition; a flat ``images/<file>`` (the
    StagedWriter ``objects`` layout) is partition ``"default"``.
    """
    out: dict[tuple[str, str], str] = {}
    for rel, sha in parse_checksums(checksums_text).items():
        parts = PurePosixPath(rel).parts
        if not parts or parts[0] != "images" or len(parts) < 2:
            continue
        partition = parts[1] if len(parts) == 3 else "default"
        key = (partition, PurePosixPath(parts[-1]).stem)
        if key in out and out[key] != sha:
            raise ValueError(f"two images share {key!r} with different sha256")
        out[key] = sha
    return out


def _table(data: bytes) -> list[dict]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(data)).to_pylist()


def _write(rows: list[dict], out_path: Path) -> int:
    _require_pyarrow()
    import pyarrow as pa
    import pyarrow.parquet as pq

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".partial")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd")
    os.replace(tmp, out_path)  # a parquet at out_path is always a finished run (resume key)
    return len(rows)


class StagedTree:
    """One staged source version on S3, read through ``fetch`` (small files only)."""

    def __init__(self, tree: str, fetch: Fetch = fetch_small) -> None:
        self.tree = tree.strip("/")
        self.fetch = fetch
        self.missing: list[tuple[str, str]] = []
        """``(key, last_status)`` of every per-item GET that failed and was skipped."""
        text = self.get("CHECKSUMS.sha256").decode()
        self.checksums = parse_checksums(text)
        self.shas = sha_index(text)

    def key(self, rel: str) -> str:
        return f"sources/{self.tree}/{rel}"

    def get(self, rel: str) -> bytes:
        return self.fetch(self.key(rel))

    def get_or_skip(self, rel: str) -> bytes | None:
        """``get`` for one item of many: a :class:`FetchFailed` is recorded in
        :attr:`missing` and returns ``None`` — one bad key never kills a full-scale run."""
        try:
            return self.get(rel)
        except FetchFailed as exc:
            self.missing.append((exc.key, exc.status))
            return None

    def table(self, rel: str) -> list[dict]:
        return _table(self.get(rel))

    def masks(self) -> list[tuple[tuple[str, str], str]]:
        """``((partition, stem), rel)`` for every ``labels/masks/<partition>/<stem>.png``."""
        out = []
        for rel in sorted(self.checksums):
            parts = PurePosixPath(rel).parts
            if parts[:2] == ("labels", "masks") and rel.endswith(".png"):
                partition = parts[2] if len(parts) == 4 else "default"
                out.append(((partition, PurePosixPath(rel).stem), rel))
        return out


def produce_points(tree: StagedTree, source_id: str, out_path: Path) -> int:
    """``points``: one row per labelled point, ``x``/``y`` normalised by the image size."""
    meta = {(r["partition"], r["stem"]): r for r in tree.table("metadata.parquet")}
    rows: list[dict] = []
    for p in tree.table("labels/points.parquet"):
        key = (p["partition"], p["stem"])
        image, sha = meta.get(key), tree.shas.get(key)
        if image is None or sha is None:
            raise ValueError(f"{source_id}: point references unstaged image {key!r}")
        rows.append(
            {
                "sha256": sha,
                "source_id": source_id,
                "label_origin": "human",
                "native_label": p["label"],
                "x": p["col"] / image["width"],
                "y": p["row"] / image["height"],
            }
        )
    return _write(rows, out_path)


def produce_image_labels(tree: StagedTree, source_id: str, out_path: Path) -> int:
    """``bleaching`` from ``labels/image_labels.parquet``: one row per (image, label)."""
    rows: list[dict] = []
    for r in tree.table("labels/image_labels.parquet"):
        key = (r["partition"], r["stem"])
        sha = tree.shas.get(key)
        if sha is None:
            raise ValueError(f"{source_id}: image label references unstaged image {key!r}")
        rows.append(
            {
                "sha256": sha,
                "source_id": source_id,
                "label_origin": "human",
                "native_label": r["label"],
                "evidence": "image",
                "pixel_count": None,
                "confidence": r.get("confidence"),
            }
        )
    return _write(rows, out_path)


def _pixel_counts(data: bytes) -> dict[int, int]:
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        arr = arr[..., 0]
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True)}


def _mask_counts(tree: StagedTree, workers: int) -> Iterable[tuple[tuple[str, str], str, dict]]:
    def counts(m: tuple[tuple[str, str], str]) -> dict[int, int] | None:
        data = tree.get_or_skip(m[1])
        return None if data is None else _pixel_counts(data)

    masks = tree.masks()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (key, rel), got in zip(masks, pool.map(counts, masks), strict=True):
            if got is not None:
                yield key, rel, got


def produce_mask_presence(
    tree: StagedTree,
    source_id: str,
    values: dict[int, str],
    out_path: Path,
    *,
    workers: int = 16,
) -> int:
    """``bleaching`` from class masks: one row per (image, class present), with pixels."""
    rows: list[dict] = []
    for key, _rel, counts in _mask_counts(tree, workers):
        sha = tree.shas.get(key)
        if sha is None:
            raise ValueError(f"{source_id}: mask references unstaged image {key!r}")
        for value, n in sorted(counts.items()):
            if value in values and n > 0:
                rows.append(
                    {
                        "sha256": sha,
                        "source_id": source_id,
                        "label_origin": "human",
                        "native_label": values[value],
                        "evidence": "mask",
                        "pixel_count": n,
                        "confidence": None,
                    }
                )
    return _write(rows, out_path)


def produce_semseg(
    tree: StagedTree,
    source_id: str,
    id_to_label: dict[int, str],
    out_path: Path,
    *,
    ignore: frozenset[int] = frozenset({0}),
    workers: int = 16,
) -> int:
    """``semseg``: one row per image, native-label pixel histogram as JSON (D-Z2)."""
    rows: list[dict] = []
    for key, rel, counts in _mask_counts(tree, workers):
        sha = tree.shas.get(key)
        if sha is None:
            raise ValueError(f"{source_id}: mask references unstaged image {key!r}")
        named: dict[str, int] = {}
        for value, n in counts.items():
            if value in ignore:
                continue
            label = id_to_label.get(value, f"__unknown_index_{value}")
            named[label] = named.get(label, 0) + n
        if named:
            rows.append(
                {
                    "sha256": sha,
                    "source_id": source_id,
                    "label_origin": "human",
                    "mask_key": rel,
                    "class_counts": json.dumps(named, sort_keys=True),
                }
            )
    return _write(rows, out_path)
