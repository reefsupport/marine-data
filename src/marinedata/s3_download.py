"""Concurrent, resumable image downloads for the S3 ingest path (2026-09-18-D3d).

Split out of :mod:`marinedata.ingest_s3` purely to keep that module under its 400-line
cap (the same reason :mod:`marinedata.staging_finish` and :mod:`marinedata.s3_listing`
exist). Everything here is the download LOOP: per-stem resume-or-fetch, fanned out
across a bounded ``ThreadPoolExecutor``.

Two guarantees this module exists to keep, even under concurrency:

*Order independence.* Threads finish in whatever order the network gives them, but
:func:`download_images` folds results back into ``selected``'s own (lexicographic)
order before building ``recorded`` and ``staged_rows`` — so ``workers=1`` and
``workers=4`` produce byte-identical output.

*No half-written final-named file.* A fresh fetch already gets this from
:func:`marinedata.checksums.download_digest`'s ``.part`` + ``os.replace``. A *resumed*
file gets the same guarantee the other way round: :func:`_resume_local` only trusts a
final-named file that is already fully on disk, never a ``.part``, and removes any
stray ``.part`` it finds before deciding.
"""

from __future__ import annotations

import hashlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import checksums
from .ingest import IngestError
from .models import Source
from .s3_listing import S3Plan, _object_url
from .tables import StagedImage

DEFAULT_WORKERS = 8
"""Effective worker count when the caller (CLI or a direct ``stage_s3_source`` call)
passes ``None`` — the "not specified" sentinel, same shape as ``slice_cap_bytes``."""

_RETRY_SLEEPS = (1, 2, 4)
"""Seconds slept before each of up to 3 retries of a failed image GET (D3a2 brief §2)
— 4 attempts total, the last of which raises rather than sleeping."""

_PLAIN_MD5_ETAG = re.compile(r"^[0-9a-fA-F]{32}$")
"""A plain (non-multipart) S3 ETag is the object's exact md5 in 32 hex characters; a
multipart upload's ETag is ``<hex>-<N>`` and is not a content hash of anything this
code can reproduce locally, so only the plain form is ever checked (D3d §4)."""


def _require_pillow():
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover — exercised via sys.modules patch
        raise ImportError(
            "pillow is not installed. Install it with: pip install 'marinedata[ingest]'"
        ) from exc
    return Image


def _download_with_retry(url: str, dest: Path, *, key: str, etag: str) -> str:
    """``download_digest``, retried up to 3x with 1/2/4s sleeps, then raised
    (D3a2 brief §2) — a dropped mid-transfer connection across thousands of
    images is expected, not exceptional.

    When ``etag`` is a plain (non-multipart) md5, each attempt also hashes md5
    in the same streaming pass and checks it before returning — a fresh
    download gets the same integrity guarantee a resumed one does (D3d §4).
    A mismatch is not retried: a retry re-fetches the same bytes from the
    same key, so a wrong hash means the listing lied, not that the network
    dropped a packet.
    """
    check_md5 = _PLAIN_MD5_ETAG.fullmatch(etag) is not None
    last: Exception | None = None
    for attempt in range(len(_RETRY_SLEEPS) + 1):
        md5 = hashlib.md5() if check_md5 else None
        try:
            digest = checksums.download_digest(url, dest, extra=md5)
        except Exception as exc:  # retried, then re-raised verbatim
            last = exc
            if attempt < len(_RETRY_SLEEPS):
                time.sleep(_RETRY_SLEEPS[attempt])
            continue
        if md5 is not None and md5.hexdigest() != etag:
            raise IngestError(
                f"{key!r}: downloaded file has md5 {md5.hexdigest()} but the listing "
                f"ETag is {etag} — refusing to stage a corrupted transfer"
            )
        return digest
    raise IngestError(f"{url}: failed after {len(_RETRY_SLEEPS) + 1} attempt(s): {last}")


def _resume_local(dest: Path, size: int, etag: str, key: str) -> str | None:
    """If ``dest`` is already a final-named file matching the listing ``size`` (and,
    when ``etag`` is a plain md5, its content too — hashed in the SAME pass as the
    sha256 this returns), keep it without re-fetching. Otherwise (missing or wrong
    size) return ``None`` so the caller re-fetches (2026-09-18-D3d §4).

    A stray ``.part`` next to ``dest`` is always removed first: an interrupted
    download never resumes from a partial file, only ``download_digest``'s atomic
    rename ever produces a trustworthy final-named one.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    part.unlink(missing_ok=True)
    if not dest.is_file():
        return None
    if dest.stat().st_size != size:
        dest.unlink()
        return None
    sha256 = hashlib.sha256()
    check_md5 = _PLAIN_MD5_ETAG.fullmatch(etag) is not None
    md5 = hashlib.md5() if check_md5 else None
    with dest.open("rb") as handle:
        while chunk := handle.read(checksums.CHUNK_SIZE):
            sha256.update(chunk)
            if md5 is not None:
                md5.update(chunk)
    if md5 is not None and md5.hexdigest() != etag:
        raise IngestError(
            f"{key!r}: kept file at {dest} has md5 {md5.hexdigest()} but the listing "
            f"ETag is {etag} — a corrupted or truncated resume artefact, not a real "
            "match; remove it and re-run"
        )
    return sha256.hexdigest()


def _stage_one(
    stem: str,
    image_groups: dict[str, tuple[str, int, str]],
    plan: S3Plan,
    version_root: Path,
) -> tuple[str, str, str, int, int]:
    """Resume-or-fetch one stem's image; returns ``(stem, key, digest, width,
    height)``. Runs inside a worker thread — no shared mutable state besides the
    filesystem, which each stem touches only at its own ``dest``."""
    key, size, etag = image_groups[stem]
    dest = version_root / "images" / plan.partition / f"{stem}{plan.image_suffix}"
    digest = _resume_local(dest, size, etag, key)
    if digest is None:
        digest = _download_with_retry(_object_url(plan, key), dest, key=key, etag=etag)
    Image = _require_pillow()
    with Image.open(dest) as im:
        width, height = im.size
    return stem, key, digest, width, height


def download_images(
    selected: list[str],
    image_groups: dict[str, tuple[str, int, str]],
    plan: S3Plan,
    version_root: Path,
    workers: int,
    *,
    source: Source | None = None,
) -> tuple[dict[str, str], list[StagedImage]]:
    """Resume-or-fetch every stem in ``selected``, ``workers`` at a time.

    ``source``, when given, supplies the registry ``split_group`` rule applied to each
    staged row. ``None`` (the default) leaves ``StagedImage.split_group`` unset — every
    existing caller that predates this parameter keeps working unchanged.

    The FIRST worker exception (whichever completes first, in wall-clock order) is
    re-raised; every not-yet-started download is cancelled before that happens.
    Already-running downloads are left to finish — abandoning mid-flight bytes would
    itself risk a half-written file — but their results are simply discarded, and a
    stray ``.part`` any of them leaves is cleaned up by :func:`_resume_local` on the
    next attempt.
    """
    results: dict[str, tuple[str, str, int, int]] = {}
    total = len(selected)
    completed = 0
    bytes_done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_stage_one, stem, image_groups, plan, version_root): stem
            for stem in selected
        }
        try:
            for future in as_completed(futures):
                stem, key, digest, width, height = future.result()
                results[stem] = (key, digest, width, height)
                completed += 1
                bytes_done += image_groups[stem][1]
                if completed % 100 == 0:
                    # Progress only — stderr, never into the staged tree (no-timestamp
                    # rule covers files; this line carries no clock anyway).
                    print(
                        f"{completed}/{total} images, {bytes_done} B",
                        file=sys.stderr,
                        flush=True,
                    )
        except Exception:
            for pending in futures:
                pending.cancel()
            raise

    recorded: dict[str, str] = {}
    staged_rows: list[StagedImage] = []
    for stem in selected:
        key, digest, width, height = results[stem]
        dest = version_root / "images" / plan.partition / f"{stem}{plan.image_suffix}"
        recorded[dest.relative_to(version_root).as_posix()] = digest
        staged_rows.append(
            StagedImage(
                stem=stem,
                partition=plan.partition,
                upstream_path=key,
                upstream_split=None,
                width=width,
                height=height,
                split_group=(
                    source.split_group_for(
                        stem=stem, upstream_path=key, partition=plan.partition
                    )
                    if source is not None
                    else None
                ),
            )
        )
    return recorded, staged_rows
