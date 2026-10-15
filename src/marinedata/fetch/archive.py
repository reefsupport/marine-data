"""Archive extraction under hard caps: per-member and cumulative byte limits, a compression
ratio ceiling, path-traversal refusal, in-memory and Range-read (remote) zip/tar readers."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from .core import (
    FetchError,
    facade_cap,
)
from .ranged import _RemoteFile


# The caps are read from the public `marinedata.fetch` facade on every call: tests and
# operators override them there (see `core.facade_cap`).
def _member_cap() -> int:
    return facade_cap("MAX_MEMBER_BYTES")


def _total_cap() -> int:
    return facade_cap("MAX_TOTAL_EXTRACT_BYTES")


def _ratio_cap() -> int:
    return facade_cap("MAX_COMPRESSION_RATIO")


def _nested_zip_window(outer: zipfile.ZipFile, member_name: str) -> tuple[int, int]:
    """Byte range of a member stored *uncompressed* inside an already-open zip.

    Returns ``(data_offset, size)`` — where the member's own bytes begin within the
    outer file, and how many there are. Only valid when the member's
    ``compress_type`` is ``ZIP_STORED``: those bytes are the member's real content
    verbatim, so if the member is itself a zip, this range can be opened as one
    without extracting it first. Raises for anything else — a DEFLATEd nested archive
    cannot be windowed this way, and guessing would produce a file that mysteriously
    fails to parse instead of a clear error naming the actual cause.
    """
    import struct

    info = outer.getinfo(member_name)
    if info.compress_type != zipfile.ZIP_STORED:
        raise FetchError(
            f"'{member_name}' is compressed (not stored) inside its outer archive — "
            f"it cannot be read as a nested zip without extracting the whole thing "
            f"first, which defeats the point of a bounded sample."
        )

    # header_offset points at the LOCAL file header, not the data — the local header
    # can carry different filename/extra-field lengths than the central directory
    # entry does, so its size has to be read, not assumed.
    fp = outer.fp
    fp.seek(info.header_offset)
    header = fp.read(30)
    if header[:4] != b"PK\x03\x04":
        raise FetchError(f"'{member_name}': malformed local file header in outer archive")
    filename_len, extra_len = struct.unpack("<HH", header[26:30])
    data_offset = info.header_offset + 30 + filename_len + extra_len
    return data_offset, info.compress_size


def _check_zip_member_bounds(label: str, member: zipfile.ZipInfo, total_so_far: int) -> None:
    """Raise before extracting ``member`` if it would breach a byte cap.

    Checked against the central-directory metadata alone — ``file_size`` and
    ``compress_size`` are both known without reading the member's data, so an oversized
    or bomb-shaped member is refused before any bytes for it are pulled or decompressed.
    """
    if member.file_size > _member_cap():
        raise FetchError(
            f"archive at {label}: member '{member.filename}' would extract to "
            f"{member.file_size} bytes, over the {_member_cap()}-byte per-member cap"
        )
    if member.compress_size and member.file_size / member.compress_size > facade_cap(
        "MAX_COMPRESSION_RATIO"
    ):
        raise FetchError(
            f"archive at {label}: member '{member.filename}' compresses "
            f"{member.file_size}:{member.compress_size} "
            f"({member.file_size / member.compress_size:.0f}:1), over the "
            f"{_ratio_cap()}:1 ratio cap — refused as a likely zip bomb"
        )
    if total_so_far + member.file_size > _total_cap():
        raise FetchError(
            f"archive at {label}: extracting member '{member.filename}' would push "
            f"cumulative extracted bytes past the {_total_cap()}-byte cap"
        )


def _check_tar_member_bounds(label: str, member: tarfile.TarInfo, total_so_far: int) -> None:
    """Tar equivalent of :func:`_check_zip_member_bounds`.

    No per-member compressed size exists in a tar stream (unlike zip's central
    directory), so there is no per-member ratio check here — only a per-member and a
    cumulative cap. A whole-archive ratio guard for a compressed tar stream lives at
    each caller that actually knows the compressed byte count (``len(payload)``).
    """
    if member.size > _member_cap():
        raise FetchError(
            f"archive at {label}: member '{member.name}' would extract to "
            f"{member.size} bytes, over the {_member_cap()}-byte per-member cap"
        )
    if total_so_far + member.size > _total_cap():
        raise FetchError(
            f"archive at {label}: extracting member '{member.name}' would push "
            f"cumulative extracted bytes past the {_total_cap()}-byte cap"
        )


def _extract_remote_zip(file_obj: _RemoteFile, root: Path, limit: int, *, label: str) -> int:
    """Extract up to ``limit`` members from a remote zip without downloading it.

    ``file_obj`` is whatever :class:`_RemoteFile` (or compatible) view the caller
    already built — a whole remote archive, or a window into one nested inside
    another. The counterpart to :func:`_extract_archive` for archives too large to
    pull in full — see ``MAX_DOWNLOAD_WITHOUT_RANGE``.
    """
    from ..adapters.zipread import read_member  # Deflate64 members (WP-6j)

    resolved_root = root.resolve()
    count = 0
    total_bytes = 0
    with zipfile.ZipFile(file_obj) as archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        for member in members[:limit] if limit else members:
            target = _safe_extract_path(resolved_root, member.filename)
            if target is None:
                continue
            _check_zip_member_bounds(label, member, total_bytes)
            total_bytes += member.file_size
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_member(archive, member))
            count += 1

    if count == 0:
        raise FetchError(f"archive at {label} contained no extractable files")
    return count


def _extract_remote_tar(file_obj: _RemoteFile, root: Path, limit: int, *, label: str) -> int:
    """Extract up to ``limit`` members from a remote **uncompressed** tar without
    downloading it.

    Unlike zip, tar has no central directory to read upfront — the only way to know
    what is in it is to read forward from the start, one 512-byte header block at a
    time. ``tarfile.TarFile`` is iterable exactly this way, and critically,
    :meth:`TarFile.getmembers` must never be called here: it exhausts the *whole*
    stream to build its list, which for a multi-GB archive means downloading all of
    it — the exact cost this function exists to avoid. Iterating the archive object
    itself and stopping after ``limit`` members only ever reads as far as the last
    member's data, because :class:`_RemoteFile` treats an unread gap as a cheap
    ``seek`` (no request) rather than bytes that must be fetched.

    Compressed tars (``.tar.gz``/``.tar.bz2``/``.tar.xz``) are not supported this way:
    decompression is inherently sequential from the very start, so a compressed
    stream cannot skip past an unwanted member's data without decompressing it
    anyway, which defeats the point of a bounded sample.
    """
    resolved_root = root.resolve()
    count = 0
    total_bytes = 0
    with tarfile.open(fileobj=file_obj, mode="r:") as archive:
        for member in archive:
            if limit and count >= limit:
                break
            if not member.isfile():
                continue
            target = _safe_extract_path(resolved_root, member.name)
            if target is None:
                continue
            _check_tar_member_bounds(label, member, total_bytes)
            total_bytes += member.size
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(extracted.read())
            count += 1

    if count == 0:
        raise FetchError(f"archive at {label} contained no extractable files")
    return count


def _safe_extract_path(resolved_root: Path, member_name: str) -> Path | None:
    """Resolve an archive member's target path, refusing to leave ``resolved_root``.

    Archive members are attacker-influenced input in general (here, a maintainer-chosen
    URL, but the extraction code itself does not know that) — an absolute path or a
    ``../`` traversal in a member name must not write outside the sample directory.
    Returns ``None`` for a member that would escape, so the caller can skip it.

    ``resolved_root`` must already be ``.resolve()``d by the caller, once, before the
    extraction loop. Resolving it fresh on every call here — comparing an *unresolved*
    root against a *resolved* target — used to reject every single member as "escaping"
    whenever the destination sat under a symlink, which on macOS ``/tmp`` always does:
    caught by a live verification run against a real archive, silently invisible to
    every synthetic-fixture test because pytest's ``tmp_path`` is never symlinked.
    """
    target = (resolved_root / member_name).resolve()
    if target != resolved_root and resolved_root not in target.parents:
        return None
    return target


def _extract_archive(payload: bytes, root: Path, name: str, limit: int) -> int:
    """Extract a small archive in place, bounded to ``limit`` files.

    Handles ``.zip`` and ``.tar`` (optionally ``.gz``/``.bz2``/``.xz``) — the two
    conventions every dataset host in this registry actually uses. Bounded because a
    verification sample is meant to stay small even when the upstream archive is not.
    """
    import io

    from ..adapters.zipread import read_member  # Deflate64 members (WP-6j)

    resolved_root = root.resolve()
    count = 0
    total_bytes = 0
    lower = name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            for member in members[:limit] if limit else members:
                target = _safe_extract_path(resolved_root, member.filename)
                if target is None:
                    continue
                _check_zip_member_bounds(name, member, total_bytes)
                total_bytes += member.file_size
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(read_member(archive, member))
                count += 1
    else:
        mode = "r:*"  # autodetect gz/bz2/xz/plain from the stream itself
        with tarfile.open(fileobj=io.BytesIO(payload), mode=mode) as archive:
            members = [m for m in archive.getmembers() if m.isfile()]
            for member in members[:limit] if limit else members:
                target = _safe_extract_path(resolved_root, member.name)
                if target is None:
                    continue
                _check_tar_member_bounds(name, member, total_bytes)
                # A compressed tar stream (.tar.gz/.bz2/.xz) has no per-member
                # compressed size — tar's own header only ever carries the
                # uncompressed size — so the ratio guard here is whole-archive:
                # bytes extracted so far against the compressed payload actually
                # received, which is the only "compressed size" a tar stream has.
                projected = total_bytes + member.size
                if len(payload) and projected / len(payload) > _ratio_cap():
                    raise FetchError(
                        f"archive '{name}': cumulative extracted size "
                        f"{total_bytes + member.size} bytes is over "
                        f"{_ratio_cap()}x the {len(payload)}-byte compressed "
                        f"archive — refused as a likely zip/tar bomb"
                    )
                total_bytes += member.size
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(extracted.read())
                count += 1

    if count == 0:
        raise FetchError(f"archive '{name}' contained no extractable files")
    return count
