"""``CHECKSUMS.sha256`` — the per-file digest manifest for one stored source version.

The target bucket layout puts a ``CHECKSUMS.sha256`` in every
``sources/<source_id>/<version>/``, and nothing produced it. This module does, in the
plain ``sha256sum`` format so that verification needs no code from this package::

    <64 lowercase hex><two spaces><posix path relative to the version dir>\\n

Lines are sorted by path and the manifest never lists itself, so ``shasum -a 256 -c
CHECKSUMS.sha256`` run from inside the version directory checks the whole tree. Sorting
uses Python's string order, which is code-point order, which UTF-8 preserves bytewise —
the two orders coincide by construction of the encoding (pinned by a test, not asserted
here).

Three properties this file exists to guarantee:

*Determinism.* The same tree yields byte-identical output. Nothing timestamped, ordered
by filesystem iteration, or platform-dependent reaches the file: paths are POSIX-joined,
the newline is forced to ``\\n``, and the sort is total.

*Bounded memory.* The corpus is 1-3 TB across potentially 100k+ files. Every file is
hashed in fixed-size chunks and lines are streamed to disk as they are produced, so the
only thing held whole is the path-to-digest map itself.

*Immutability.* A version, once written, is what its digests say it is. Re-running over
a directory that already carries a manifest re-derives the digests and raises if any
path changed, appeared or vanished. This follows the rest of the registry's gates: a
contract that cannot be honoured raises, it never warns and never overwrites.

There are no recorded per-file digests anywhere else in this package to consume:
:func:`marinedata.fetch.sample_digest` folds a bounded ~100-item sample into a single
truncated tree digest, and :mod:`marinedata.lineage` hashes a JSON payload. Both answer
"did this change", not "what is each file". So the staged tree is the authority here,
and ``recorded`` (see :func:`scan_digests`) is only an optimisation for a caller that
already streamed a file's bytes once — an uploader, typically — and should not pay to
read terabytes twice. The chunked reader itself is shared with ``sample_digest`` so that
``hashlib.sha256`` is fed file bytes in exactly one place in this package.
"""

from __future__ import annotations

import hashlib
import os
import re
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

CHECKSUM_FILE = "CHECKSUMS.sha256"

CHUNK_SIZE = 1 << 20
"""Bytes per read. Large enough that syscall overhead vanishes on a multi-GB file,
small enough that hashing a 100 GB archive never grows the process."""

_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
_HEX = re.compile(r"^[0-9a-f]{64}$")

_ILLEGAL_IN_PATH = ("\n", "\r", "\\")
"""``sha256sum`` gives these meaning: a newline ends a record and a backslash starts an
escape. A path containing one cannot be represented unambiguously, so it is refused at
write time rather than written into a manifest that silently checks the wrong file."""


class ChecksumError(ValueError):
    """A manifest is malformed, or writing one would contradict an existing manifest."""


@dataclass(frozen=True)
class ChecksumManifest:
    """What a written manifest covers, and the digest that pins it."""

    path: Path
    root_digest: str
    """sha256 of the ``CHECKSUMS.sha256`` file itself — one value that pins the whole
    version, small enough to live in the registry."""

    files: int
    size_bytes: int
    """Total bytes covered, from ``stat`` rather than a second read. The definition of
    done pairs ``size_bytes`` with the manifest, and this pass already visits every file."""


def stream_into(digest: hashlib._Hash, path: Path) -> int:
    """Feed a file's bytes into ``digest`` in fixed-size chunks. Returns bytes read.

    The one place in this package that reads file bytes into a hash. Chunked so that a
    file larger than memory is not a failure mode.
    """
    read = 0
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
            read += len(chunk)
    return read


def file_digest(path: str | Path) -> str:
    """Lowercase 64-hex sha256 of one file's bytes, streamed.

    Bare hex with no ``sha256:`` prefix and no truncation, because the value goes into a
    ``sha256sum``-format manifest where any decoration would break ``-c``.
    """
    digest = hashlib.sha256()
    stream_into(digest, Path(path))
    return digest.hexdigest()


def write_digest(path: Path, payload: bytes) -> str:
    """Write ``payload`` to ``path`` and return its sha256. One pass over the bytes.

    The write-side counterpart to :func:`file_digest`: :func:`stream_into` opens its
    path for *reading*, so calling it right after a write would be a second pass over
    bytes already in hand. This hashes the buffer directly instead.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(payload)
    path.write_bytes(payload)
    return digest.hexdigest()


def copy_digest(src: Path, dest: Path) -> str:
    """Stream ``src`` to ``dest`` in :data:`CHUNK_SIZE` chunks, hashing as it goes.

    Carries large files (images) without holding them whole in memory, and without a
    second read: each chunk is written to ``dest`` and folded into the digest in the
    same pass.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with src.open("rb") as source, dest.open("wb") as target:
        while chunk := source.read(CHUNK_SIZE):
            digest.update(chunk)
            target.write(chunk)
    return digest.hexdigest()


DOWNLOAD_USER_AGENT = "marinedata/0.1 (+https://github.com/reefsupport/marine-data)"
"""The one ``User-Agent`` literal for this package. :mod:`marinedata.fetch` already
imports :data:`CHUNK_SIZE` from this module, so the reverse import — this module
calling :func:`marinedata.fetch.get_stream` to avoid a second HTTP stack — would be
circular; this module is where both call sites (here, ``fetch.py``, and
:mod:`marinedata.s3_listing`, D3a2) reach a shared opener without that cycle.
``fetch.py`` imports this name as ``USER_AGENT`` rather than holding a second literal
(D3a2 tidy)."""


def download_digest(
    url: str, dest: str | Path, *, timeout: int = 60, extra: hashlib._Hash | None = None
) -> str:
    """GET ``url``, streaming the body to ``dest`` and sha256-ing it in the SAME pass.

    Same request plumbing as :func:`marinedata.fetch.get_stream` — a ``User-Agent``
    header, redirects handled by ``urlopen`` itself, :data:`CHUNK_SIZE` pieces (the
    same constant ``fetch.py`` imports from here) — without importing it back, which
    would be circular (see :data:`_DOWNLOAD_USER_AGENT`).

    Writes to ``dest.with_suffix(dest.suffix + ".part")`` first and ``os.replace``s it
    into place only once the whole body has landed, so a failed or interrupted transfer
    never leaves a partial file at ``dest`` — and the ``.part`` file itself is removed
    on any error rather than left behind.

    ``extra`` (2026-09-18-D3d §4), when given, is fed every chunk alongside the sha256
    — the S3 path's way of getting a second digest (an md5 to check against a listing
    ETag) without a second read of a multi-hundred-MB file. The caller owns ``extra``
    and reads its digest back once this returns.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": DOWNLOAD_USER_AGENT})
    try:
        with (
            urllib.request.urlopen(request, timeout=timeout) as response,
            part.open("wb") as handle,
        ):
            while chunk := response.read(CHUNK_SIZE):
                handle.write(chunk)
                digest.update(chunk)
                if extra is not None:
                    extra.update(chunk)
    except Exception:
        part.unlink(missing_ok=True)
        raise
    os.replace(part, dest)
    return digest.hexdigest()


def _relative_paths(root: Path) -> Iterator[tuple[str, Path]]:
    """Walk ``root``, yielding ``(posix relative path, absolute path)`` for every file.

    ``os.walk`` rather than ``rglob`` because it does not follow directory symlinks —
    a staging mistake that makes a loop becomes an error, not a hang — and because it
    does not materialise the whole tree before the first file is hashed.
    """
    manifest = root / CHECKSUM_FILE
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        for name in filenames:
            path = directory / name
            if path == manifest:
                continue  # A manifest never lists itself: it cannot contain its own digest.
            if not path.is_file():
                continue
            yield path.relative_to(root).as_posix(), path


def scan_digests(root: str | Path, *, recorded: Mapping[str, str] | None = None) -> dict[str, str]:
    """Map every file under ``root`` to its sha256, as POSIX paths relative to ``root``.

    The tree decides which paths exist; ``recorded`` only supplies digests already known
    for them. A key in ``recorded`` matching a walked path is trusted and the file is not
    read; keys naming paths that are not there are ignored, and paths ``recorded`` does
    not cover are hashed. So a partial or stale map still produces a complete, correct
    manifest — it can only make the run slower, never wrong. A recorded value that is not
    64 lowercase hex raises: a malformed digest written into a manifest would make every
    later check fail against the wrong expectation.
    """
    root = Path(root)
    if not root.is_dir():
        raise ChecksumError(f"{root} is not a directory — nothing to checksum")

    known = dict(recorded or {})
    digests: dict[str, str] = {}
    for rel, path in _relative_paths(root):
        for bad in _ILLEGAL_IN_PATH:
            if bad in rel:
                raise ChecksumError(
                    f"{rel!r} contains {bad!r}, which sha256sum cannot represent "
                    f"unambiguously — rename it before checksumming {root}"
                )
        supplied = known.get(rel)
        if supplied is not None:
            if not _HEX.match(supplied):
                raise ChecksumError(
                    f"recorded digest for {rel!r} is {supplied!r}, "
                    "expected 64 lowercase hex characters"
                )
            digests[rel] = supplied
        else:
            digests[rel] = file_digest(path)
    return digests


def iter_lines(digests: Mapping[str, str]) -> Iterator[str]:
    """Render manifest lines in sorted path order, one per file.

    The single definition of the on-disk format: :func:`write_checksums` streams this
    into a file handle and tests join it, so there is no second renderer to drift.
    """
    for rel in sorted(digests):
        yield f"{digests[rel]}  {rel}\n"


def parse_checksums(text: str) -> dict[str, str]:
    """Parse ``sha256sum`` manifest text into a path-to-digest map.

    Strict by intent: a line this cannot parse raises rather than being skipped, because
    a silently dropped line is a file that quietly stops being verified.
    """
    digests: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LINE.match(line)
        if match is None:
            raise ChecksumError(f"line {number} is not '<64 hex><two spaces><path>': {line!r}")
        digest, rel = match.group(1), match.group(2)
        if rel in digests:
            raise ChecksumError(f"line {number} repeats path {rel!r}")
        digests[rel] = digest
    return digests


def read_checksums(path: str | Path) -> dict[str, str] | None:
    """Read a manifest, or ``None`` if there is not one yet.

    A missing manifest is the ordinary state of a version dir that has never been
    written; it is not an error.
    """
    p = Path(path)
    if not p.exists():
        return None
    return parse_checksums(p.read_text(encoding="utf-8"))


def _describe(label: str, paths: Iterable[str], limit: int = 5) -> str:
    listed = sorted(paths)
    if not listed:
        return ""
    shown = ", ".join(repr(p) for p in listed[:limit])
    more = f" (+{len(listed) - limit} more)" if len(listed) > limit else ""
    return f"{label}: {shown}{more}"


def assert_unchanged(
    existing: Mapping[str, str], current: Mapping[str, str], *, root: Path
) -> None:
    """Raise unless ``current`` reproduces ``existing`` exactly.

    A version is immutable, so all three ways a tree can differ from its manifest —
    a file's bytes changed, a file appeared, a file vanished — are equally violations.
    Re-running over an untouched tree is a no-op and must stay silent, which is why this
    compares rather than refusing outright when a manifest is present.
    """
    changed = {
        rel for rel, digest in current.items() if rel in existing and existing[rel] != digest
    }
    added = set(current) - set(existing)
    removed = set(existing) - set(current)
    if not (changed or added or removed):
        return
    details = [
        d
        for d in (
            _describe("changed", changed),
            _describe("added", added),
            _describe("removed", removed),
        )
        if d
    ]
    raise ChecksumError(
        f"{root} already has a {CHECKSUM_FILE} that disagrees with the tree — "
        f"a published version is immutable, so this is a new version, not a rewrite. "
        + "; ".join(details)
    )


def write_checksums(
    root: str | Path, *, recorded: Mapping[str, str] | None = None
) -> ChecksumManifest:
    """Write ``<root>/CHECKSUMS.sha256`` for a staged version directory.

    Idempotent on an unchanged tree and fatal on a changed one — see
    :func:`assert_unchanged`. Lines are streamed to the file as they are rendered, and
    the returned ``root_digest`` is read back off the finished file, so the digest is of
    the bytes that are actually on disk rather than of what was meant to be written.
    """
    root = Path(root)
    digests = scan_digests(root, recorded=recorded)
    target = root / CHECKSUM_FILE

    existing = read_checksums(target)
    if existing is not None:
        assert_unchanged(existing, digests, root=root)

    size_bytes = sum((root / rel).stat().st_size for rel in digests)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(iter_lines(digests))

    return ChecksumManifest(
        path=target,
        root_digest=file_digest(target),
        files=len(digests),
        size_bytes=size_bytes,
    )
