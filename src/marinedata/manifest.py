"""``MANIFEST.tsv`` for a built release tree (WP-3, format hardening after S59).

One streaming pass per file computes three independent checks in one read: a sha256 (the
content-addressed truth), the S3 ETag the object will carry once uploaded (single-part md5,
or the md5-of-md5s ``-N`` multipart form at 64 MiB parts — the same convention
``s3_upload.local_digest`` uses elsewhere in this package, reimplemented here since that
module is not yet on this branch), and — for a Parquet file — its row count and a schema
fingerprint. Memory is bounded by one 64 MiB part, never the file size.

:func:`build_manifest` writes the file at ``<root>/MANIFEST.tsv``: columns ``path`` (POSIX,
relative to ``root``), ``size``, ``sha256``, ``s3_etag``, ``rows`` and ``schema_fingerprint``
(the last two blank for a non-Parquet file). ``MANIFEST.tsv`` never lists itself.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

MiB = 1 << 20
PART_SIZE = 64 * MiB
MANIFEST_NAME = "MANIFEST.tsv"
HEADER = ("path", "size", "sha256", "s3_etag", "rows", "schema_fingerprint")


@dataclass(frozen=True)
class FileDigest:
    size: int
    sha256: str
    s3_etag: str


def digest_file(path: Path, *, part_size: int = PART_SIZE) -> FileDigest:
    """One read pass: sha256 over the whole file, plus the ETag S3 reports (md5 for a
    single part, ``md5(concat part md5s)-N`` for more than one 64 MiB part)."""
    sha = hashlib.sha256()
    part_md5s: list[str] = []
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(part_size):
            sha.update(chunk)
            part_md5s.append(hashlib.md5(chunk, usedforsecurity=False).hexdigest())
            size += len(chunk)
    if len(part_md5s) <= 1:
        etag = part_md5s[0] if part_md5s else hashlib.md5(b"", usedforsecurity=False).hexdigest()
    else:
        joined = b"".join(bytes.fromhex(m) for m in part_md5s)
        etag = f"{hashlib.md5(joined, usedforsecurity=False).hexdigest()}-{len(part_md5s)}"
    return FileDigest(size, sha.hexdigest(), etag)


def parquet_fingerprint(path: Path) -> tuple[int, str] | None:
    """``(row_count, schema_fingerprint)`` for a Parquet file, else ``None``. The fingerprint
    is a sha256 over the schema's string form — stable across files with identical columns
    regardless of row-group layout."""
    if path.suffix != ".parquet":
        return None
    try:
        meta = pq.ParquetFile(path)
    except Exception:
        return None
    rows = meta.metadata.num_rows
    fingerprint = hashlib.sha256(str(meta.schema_arrow).encode("utf-8")).hexdigest()
    return rows, fingerprint


def iter_files(root: Path) -> Iterator[Path]:
    """Every regular file under ``root``, sorted by POSIX-relative path, excluding an
    existing ``MANIFEST.tsv`` at the root."""
    paths = [p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_NAME]
    return iter(sorted(paths, key=lambda p: p.relative_to(root).as_posix()))


def build_manifest(root: Path, out: Path | None = None) -> Path:
    """Write ``<root>/MANIFEST.tsv`` (or ``out``) and return its path. Streams one row at a
    time to disk — RAM use is bounded by one file's digest state, never the corpus size."""
    root = root.resolve()
    dest = out.resolve() if out else root / MANIFEST_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fh:
        fh.write("\t".join(HEADER) + "\n")
        for path in iter_files(root):
            d = digest_file(path)
            pq_info = parquet_fingerprint(path)
            rows = str(pq_info[0]) if pq_info else ""
            fingerprint = pq_info[1] if pq_info else ""
            rel = path.relative_to(root).as_posix()
            fh.write(f"{rel}\t{d.size}\t{d.sha256}\t{d.s3_etag}\t{rows}\t{fingerprint}\n")
    return dest


@dataclass(frozen=True)
class ManifestRow:
    path: str
    size: int
    sha256: str
    s3_etag: str
    rows: int | None
    schema_fingerprint: str | None


def read_manifest(path: Path) -> dict[str, ManifestRow]:
    """Parse a ``MANIFEST.tsv`` into ``{relative_path: ManifestRow}``."""
    rows: dict[str, ManifestRow] = {}
    with path.open(encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if tuple(header) != HEADER:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            p, size, sha256, s3_etag, row_count, fingerprint = line.split("\t")
            rows[p] = ManifestRow(
                p,
                int(size),
                sha256,
                s3_etag,
                int(row_count) if row_count else None,
                fingerprint or None,
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m marinedata.manifest")
    parser.add_argument("dir", type=Path, help="build root to manifest")
    parser.add_argument("--out", type=Path, default=None, help="default: <dir>/MANIFEST.tsv")
    args = parser.parse_args(argv)
    dest = build_manifest(args.dir, args.out)
    n = sum(1 for _ in read_manifest(dest))
    print(f"manifest: {n} files -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
