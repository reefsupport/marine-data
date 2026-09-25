"""``marinedata verify-release`` — check a built Hub release against its ``MANIFEST.tsv``
(WP-3, format hardening after S59).

Two targets:

* **local** ``<dir>``: re-hash every file on disk and compare size + sha256 + (for a
  Parquet file) row count to the manifest. A manifest entry with no file on disk is a
  failure; a file on disk not in the manifest is a warning, or a failure under
  ``--strict``.
* **S3** ``s3://bucket/prefix``: a flat, paginated ``list_objects_v2`` (no download) compared
  by size + ETag only. ``--deep N`` (or ``--deep all``) additionally downloads N objects,
  chosen at random, to a caller-given scratch directory, checks their sha256 against the
  manifest, and deletes each file immediately after — never more than one object on disk
  at a time.

Named ``verify-release`` rather than ``verify``: the CLI already has a ``verify`` command
(``_cmd_verify``, registry layouts against fetched samples) — a different check entirely,
disclosed in the WP-3 report rather than colliding on the name.
"""

from __future__ import annotations

import argparse
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .manifest import MANIFEST_NAME, ManifestRow, digest_file, parquet_fingerprint, read_manifest


class VerifyError(RuntimeError):
    """A verify target could not be checked at all (bad URL, unreadable manifest, ...)."""


@dataclass
class VerifyReport:
    target: str
    checked: int = 0
    missing: list[str] = field(default_factory=list)
    mismatched: list[str] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    deep_checked: int = 0
    deep_failed: list[str] = field(default_factory=list)

    def ok(self, *, strict: bool) -> bool:
        if self.missing or self.mismatched or self.deep_failed:
            return False
        return not (strict and self.unlisted)

    def summary(self, *, strict: bool) -> str:
        status = "OK" if self.ok(strict=strict) else "FAIL"
        return (
            f"verify {status}: {self.target} — {self.checked} checked, "
            f"{len(self.missing)} missing, {len(self.mismatched)} mismatched, "
            f"{len(self.unlisted)} unlisted, {self.deep_checked} deep-checked "
            f"({len(self.deep_failed)} deep-failed)"
        )


def _local_files(root: Path) -> set[str]:
    return {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != MANIFEST_NAME
    }


def verify_local(root: Path, manifest: dict[str, ManifestRow]) -> VerifyReport:
    root = root.resolve()
    report = VerifyReport(target=str(root))
    on_disk = _local_files(root)
    for rel, row in sorted(manifest.items()):
        report.checked += 1
        path = root / rel
        if not path.is_file():
            report.missing.append(rel)
            continue
        d = digest_file(path)
        if d.size != row.size or d.sha256 != row.sha256:
            report.mismatched.append(rel)
            continue
        if row.rows is not None:
            pq_info = parquet_fingerprint(path)
            if pq_info is None or pq_info[0] != row.rows:
                report.mismatched.append(rel)
    report.unlisted = sorted(on_disk - set(manifest))
    return report


def _parse_s3_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise VerifyError(f"not an s3:// URL: {url!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _list_s3(client: Any, bucket: str, prefix: str) -> Iterator[tuple[str, int, str]]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], int(obj["Size"]), str(obj["ETag"]).strip('"')


def verify_s3(
    url: str,
    manifest: dict[str, ManifestRow],
    *,
    client: Any,
    deep: int | str | None = None,
    scratch: Path | None = None,
) -> VerifyReport:
    bucket, prefix = _parse_s3_url(url)
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    report = VerifyReport(target=url)
    listed = {
        key[len(prefix) :]: (size, etag)
        for key, size, etag in _list_s3(client, bucket, prefix)
        if key != prefix + MANIFEST_NAME
    }
    for rel, row in sorted(manifest.items()):
        report.checked += 1
        if rel not in listed:
            report.missing.append(rel)
            continue
        size, etag = listed[rel]
        if size != row.size or etag != row.s3_etag:
            report.mismatched.append(rel)
    report.unlisted = sorted(set(listed) - set(manifest))

    if deep:
        candidates = [r for r in manifest if r not in report.missing and r not in report.mismatched]
        n = len(candidates) if deep == "all" else min(int(deep), len(candidates))
        sample = random.sample(candidates, n) if n else []
        scratch = scratch or Path.cwd()
        scratch.mkdir(parents=True, exist_ok=True)
        for rel in sample:
            row = manifest[rel]
            dest = scratch / Path(rel).name
            try:
                client.download_file(bucket, prefix + rel, str(dest))
                d = digest_file(dest)
                report.deep_checked += 1
                if d.sha256 != row.sha256 or d.size != row.size:
                    report.deep_failed.append(rel)
            finally:
                dest.unlink(missing_ok=True)
    return report


def verify(
    target: str,
    *,
    manifest_path: Path | None = None,
    deep: int | str | None = None,
    strict: bool = False,
    scratch: Path | None = None,
    s3_client_factory: Any = None,
) -> VerifyReport:
    """Dispatch on ``target`` (``s3://...`` or a local directory)."""
    if target.startswith("s3://"):
        if manifest_path is None:
            raise VerifyError("--manifest is required for an s3:// target")
        manifest = read_manifest(manifest_path)
        if s3_client_factory is None:
            from .s3_client import client_from_rclone

            s3_client_factory = client_from_rclone
        return verify_s3(target, manifest, client=s3_client_factory(), deep=deep, scratch=scratch)
    root = Path(target)
    manifest = read_manifest(manifest_path or root / MANIFEST_NAME)
    return verify_local(root, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m marinedata.verify_release")
    parser.add_argument("target", help="a local directory or an s3://bucket/prefix")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--deep", default=None, help="N random objects, or 'all' (s3 only)")
    parser.add_argument("--strict", action="store_true", help="unlisted files also fail")
    parser.add_argument("--scratch", type=Path, default=None, help="deep-check download dir")
    args = parser.parse_args(argv)
    deep: int | str | None = args.deep
    if deep is not None and deep != "all":
        deep = int(deep)
    report = verify(
        args.target,
        manifest_path=args.manifest,
        deep=deep,
        strict=args.strict,
        scratch=args.scratch,
    )
    print(report.summary(strict=args.strict))
    return 0 if report.ok(strict=args.strict) else 1


if __name__ == "__main__":
    raise SystemExit(main())
