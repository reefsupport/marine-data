"""Resumable, streaming S3 uploads + the D-F disk guard (WP-6).

* Multipart at ``part_size`` (64 MiB default); memory is bounded by one part.
* Skip: an object already present with equal size AND (equal ETag OR equal
  ``x-amz-meta-sha256``) is not re-sent (D-G "skip keys already present").
* Resume: a JSON checkpoint per (bucket, key) holds the multipart ``UploadId``; on restart
  ``list_parts`` is the authority (a part sent before a kill but after the last checkpoint
  write is still reused), and a part is reused only if its ETag equals the md5 of the
  local bytes at that offset — so a changed file can never splice into an old upload.
* Verify: after completion ``head_object`` must report the locally computed size and
  ETag (md5 for single-part, ``md5(concat part md5s)-N`` for multipart).

The client is any boto3-compatible S3 client; :func:`client_from_rclone` builds one from
``~/.config/rclone/rclone.conf`` without ever printing or logging the credentials.
"""

from __future__ import annotations

import configparser
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MiB = 1 << 20
GiB = 1 << 30
DEFAULT_PART = 64 * MiB


class DiskFloorError(RuntimeError):
    """Free disk is below the D-F floor: refuse to write more."""


class TempCapError(RuntimeError):
    """Writing would take this worker's temp dir past its cap (D-F/D-L)."""


@dataclass
class DiskGuard:
    """``check`` refuses below the free-space floor; ``reserve`` enforces the temp cap and
    records the peak temp footprint (asserted by tests and reported by the CLI)."""

    temp_root: Path
    temp_cap_bytes: int = 6 * GiB
    floor_bytes: int = 40 * GiB
    peak_bytes: int = 0

    def free_bytes(self) -> int:
        self.temp_root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.temp_root).free

    def temp_used(self) -> int:
        total = 0
        for dirpath, _, names in os.walk(self.temp_root):
            for n in names:
                try:
                    total += (Path(dirpath) / n).stat().st_size
                except FileNotFoundError:
                    continue
        return total

    def check(self) -> None:
        free = self.free_bytes()
        if free < self.floor_bytes:
            raise DiskFloorError(
                f"free {free / GiB:.1f} GiB < floor {self.floor_bytes / GiB:.1f} GiB — pause"
            )

    def observe(self, nbytes: int) -> None:
        """Record a caller-tracked live-bytes figure (cheap; no directory walk)."""
        self.peak_bytes = max(self.peak_bytes, nbytes)

    def sample(self) -> int:
        used = self.temp_used()
        self.peak_bytes = max(self.peak_bytes, used)
        return used

    def reserve(self, nbytes: int) -> None:
        """Refuse if ``nbytes`` more would exceed the cap, or the disk is below the floor."""
        self.check()
        used = self.sample()
        if used + nbytes > self.temp_cap_bytes:
            raise TempCapError(f"temp {used + nbytes} B would exceed cap {self.temp_cap_bytes} B")


@dataclass(frozen=True)
class LocalDigest:
    size: int
    sha256: str
    etag: str
    part_md5s: tuple[str, ...]


def local_digest(path: Path, part_size: int = DEFAULT_PART) -> LocalDigest:
    """One read pass: sha256, per-part md5s, and the ETag S3 will report."""
    sha = hashlib.sha256()
    md5s: list[str] = []
    size = 0
    with path.open("rb") as fh:
        while chunk := fh.read(part_size):
            sha.update(chunk)
            md5s.append(hashlib.md5(chunk, usedforsecurity=False).hexdigest())
            size += len(chunk)
    if size <= part_size:
        etag = md5s[0] if md5s else hashlib.md5(b"", usedforsecurity=False).hexdigest()
    else:
        joined = b"".join(bytes.fromhex(m) for m in md5s)
        etag = f"{hashlib.md5(joined, usedforsecurity=False).hexdigest()}-{len(md5s)}"
    return LocalDigest(size, sha.hexdigest(), etag, tuple(md5s))


@dataclass(frozen=True)
class UploadResult:
    key: str
    size: int
    etag: str
    sha256: str
    skipped: bool
    parts_sent: int


def _head(client: Any, bucket: str, key: str) -> dict | None:
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # botocore ClientError 404/NoSuchKey
        code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def remote_matches(client: Any, bucket: str, key: str, d: LocalDigest) -> bool:
    head = _head(client, bucket, key)
    if head is None or int(head["ContentLength"]) != d.size:
        return False
    etag = str(head.get("ETag", "")).strip('"')
    return etag == d.etag or (head.get("Metadata") or {}).get("sha256") == d.sha256


def _checkpoint_path(checkpoint_dir: Path, bucket: str, key: str) -> Path:
    name = hashlib.sha1(f"{bucket}/{key}".encode(), usedforsecurity=False).hexdigest()
    return checkpoint_dir / f"{name}.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True))
    tmp.replace(path)


def _remote_parts(client: Any, bucket: str, key: str, upload_id: str) -> dict[int, str] | None:
    parts: dict[int, str] = {}
    marker = 0
    while True:
        try:
            resp = client.list_parts(
                Bucket=bucket, Key=key, UploadId=upload_id, PartNumberMarker=marker
            )
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in {"NoSuchUpload", "404"}:
                return None
            raise
        for p in resp.get("Parts") or []:
            parts[int(p["PartNumber"])] = str(p["ETag"]).strip('"')
        if not resp.get("IsTruncated"):
            return parts
        marker = int(resp["NextPartNumberMarker"])


def upload_file(
    client: Any,
    bucket: str,
    key: str,
    path: Path,
    *,
    checkpoint_dir: Path,
    part_size: int = DEFAULT_PART,
    digest: LocalDigest | None = None,
) -> UploadResult:
    """Upload ``path`` to ``s3://bucket/key`` resumably; verify; return the facts."""
    d = digest or local_digest(path, part_size)
    if remote_matches(client, bucket, key, d):
        return UploadResult(key, d.size, d.etag, d.sha256, skipped=True, parts_sent=0)
    meta = {"sha256": d.sha256}
    if d.size <= part_size:
        with path.open("rb") as fh:
            client.put_object(Bucket=bucket, Key=key, Body=fh, Metadata=meta)
        _verify(client, bucket, key, d)
        return UploadResult(key, d.size, d.etag, d.sha256, skipped=False, parts_sent=1)

    ckpt = _checkpoint_path(checkpoint_dir, bucket, key)
    identity = {
        "bucket": bucket,
        "key": key,
        "size": d.size,
        "sha256": d.sha256,
        "part_size": part_size,
    }
    done: dict[int, str] = {}
    upload_id: str | None = None
    if ckpt.exists():
        saved = json.loads(ckpt.read_text())
        if {k: saved.get(k) for k in identity} == identity:
            remote = _remote_parts(client, bucket, key, saved["upload_id"])
            if remote is not None:
                upload_id = saved["upload_id"]
                done = {
                    n: e
                    for n, e in remote.items()
                    if n <= len(d.part_md5s) and e == d.part_md5s[n - 1]
                }
    if upload_id is None:
        upload_id = client.create_multipart_upload(Bucket=bucket, Key=key, Metadata=meta)[
            "UploadId"
        ]
        _write_json_atomic(ckpt, {**identity, "upload_id": upload_id, "parts": {}})
    sent = 0
    with path.open("rb") as fh:
        for n, md5 in enumerate(d.part_md5s, start=1):
            if n in done:
                continue
            fh.seek((n - 1) * part_size)
            body = fh.read(part_size)
            etag = str(
                client.upload_part(
                    Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=n, Body=body
                )["ETag"]
            ).strip('"')
            if etag != md5:
                raise RuntimeError(f"{key} part {n}: ETag {etag} != local md5 {md5}")
            done[n] = etag
            sent += 1
            _write_json_atomic(
                ckpt,
                {**identity, "upload_id": upload_id, "parts": {str(k): v for k, v in done.items()}},
            )
    client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [{"PartNumber": n, "ETag": f'"{done[n]}"'} for n in sorted(done)]
        },
    )
    _verify(client, bucket, key, d)
    ckpt.unlink(missing_ok=True)
    return UploadResult(key, d.size, d.etag, d.sha256, skipped=False, parts_sent=sent)


def _verify(client: Any, bucket: str, key: str, d: LocalDigest) -> None:
    if not remote_matches(client, bucket, key, d):
        raise RuntimeError(f"verify failed for s3://{bucket}/{key}: size/ETag mismatch")


def client_from_rclone(remote: str = "rs-hel1", conf: Path | None = None) -> Any:
    """boto3 S3 client from an rclone remote. Credentials are never printed or logged."""
    import boto3
    from botocore.config import Config

    parser = configparser.ConfigParser()
    parser.read(conf or Path.home() / ".config/rclone/rclone.conf")
    if remote not in parser:
        raise KeyError(f"rclone remote {remote!r} not found")
    sec = parser[remote]
    return boto3.client(
        "s3",
        endpoint_url=sec.get("endpoint"),
        region_name=sec.get("region") or None,
        aws_access_key_id=sec.get("access_key_id"),
        aws_secret_access_key=sec.get("secret_access_key"),
        config=Config(
            retries={"max_attempts": 8, "mode": "standard"},
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )
