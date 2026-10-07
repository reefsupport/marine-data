"""Verify-only stage of ``ingest-batch --verify-only``: for a version whose CHECKSUMS.sha256
marker is already in the bucket (a run that died in ``verify`` or was ``skip-done``), HEAD
every object against the manifest and write ``registry-stub-<id>.yaml`` exactly as the
streaming run's tail does. Read-only: nothing is uploaded, deleted or overwritten."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import checksums
from .concurrency import retry_with_backoff, tag_s3_key
from .ingest_source import IngestReport, IngestSpec, _stub
from .ingest_stream import MemoryBudget, _list_existing, _Putter

VERIFY_JOBS = 16


def _get(client: Any, bucket: str, key: str) -> bytes:
    return tag_s3_key(
        key,
        lambda: retry_with_backoff(
            lambda: client.get_object(Bucket=bucket, Key=key)["Body"].read(),
            retries=6,
            base=1.0,
            cap=30.0,
        ),
    )


def verify_staged(
    spec: IngestSpec, version: str, work: Path, client: Any, *, jobs: int = VERIFY_JOBS
) -> IngestReport:
    """Verify ``spec.prefix/<id>/<version>`` against its CHECKSUMS and write the stub."""
    prefix = f"{spec.prefix}/{spec.id}/{version}"
    manifest = _get(client, spec.bucket, f"{prefix}/{checksums.CHECKSUM_FILE}")
    digests = checksums.parse_checksums(manifest.decode())
    info = json.loads(_get(client, spec.bucket, f"{prefix}/INGEST.json"))
    listed = _list_existing(client, spec.bucket, prefix)
    gone = sorted(set(digests) - set(listed))
    extra = sorted(set(listed) - set(digests) - {checksums.CHECKSUM_FILE})
    if gone or extra or listed.get(checksums.CHECKSUM_FILE, (None,))[0] != len(manifest):
        raise RuntimeError(f"verify-only {prefix}: unlisted={gone[:3]} unexpected={extra[:3]}")
    root = hashlib.sha256(manifest).hexdigest()
    files = {rel: (sha, listed[rel][0]) for rel, sha in digests.items()}
    files[checksums.CHECKSUM_FILE] = (root, len(manifest))
    putter = _Putter(client, spec.bucket, prefix, MemoryBudget(1), 1, {})
    try:
        verified = putter.verify(files, jobs=jobs)
    finally:
        putter.close(wait=True)
    report = IngestReport(
        source_id=spec.id,
        version=version,
        prefix=prefix,
        layout=str(info["layout"]),
        images=int(info["images"]),
        files=len(files),
        bytes=sum(size for _, size in files.values()),
        verified=verified,
        missing=int(info.get("missing_items", 0)),
        root_digest=root,
    )
    report.plan["mode"] = "verify-only"
    work.mkdir(parents=True, exist_ok=True)
    (work / f"registry-stub-{spec.id}.yaml").write_text(_stub(spec, report))
    return report
