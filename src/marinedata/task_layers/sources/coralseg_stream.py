"""Coralseg restage, streamed (WP-8e): legacy prefix -> D-D staged tree, no image bytes on disk.

WP-8d's :func:`marinedata.task_layers.sources.coralseg.restage` stages every file to a
local work dir before uploading, and its DiskGuard floor stopped it before a byte moved.
This variant keeps the same layout, stems, ``metadata.parquet`` rows and
``CHECKSUMS.sha256`` (it reuses :class:`~marinedata.staged_writer.StagedWriter` and only
swaps the file sink): every image and mask is GET into memory, hashed, and PUT straight to
``sources/<SOURCE_ID>/<version>/``. Only the two small manifests touch the work dir.

After the manifest (uploaded LAST, the version-complete marker) every expected object is
HEAD-checked for existence and exact size (:func:`verify_heads`).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ... import checksums
from ...concurrency import head_with_retry
from . import coralseg as base


class _S3SinkWriter(base.StagedWriter):
    """StagedWriter whose files go to S3 from memory instead of ``root`` on disk."""

    def __init__(
        self,
        root: Path,
        cfg: Any,
        client: Any,
        bucket: str,
        key_prefix: str,
        pool: ThreadPoolExecutor,
    ) -> None:
        super().__init__(root, cfg)
        self.client, self.bucket, self.key_prefix, self.pool = client, bucket, key_prefix, pool
        self.pending: list[Future] = []

    def _put(self, rel: str, data: bytes) -> None:
        self.pending.append(
            self.pool.submit(
                self.client.put_object,
                Bucket=self.bucket,
                Key=f"{self.key_prefix}/{rel}",
                Body=data,
                ContentLength=len(data),
            )
        )

    def _write_file(self, rel: str, data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        self.files[rel] = (sha, len(data))
        self._put(rel, data)
        return sha

    def register(self, path: Path) -> None:  # small manifests only (metadata.parquet)
        rel = path.relative_to(self.root).as_posix()
        data = path.read_bytes()
        self.files[rel] = (hashlib.sha256(data).hexdigest(), len(data))
        self._put(rel, data)

    def wait(self) -> None:
        pending, self.pending = self.pending, []
        for fut in pending:
            fut.result()


def verify_heads(
    client: Any, bucket: str, key_prefix: str, files: dict[str, int], workers: int = 16
) -> list[str]:
    """HEAD every ``key_prefix/rel`` and return the rels that are missing or mis-sized."""

    def bad(item: tuple[str, int]) -> str | None:
        rel, size = item
        try:
            head = head_with_retry(client, bucket, f"{key_prefix}/{rel}")
        except Exception:
            return rel
        return None if int(head["ContentLength"]) == size else rel

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(r for r in pool.map(bad, sorted(files.items())) if r is not None)


def restage_streaming(
    client: Any,
    work: Path,
    *,
    version: str = base.VERSION,
    fetch_date: dt.date | None = None,
    limit: int | None = None,
    chunk: int = 32,
    workers: int = 16,
) -> dict[str, Any]:
    pairs = base.list_pairs(client)
    if limit is not None:
        pairs = pairs[:limit]
    bucket = base.LEGACY_BUCKET
    key_prefix = f"{base.DEST_PREFIX}/{base.SOURCE_ID}/{version}"
    root = work / "stage" / base.SOURCE_ID / version
    root.mkdir(parents=True, exist_ok=True)
    cfg = base.WriterConfig(
        base.SOURCE_ID,
        version,
        base.LICENSE,
        base.ATTRIBUTION,
        fetch_date or dt.date.today(),
        split_group=base.SPLIT_GROUP,
    )
    class_counts: dict[str, dict[str, int]] = {}

    def get(pair: base.Pair) -> tuple[base.Pair, bytes, bytes]:
        body = client.get_object(Bucket=bucket, Key=pair.image_key)["Body"].read()
        mask = client.get_object(Bucket=bucket, Key=pair.mask_key)["Body"].read()
        return pair, body, mask

    with (
        ThreadPoolExecutor(max_workers=workers) as fetch_pool,
        ThreadPoolExecutor(max_workers=workers) as put_pool,
    ):
        writer = _S3SinkWriter(root, cfg, client, bucket, key_prefix, put_pool)
        for start in range(0, len(pairs), chunk):
            for pair, image, mask in fetch_pool.map(get, pairs[start : start + chunk]):
                item = base.RemoteItem(key=pair.image_key, url=f"s3://{bucket}/{pair.image_key}")
                row = writer.add(
                    item,
                    base.Decoded(
                        upstream_id=f"{pair.split}/{pair.stem}",
                        data=image,
                        suffix=Path(pair.image_key).suffix,
                        upstream_url=item.url,
                        split_hint=pair.split,
                        label_files={f"{pair.stem}.png": mask},
                    ),
                )
                writer.finish_item(None)
                if row is not None:
                    class_counts[row.image_sha256] = base.class_counts_from_mask(mask)
            writer.wait()
        writer.finalize()
        base.write_samples(root / "metadata.parquet", writer.rows)
        writer.register(root / "metadata.parquet")
        writer.wait()
        digests = {rel: sha for rel, (sha, _) in writer.files.items()}
        manifest = "".join(checksums.iter_lines(digests)).encode()
        (root / checksums.CHECKSUM_FILE).write_bytes(manifest)
        writer._put(checksums.CHECKSUM_FILE, manifest)  # LAST: the version-complete marker
        writer.wait()
    sizes = {rel: size for rel, (_, size) in writer.files.items()}
    sizes[checksums.CHECKSUM_FILE] = len(manifest)
    failed = verify_heads(client, bucket, key_prefix, sizes, workers)
    return {
        "key_prefix": key_prefix,
        "images": len(writer.rows),
        "objects": len(sizes),
        "bytes": sum(sizes.values()),
        "root_digest": hashlib.sha256(manifest).hexdigest(),
        "head_failed": failed,
        "rows": writer.rows,
        "class_counts": class_counts,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True, help="dir for the two small manifests")
    p.add_argument("--out", type=Path, required=True, help="the coralseg semseg.parquet to write")
    p.add_argument("--limit", type=int, default=None, help="cap pairs (testing only)")
    p.add_argument("--version", default=base.VERSION)
    args = p.parse_args(argv)
    client = base.client_from_rclone("rs-hel1")
    result = restage_streaming(client, args.work, version=args.version, limit=args.limit)
    n = base.write_semseg_parquet(result.pop("rows"), result.pop("class_counts"), args.out)
    print(json.dumps({**result, "head_failed": len(result["head_failed"]), "semseg_rows": n}))
    return 0 if not result["head_failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
