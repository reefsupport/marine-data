"""Coralseg (UCSD) restage into D-D + ``data/_tasklabels/coralseg/semseg.parquet`` (WP-8d, D-Z2).

Restages the legacy prefix ``benthic_datasets/mask_labels/Coralseg/<split>/{Image,Mask}/
<stem>.{jpg,png}`` into the D-D staged-tree layout, ``sources/coralseg/<version>/{images/,
labels/files/,metadata.parquet,CHECKSUMS.sha256}``, reusing the existing staging primitives
(:class:`~marinedata.staged_writer.StagedWriter`, :class:`~marinedata.ingest_source._Uploader`,
:mod:`marinedata.checksums`) rather than inventing a new upload path. D-Z2 (the 2026-09-25
charter, superseding D-Z's "never the images" for Coralseg only) authorizes opening and
restaging Coralseg's images so masks can be keyed by the sha256 the staging step itself
computes — see ``docs/task-layers.md``'s WP-8b finding this resolves.

Each mask's class counts come from its red channel (values 0/1 only), per the registry's
already-verified ``coralseg-r-channel`` loader note.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ... import checksums
from ...adapters import Decoded, RemoteItem
from ...ingest_source import IngestReport, IngestSpec, _Uploader
from ...s3_upload import DiskGuard, GiB, client_from_rclone
from ...sample_schema import SampleRow, write_samples
from ...staged_writer import StagedWriter, WriterConfig

SOURCE_ID = "coralseg-ucsd-mosaics"
LEGACY_BUCKET = "rs-storage-open"
LEGACY_PREFIX = "benthic_datasets/mask_labels/Coralseg/"
DEST_PREFIX = "sources"
VERSION = "unversioned"
LICENSE = "NO-LICENCE-STATED"
ATTRIBUTION = "Coralseg (UCSD)"
LABEL_ORIGIN = "human"  # no label-origin.yaml exists in this repo yet; see docs note.


@dataclass(frozen=True)
class Pair:
    split: str
    stem: str
    image_key: str
    mask_key: str


def list_pairs(client: Any, bucket: str = LEGACY_BUCKET, prefix: str = LEGACY_PREFIX) -> list[Pair]:
    """Flat ``list_objects_v2`` paginator only (creds rule) — pair Image/Mask by stem."""
    images: dict[tuple[str, str], str] = {}
    masks: dict[tuple[str, str], str] = {}
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix) :]
            parts = rel.split("/")
            if len(parts) != 3:
                continue
            split, kind, fname = parts
            stem = Path(fname).stem
            if kind == "Image":
                images[(split, stem)] = obj["Key"]
            elif kind == "Mask":
                masks[(split, stem)] = obj["Key"]
    keys = sorted(k for k in images if k in masks)
    return [Pair(k[0], k[1], images[k], masks[k]) for k in keys]


def class_counts_from_mask(data: bytes) -> dict[str, int]:
    """Red-channel histogram (0/1 only, per the registered ``coralseg-r-channel`` loader)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        hist = im.convert("RGB").getchannel("R").histogram()
    return {str(i): n for i, n in enumerate(hist) if n}


def restage(
    client: Any,
    work: Path,
    *,
    version: str = VERSION,
    fetch_date: dt.date | None = None,
    part_size: int = 64 * (1 << 20),
    guard: DiskGuard | None = None,
    limit: int | None = None,
) -> tuple[IngestReport, dict[str, dict[str, int]], list[SampleRow]]:
    """Stream Coralseg pairs into the D-D tree, uploading + deleting each closed batch."""
    guard = guard or DiskGuard(work, temp_cap_bytes=4 * GiB, floor_bytes=34 * GiB)
    guard.check()
    pairs = list_pairs(client)
    if limit is not None:
        pairs = pairs[:limit]
    key_prefix = f"{DEST_PREFIX}/{SOURCE_ID}/{version}"
    report = IngestReport(SOURCE_ID, version, key_prefix, "objects")
    root = work / "stage" / SOURCE_ID / version
    writer = StagedWriter(
        root,
        WriterConfig(SOURCE_ID, version, LICENSE, ATTRIBUTION, fetch_date or dt.date.today()),
    )
    spec = IngestSpec(
        id=SOURCE_ID,
        adapter="bucket",
        params={},
        license=LICENSE,
        attribution=ATTRIBUTION,
        version=version,
        bucket=LEGACY_BUCKET,
    )
    up = _Uploader(client, spec, key_prefix, root, work, part_size, report)
    class_counts: dict[str, dict[str, int]] = {}
    flush_at = guard.temp_cap_bytes // 2
    live = 0

    def flush() -> None:
        nonlocal live
        guard.sample()
        for rel in writer.drain_closed():
            up.push(rel)
        live = guard.sample()

    for pair in pairs:
        guard.check()
        image_bytes = client.get_object(Bucket=LEGACY_BUCKET, Key=pair.image_key)["Body"].read()
        mask_bytes = client.get_object(Bucket=LEGACY_BUCKET, Key=pair.mask_key)["Body"].read()
        item = RemoteItem(key=pair.image_key, url=f"s3://{LEGACY_BUCKET}/{pair.image_key}")
        decoded = Decoded(
            upstream_id=f"{pair.split}/{pair.stem}",
            data=image_bytes,
            suffix=Path(pair.image_key).suffix,
            upstream_url=item.url,
            split_hint=pair.split,
            label_files={f"{pair.stem}.png": mask_bytes},
        )
        row = writer.add(item, decoded)
        writer.finish_item(None)
        live += len(image_bytes) + len(mask_bytes)
        guard.observe(live)
        if row is not None:
            class_counts[row.image_sha256] = class_counts_from_mask(mask_bytes)
        if live >= flush_at:
            flush()
    writer.finalize()
    rows = writer.rows
    report.images = len(rows)
    write_samples(root / "metadata.parquet", rows)
    writer.register(root / "metadata.parquet")
    flush()
    digests = {rel: sha for rel, (sha, _) in writer.files.items()}
    manifest = "".join(checksums.iter_lines(digests)).encode()
    (root / checksums.CHECKSUM_FILE).write_bytes(manifest)
    report.root_digest = hashlib.sha256(manifest).hexdigest()
    report.files = len(digests) + 1
    report.bytes = sum(size for _, size in writer.files.values()) + len(manifest)
    up.push(checksums.CHECKSUM_FILE)  # LAST: the version-complete marker
    report.verified = up.verify(sorted([*digests, checksums.CHECKSUM_FILE]))
    report.peak_temp_bytes = guard.peak_bytes
    return report, class_counts, rows


def write_semseg_parquet(
    rows: list[SampleRow], class_counts: dict[str, dict[str, int]], out_path: Path
) -> int:
    """``data/_tasklabels/coralseg/semseg.parquet``: sha256, source_id, label_origin, mask_key,
    class_counts (D-Z2's fixed columns + the semseg payload)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for row in rows:
        counts = class_counts.get(row.image_sha256)
        if counts is None:
            continue
        mask_ref = next((r for r in row.label_refs if r.startswith("labels/files/")), None)
        records.append(
            {
                "sha256": row.image_sha256,
                "source_id": SOURCE_ID,
                "label_origin": LABEL_ORIGIN,
                "mask_key": mask_ref,
                "class_counts": json.dumps(counts, sort_keys=True),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if records:
        pq.write_table(pa.Table.from_pylist(records), out_path, compression="zstd")
    return len(records)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True, help="scratch dir (deleted as it drains)")
    p.add_argument(
        "--out", type=Path, required=True, help="data/_tasklabels/coralseg/semseg.parquet"
    )
    p.add_argument("--limit", type=int, default=None, help="cap pairs processed (testing only)")
    p.add_argument("--remote", default="rs-hel1")
    args = p.parse_args(argv)
    client = client_from_rclone(args.remote)
    report, counts, rows = restage(client, args.work, limit=args.limit)
    n = write_semseg_parquet(rows, counts, args.out)
    print(
        json.dumps(
            {
                "images": report.images,
                "files": report.files,
                "bytes": report.bytes,
                "uploaded": report.uploaded,
                "skipped": report.skipped,
                "verified": report.verified,
                "root_digest": report.root_digest,
                "semseg_rows": n,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
