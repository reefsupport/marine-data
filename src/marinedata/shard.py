"""WebDataset shard writing — the format that makes cloud training viable.

Why this exists: a training run that reads 500,000 individual objects from S3 spends most
of its wall-clock on request latency, not compute. One GET per image, at maybe 20–50 ms
each, starves a GPU no matter how many workers you throw at it. Sequential reads from a
few hundred large tar shards do not.

The [WebDataset](https://github.com/webdataset/webdataset) convention is the field
standard: a plain tar of files grouped by key, where everything sharing a stem is one
sample.

    000000.tar
      reef-support-benthic__site1__img001.jpg
      reef-support-benthic__site1__img001.mask.png
      reef-support-benthic__site1__img001.json

Written with the standard library only — `tarfile` is enough, and requiring the
`webdataset` package to *produce* shards would be a dependency for no benefit. Reading
them works with `webdataset`, `torchdata`, or the small reader here.

**Sharding is a derivative act.** Every writer path here goes through
:func:`~marinedata.mirror.evaluate_mirror` with ``TRAINING_SHARD``, so a no-derivatives
source cannot be sharded even by accident.
"""

from __future__ import annotations

import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .labelindex import IGNORE_INDEX
from .mirror import MirrorTarget, evaluate_mirror
from .sample import Sample

if TYPE_CHECKING:  # pragma: no cover
    from .builder import Dataset, StreamingDataset

DEFAULT_SHARD_BYTES = 512 * 1024 * 1024
"""512 MB. Large enough to amortise request latency, small enough to redistribute work
across dataloader ranks without one straggler shard dominating an epoch."""


class ShardError(Exception):
    """Sharding could not proceed."""


@dataclass(frozen=True)
class ShardResult:
    output_dir: Path
    shards: tuple[str, ...]
    samples: int
    bytes_written: int
    skipped: int

    def summary(self) -> str:
        return (
            f"{len(self.shards)} shard(s)  {self.samples:,} samples  "
            f"{self.bytes_written / 1e9:.2f} GB  skipped={self.skipped}  → {self.output_dir}"
        )


def _sample_key(sample: Sample, position: int) -> str:
    """Flat, collision-proof key. Tar has no directories worth relying on.

    Includes the source id and partition so a shard is self-describing: if one turns up
    detached from its manifest you can still tell what is in it and where it came from.
    """
    partition = str(sample.meta.get("partition", "")).replace("/", "-")
    stem = Path(sample.key).stem.replace("/", "-").replace(" ", "_")
    parts = [sample.source_id, partition, stem or f"{position:08d}"]
    return "__".join(p for p in parts if p)


def _metadata(sample: Sample, dataset: Dataset | StreamingDataset) -> dict:
    encoded = dataset.label_index.encode(sample)
    return {
        "source_id": sample.source_id,
        "key": sample.key,
        "partition": sample.meta.get("partition"),
        "licence_tier": sample.licence_tier.value if sample.licence_tier else None,
        "labels": {a.value: v.node_id for a, v in sample.labels.items()},
        "label_index": {a.value: i for a, i in encoded.items()},
        "supervised": sorted(a.value for a in sample.supervised),
        "boxes": [list(b) for b in sample.boxes],
        "points": [list(p) for p in sample.points],
        "ignore_index": IGNORE_INDEX,
    }


def write_shards(
    dataset: Dataset | StreamingDataset,
    output_dir: str | Path,
    *,
    split: str | None = None,
    shard_bytes: int = DEFAULT_SHARD_BYTES,
    prefix: str = "shard",
    include_masks: bool = True,
    strict: bool = False,
) -> ShardResult:
    """Write a dataset (or one split) to WebDataset tar shards.

    Accepts a :class:`~marinedata.builder.StreamingDataset` as well as an eager
    :class:`~marinedata.builder.Dataset` — sharding a corpus too large to hold in memory
    should not require holding it in memory a second time to write it out. The
    contributing source ids come from the scan's counters (``corpus.sources``), which
    cost nothing extra: the licence gate still runs before any byte is written, it just
    no longer needs `{s.source_id for s in samples}` over a materialised list to know
    what to check.

    Args:
        shard_bytes: target size per shard. Shards close *after* crossing it, so the
            last sample in each slightly overshoots; exact sizing is not worth the
            complexity of buffering a sample.
        strict: raise on a missing image instead of skipping it. Off by default because
            sampled datasets legitimately have gaps, and a shard job that dies on sample
            400,000 of 500,000 is worse than one that reports 3 skips.

    Raises:
        ShardError: if any contributing source may not be sharded, or nothing was written.
    """
    from .builder import StreamingDataset as _StreamingDataset
    from .registry import Registry, RegistryError

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, _StreamingDataset):
        expected = dataset.split_sizes().get(split, 0) if split else dataset.corpus.total
        if not expected:
            raise ShardError(f"No samples to shard for split={split!r}")
        contributing_source_ids = sorted(dataset.corpus.sources)
        samples: Iterator[Sample] = dataset.split_stream(split) if split else iter(dataset)
    else:
        materialised = dataset.split_samples(split) if split else dataset.samples
        if not materialised:
            raise ShardError(f"No samples to shard for split={split!r}")
        contributing_source_ids = sorted({s.source_id for s in materialised})
        samples = iter(materialised)

    # Sharding is a derivative act. Check every contributing source before writing a byte.
    registry = Registry.load()
    denied = []
    for source_id in contributing_source_ids:
        try:
            source = registry.source(source_id)
        except RegistryError:
            # FAIL CLOSED. An earlier version skipped unrecognised source ids, so any
            # sample carrying an id absent from the registry bypassed the licence gate
            # entirely — the exact failure this module exists to prevent. A source we
            # cannot resolve is a source we cannot clear.
            #
            # This has now regressed once already, having been fixed without a test to
            # pin it. See test_sharding_an_unregistered_source_fails_closed.
            denied.append(
                f"{source_id}: not in the registry, so its licence cannot be resolved. "
                f"Register it, or shard a Dataset built from registry sources."
            )
            continue
        decision = evaluate_mirror(source, MirrorTarget.TRAINING_SHARD)
        if not decision.allowed:
            denied.append(decision.reason)
    if denied:
        raise ShardError("These sources may not be sharded:\n  - " + "\n  - ".join(denied))

    shards: list[str] = []
    total_samples = 0
    total_bytes = 0
    skipped = 0

    handle: tarfile.TarFile | None = None
    current_bytes = 0
    current_path: Path | None = None

    def _open_shard() -> tarfile.TarFile:
        nonlocal current_path, current_bytes
        current_path = out / f"{prefix}-{len(shards):06d}.tar"
        current_bytes = 0
        shards.append(current_path.name)
        return tarfile.open(current_path, "w")

    def _add_bytes(tar: tarfile.TarFile, name: str, payload: bytes) -> int:
        import io

        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        info.mtime = 0  # deterministic: identical content gives an identical shard
        tar.addfile(info, io.BytesIO(payload))
        return len(payload)

    try:
        handle = _open_shard()
        for position, sample in enumerate(samples):
            if sample.image is None or not Path(sample.image).is_file():
                if strict:
                    raise ShardError(f"{sample.source_id}/{sample.key}: image missing")
                skipped += 1
                continue

            key = _sample_key(sample, position)
            image_path = Path(sample.image)
            written = _add_bytes(
                handle, f"{key}{image_path.suffix.lower()}", image_path.read_bytes()
            )

            if include_masks and sample.mask is not None and Path(sample.mask).is_file():
                mask_path = Path(sample.mask)
                written += _add_bytes(
                    handle, f"{key}.mask{mask_path.suffix.lower()}", mask_path.read_bytes()
                )

            written += _add_bytes(
                handle,
                f"{key}.json",
                json.dumps(_metadata(sample, dataset), sort_keys=True).encode(),
            )

            current_bytes += written
            total_bytes += written
            total_samples += 1

            if current_bytes >= shard_bytes:
                handle.close()
                handle = _open_shard()
    finally:
        if handle is not None:
            handle.close()

    # A zero-sample shard is an artefact of closing on the boundary; drop it.
    if current_path is not None and current_bytes == 0 and len(shards) > 1:
        current_path.unlink(missing_ok=True)
        shards.pop()

    if total_samples == 0:
        raise ShardError(
            f"No samples were written ({skipped} skipped). Images are probably not "
            f"present on disk — shard from a fetched dataset, not a registry query."
        )

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "shards": shards,
        "samples": total_samples,
        "skipped": skipped,
        "bytes": total_bytes,
        "shard_bytes_target": shard_bytes,
        "format": "webdataset",
        "ignore_index": IGNORE_INDEX,
        "num_classes": {a.value: n for a, n in dataset.label_index.num_classes().items()},
        "classes": {a.value: list(idx.classes) for a, idx in dataset.label_index.axes.items()},
        "lineage": json.loads(dataset.lineage.to_json()),
    }
    (out / "SHARD_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(dataset.lineage.attribution_text() + "\n", encoding="utf-8")

    return ShardResult(out, tuple(shards), total_samples, total_bytes, skipped)


def read_shard(path: str | Path) -> Iterator[dict]:
    """Minimal WebDataset reader — grouped members, one dict per sample.

    Exists so shards are inspectable without installing `webdataset`. For training, use
    `webdataset` or `torchdata`: they handle sharding across ranks, shuffling and
    decoding, none of which belongs here.
    """
    grouped: dict[str, dict] = {}
    with tarfile.open(path, "r") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name
            # Split on the first dot so `.mask.png` stays one suffix.
            stem, _, suffix = name.partition(".")
            payload = tar.extractfile(member)
            if payload is None:
                continue
            data = payload.read()
            entry = grouped.setdefault(stem, {"__key__": stem})
            if suffix == "json":
                entry["metadata"] = json.loads(data)
            elif suffix.startswith("mask"):
                entry["mask"] = data
            else:
                entry["image"] = data

    yield from grouped.values()
