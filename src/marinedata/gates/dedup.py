"""Dedup gate (P1.6): no exact or perceptual duplicate across sources or across splits.

Every embedded image of the build (the ``images`` config, else every image-bearing config) is
hashed twice: sha256 of the bytes and a 64-bit DCT pHash (:mod:`.phash`, stored in
``<build>/phash.parquet``). Failures:

* ``sha256_mismatch``            the ``image_sha256`` column is not the sha256 of the bytes;
* ``exact_dup_across_splits`` / ``exact_dup_across_sources`` / ``exact_dup_within_split``;
* ``near_dup_across_splits`` / ``near_dup_across_sources``  pHash Hamming <= ``--phash-bits``
  (default 8); near copies inside one split and one source (burst frames) are only counted;
* ``undecodable_image``          the bytes do not decode;
* ``dedup_report``               ``DEDUP_GATE.json`` (the WP-10 split-leak report written by
  ``release build``) is present and not ok.

Sources that republish one upstream image set (the Roboflow Universe forks of one web-scraped set)
declare the same ``family`` in the registry (:func:`marinedata.licence_class.source_families`,
passed in as ``families``). A near pair of two sources of one family inside one split is disclosed,
not failed: it is counted as ``near_pairs_across_sources_in_family`` and a capped sample is listed
as ``near_pairs_across_sources_in_family_sample``. The same pair across two splits is a leak and
still fails (``near_dup_across_splits``), as does a same-split pair whose sources share no family.
Without ``families`` every cross-source pair fails. Exact duplicates are not affected.

A same-split pair of two sources that share no family is confirmed before it fails: a pHash-64 at
the threshold only nominates it (two unrelated textured scenes land within 8 bits by chance among
millions of pairs; run 6 had six at exactly 8 bits and pHash-256 76 to 106 bits apart). The pair is
waived, and counted as ``near_pairs_across_sources_unconfirmed`` with a capped sample of counts and
the two source ids (no sha256), only when the images' pHash-256 (:mod:`marinedata.dedup.features`)
differ in more than 32 bits AND both carry at least 4.0 bits of grey entropy: the confirm-hash of
the split-map stage (:class:`marinedata.neardup.UnionConfirm`). A re-host or re-encode (pHash-256
within 32), a flat frame (entropy below 4.0) and bytes that cannot be re-read all still fail.

A build with no image bytes (tsv-only) is ``skipped`` with the reason, never passed."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path

from ..neardup import UNION_CONFIRM_MIN_ENTROPY_BITS as MIN_ENTROPY_BITS
from ..neardup import UNION_CONFIRM_PHASH256_MAX as PHASH256_MAX
from . import MAX_FAILURES, GateResult, finish, skip
from .phash import ImageRecord, near_pairs, scan_images, write_table

NAME = "dedup"
DEFAULT_PHASH_BITS = 8
IN_FAMILY = "near_pairs_across_sources_in_family"
UNCONFIRMED = "near_pairs_across_sources_unconfirmed"
_REPORT = "DEDUP_GATE.json"
_ROW = re.compile(r"^(?P<config>.+)/(?P<shard>[^/\[]+)\[(?P<row>\d+)\]$")  # ImageRecord.where
_REPORT_LISTS = ("spanning", "upstream_test_in_train", "ungrouped")


def _report(*folders: Path | None) -> tuple[str, str | None]:
    for folder in folders:
        path = Path(folder) / _REPORT if folder else None
        if path is not None and path.is_file():
            data = json.loads(path.read_text())
            bad = data.get("ok") is False or any(data.get(k) for k in _REPORT_LISTS)
            return ("fail" if bad else "pass"), str(path)
    return "absent", None


def _where(r: ImageRecord) -> str:
    return f"{r.config}/{r.split} {r.image_sha256[:12]}"


def _shared_family(a: str, b: str, families: Mapping[str, str]) -> str | None:
    """The family both sources declare, else None (an undeclared source shares none)."""
    family = families.get(a)
    return family if family is not None and family == families.get(b) else None


def _fingerprint(data: bytes | None) -> tuple[bytes, float] | None:
    """``(pHash-256, grey entropy)`` of encoded image bytes, ``None`` when they do not decode."""
    import io

    from PIL import Image

    from .. import neardup
    from ..dedup.features import features_from_image

    if not data:
        return None
    try:
        with Image.open(io.BytesIO(data)) as image:
            return features_from_image(image, "").phash, neardup.grey_entropy_bits(image)
    except Exception:
        return None


def _fingerprints(build_dir: Path, wanted: Iterable[str]) -> dict[str, tuple[bytes, float]]:
    """``{where: (pHash-256, grey entropy)}`` for the rows ``config/shard[row]`` that were scanned,
    read back from their shards one row group at a time. A row that cannot be read is absent."""
    import pyarrow.parquet as pq

    by_shard: dict[Path, dict[int, str]] = defaultdict(dict)
    for where in wanted:
        if found := _ROW.match(where):
            shard = Path(build_dir) / "data" / found["config"] / found["shard"]
            by_shard[shard][int(found["row"])] = where
    out: dict[str, tuple[bytes, float]] = {}
    for path, rows in by_shard.items():
        shard, start = pq.ParquetFile(path), 0
        for group in range(shard.num_row_groups):
            size = shard.metadata.row_group(group).num_rows
            if hit := [r for r in rows if start <= r < start + size]:
                column = shard.read_row_group(group, columns=["image"]).column("image")
                for r in hit:
                    if print_ := _fingerprint((column[r - start].as_py() or {}).get("bytes")):
                        out[rows[r]] = print_
            start += size
    return out


def _confirmed_apart(
    a: tuple[bytes, float] | None, b: tuple[bytes, float] | None
) -> tuple[bool, str]:
    """``(waive, evidence)``: waive only when both fingerprints exist, the pHash-256 are more than
    ``PHASH256_MAX`` bits apart and the flatter image has at least ``MIN_ENTROPY_BITS``."""
    if a is None or b is None:
        return False, "confirm-hash unavailable"
    from ..dedup.features import phash_distance

    far, low = phash_distance(a[0], b[0]), min(a[1], b[1])
    evidence = f"pHash-256 {far} (waiver needs > {PHASH256_MAX}), min grey entropy {low:.1f}"
    return (
        far > PHASH256_MAX and low >= MIN_ENTROPY_BITS,
        f"{evidence} (needs >= {MIN_ENTROPY_BITS:g})",
    )


def run_dedup_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    phash_bits: int = DEFAULT_PHASH_BITS,
    families: Mapping[str, str] | None = None,
) -> GateResult:
    """``families`` maps a source id to its registry ``family``; ``None`` or empty declares none."""
    build_dir = Path(build_dir)
    families = families or {}
    report, report_path = _report(build_dir, release_dir)
    records = scan_images(build_dir)
    if not records:
        return skip(
            NAME,
            "no image bytes in the build (tsv-only or manifest-only): exact and perceptual "
            "dedup need pixels",
            {"dedup_gate_json": report},
        )
    write_table(build_dir, records)
    counts: Counter = Counter(images=len(records))
    failures: list[str] = []
    in_family: list[str] = []
    candidates: list[tuple[ImageRecord, ImageRecord, int]] = []

    def fail(kind: str, where: str, message: str) -> None:
        counts[f"failed_{kind}"] += 1
        failures.append(f"{where} {kind}: {message}")

    if report == "fail":
        fail("dedup_report", report_path or _REPORT, "the split-leak report is not ok")
    by_digest: dict[tuple[str, str], list[ImageRecord]] = defaultdict(list)
    for r in records:
        by_digest[(r.config, r.digest)].append(r)
        if r.exif_transposed:
            # image_sha256 stays the upstream-bytes identity; the stored bytes were re-encoded after
            # the EXIF rotation, so they only have to decode (below) and match the row's size
            if r.row_size and r.decoded_size and r.row_size != r.decoded_size:
                fail(
                    "size_mismatch",
                    _where(r),
                    f"stored bytes are {r.decoded_size[0]}x{r.decoded_size[1]}, "
                    f"the row says {r.row_size[0]}x{r.row_size[1]}",
                )
        elif r.image_sha256 != r.digest:
            fail(
                "sha256_mismatch",
                _where(r),
                f"image_sha256 column != sha256(bytes) {r.digest[:12]}",
            )
        if r.phash is None:
            fail("undecodable_image", _where(r), "the embedded bytes do not decode")
    for (config, digest), group in sorted(by_digest.items()):
        if len(group) < 2:
            continue
        splits, sources = {r.split for r in group}, {r.source for r in group}
        kind = (
            "exact_dup_across_splits"
            if len(splits) > 1
            else "exact_dup_across_sources"
            if len(sources) > 1
            else "exact_dup_within_split"
        )
        fail(
            kind,
            f"{config} {digest[:12]}",
            f"{len(group)} rows in {sorted(splits)} from {sorted(sources)}",
        )
    scored = [r for r in records if r.phash is not None]
    for i, j, dist in near_pairs([r.phash for r in scored], phash_bits):
        a, b = scored[i], scored[j]
        if a.config != b.config or a.digest == b.digest:
            continue
        if a.split != b.split:
            kind = "near_dup_across_splits"
        elif a.source != b.source:
            family = _shared_family(a.source, b.source, families)
            if family is not None:
                counts[IN_FAMILY] += 1
                in_family.append(
                    f"{_where(a)} near_dup_in_family: hamming {dist} <= {phash_bits} to "
                    f"{_where(b)} ({a.source} vs {b.source}, family {family})"
                )
                continue
            candidates.append((a, b, dist))  # confirmed below, once the bytes are read back
            continue
        else:
            counts["near_pairs_same_split_source"] += 1
            continue
        fail(
            kind,
            _where(a),
            f"hamming {dist} <= {phash_bits} to {_where(b)} ({a.source} vs {b.source})",
        )
    unconfirmed: list[str] = []
    prints = _fingerprints(build_dir, {r.where for a, b, _ in candidates for r in (a, b)})
    for a, b, dist in candidates:
        waive, evidence = _confirmed_apart(prints.get(a.where), prints.get(b.where))
        sources = f"{a.source} vs {b.source}"
        if waive:
            counts[UNCONFIRMED] += 1
            unconfirmed.append(
                f"{a.config}/{a.split} near_dup_unconfirmed: hamming {dist} <= {phash_bits}, "
                f"{evidence} ({sources})"
            )
        else:
            fail(
                "near_dup_across_sources",
                _where(a),
                f"hamming {dist} <= {phash_bits} to {_where(b)} ({sources}); {evidence}",
            )
    out = {**counts, "phash_bits": phash_bits, "dedup_gate_json": report}
    if in_family:
        out[f"{IN_FAMILY}_sample"] = sorted(in_family)[:MAX_FAILURES]
    if unconfirmed:
        out[f"{UNCONFIRMED}_sample"] = sorted(unconfirmed)[:MAX_FAILURES]
    return finish(NAME, out, sorted(failures))
