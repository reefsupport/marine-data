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

A build with no image bytes (tsv-only) is ``skipped`` with the reason, never passed."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from . import GateResult, finish, skip
from .phash import ImageRecord, near_pairs, scan_images, write_table

NAME = "dedup"
DEFAULT_PHASH_BITS = 8
_REPORT = "DEDUP_GATE.json"
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


def run_dedup_gate(
    build_dir: Path, *, release_dir: Path | None = None, phash_bits: int = DEFAULT_PHASH_BITS
) -> GateResult:
    build_dir = Path(build_dir)
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
            kind = "near_dup_across_sources"
        else:
            counts["near_pairs_same_split_source"] += 1
            continue
        fail(
            kind,
            _where(a),
            f"hamming {dist} <= {phash_bits} to {_where(b)} ({a.source} vs {b.source})",
        )
    return finish(
        NAME, {**counts, "phash_bits": phash_bits, "dedup_gate_json": report}, sorted(failures)
    )
