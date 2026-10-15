"""Registry counts and digests agree with the bucket (P1.2).

Network tests (marker ``network``: ``pytest -m network tests/test_registry_counts.py``): for every
entry that names a staged tree (an ``s3`` pointer, or a recorded ``checksums`` digest, whose tree
sits at ``sources/<id>/<version>/``) read that tree's own keys over anonymous HTTPS (single-key
GETs, never a listing) and compare

* ``checksums.root_digest`` with the sha256 of the tree's CHECKSUMS.sha256, and
* ``n_images`` with INGEST.json ``images`` / the metadata.parquet footer row count, when that is
  the recorded ``n_images_method``.

The private bucket cannot be read anonymously: those entries are skipped. When D9 closes, the
open bucket needs signed reads too; the nightly job (.github/workflows/registry-nightly.yml) is
where the S3_ACCESS_KEY_ID / S3_SECRET_ACCESS_KEY repo secrets go.

The offline tests need no bucket and run in the default suite.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from marinedata.licence_class import SPEC_ALIASES
from marinedata.registry_probe import (
    OPEN_BUCKET,
    read_checksums,
    read_ingest_images,
    read_parquet_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def _entries() -> list[dict]:
    out: list[dict] = []
    for path in sorted((ROOT / "registry" / "sources").rglob("*.yaml")):
        out += (yaml.safe_load(path.read_text()) or {}).get("sources", [])
    return out


def _tree(entry: dict) -> tuple[str, str] | None:
    """``(bucket, prefix)`` of the staged tree an entry names, or None."""
    access = entry.get("access") or {}
    params = access.get("params") or {}
    if access.get("method") == "s3" and params.get("bucket") and params.get("prefix"):
        return params["bucket"], params["prefix"]
    if entry.get("checksums"):
        sid = SPEC_ALIASES.get(entry["id"], entry["id"])
        return OPEN_BUCKET, f"sources/{sid}/{entry['checksums']['version']}/"
    return None


def _parquet_rows(bucket: str, prefix: str) -> int | None:
    got = read_parquet_rows(bucket, prefix)
    return got[0] if got else None


_LIVE = {"ingest-json": read_ingest_images, "metadata-parquet": _parquet_rows}


@pytest.mark.network
@pytest.mark.parametrize("entry", [pytest.param(e, id=e["id"]) for e in _entries() if _tree(e)])
def test_registry_matches_bucket(entry: dict) -> None:
    bucket, prefix = _tree(entry) or ("", "")
    if bucket != OPEN_BUCKET:
        pytest.skip("private bucket: no anonymous read (signed reads follow D9)")
    method, n, ck = entry.get("n_images_method"), entry.get("n_images"), entry.get("checksums")
    if not ck and not (n is not None and method in _LIVE):
        pytest.skip("nothing recorded that the bucket can confirm")
    problems: list[str] = []
    if ck:
        got = read_checksums(bucket, prefix)
        if got is None:
            problems.append("CHECKSUMS.sha256 not readable")
        elif got[0] != ck["root_digest"]:
            problems.append(f"root_digest {ck['root_digest'][:12]}.. != bucket {got[0][:12]}..")
    if n is not None and method in _LIVE:
        live = _LIVE[method](bucket, prefix)
        if live != n:
            problems.append(f"n_images {n} != bucket {live} ({method})")
    assert not problems, "; ".join(problems)


def test_every_source_states_how_its_count_was_obtained() -> None:
    bad = [e["id"] for e in _entries() if not e.get("n_images_method")]
    assert not bad, bad


def test_every_count_names_a_method_and_no_method_is_a_guess() -> None:
    known = (
        "ingest-json",
        "metadata-parquet",
        "listing",
        "spec:",
        "registry-prior",
        "unavailable:",
    )
    bad = [e["id"] for e in _entries() if not str(e.get("n_images_method")).startswith(known)]
    assert not bad, bad
    assert not [e["id"] for e in _entries() if e.get("n_images") is not None and e["n_images"] < 1]


def test_every_checksummed_tree_that_an_entry_pins_has_its_digest() -> None:
    entries = {(SPEC_ALIASES.get(e["id"], e["id"]), str(e.get("version"))): e for e in _entries()}
    with (ROOT / "registry" / "inventory" / "bucket-trees.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["bucket"] == OPEN_BUCKET]
    missing = [
        r["source_id"]
        for r in rows
        if r["has_checksums"] == "true"
        and (e := entries.get((r["source_id"], r["version"]))) is not None
        and not e.get("checksums")
    ]
    assert not missing, missing
