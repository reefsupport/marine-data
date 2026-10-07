"""``registry verify`` (RB-3): offline listing-snapshot check on a small fixture."""

from __future__ import annotations

import io
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from marinedata.cli import main
from marinedata.registry import Registry
from marinedata.registry_verify import (
    HOLLOW,
    MISSING,
    NO_MANIFEST,
    OK,
    SKIPPED,
    failed,
    load_listing,
    summarise,
    verify_live,
    verify_offline,
)

OPEN = "rs-storage-open"


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


def _pick(reg: Registry) -> tuple[str, str, str]:
    """Three s3 sources with a bucket+prefix pointer, to place in / out of the fixture."""
    ids = [
        s.id
        for s in reg.sources
        if s.access.method.value == "s3" and (s.access.params or {}).get("prefix") and not s.retired
    ]
    assert len(ids) >= 3
    return ids[0], ids[1], ids[2]


def _prefix(reg: Registry, sid: str) -> tuple[str, str]:
    p = reg.source(sid).access.params
    return p["bucket"], p["prefix"].rstrip("/") + "/"


def test_offline_verify_classifies_ok_no_manifest_missing_and_skips_retired(reg: Registry) -> None:
    a, b, c = _pick(reg)
    ba, pa_ = _prefix(reg, a)
    bb, pb = _prefix(reg, b)
    listing: dict[str, set[str]] = {}
    listing.setdefault(ba, set()).update({pa_ + "CHECKSUMS.sha256", pa_ + "images/x.jpg"})
    listing.setdefault(bb, set()).add(pb + "images/y.jpg")
    checks = {k.source_id: k for k in verify_offline([reg.source(i) for i in (a, b, c)], listing)}
    assert checks[a].status == OK
    assert checks[b].status == NO_MANIFEST
    assert checks[c].status == MISSING
    assert {k.source_id for k in failed(checks.values())} == {b, c}
    retired = verify_offline([reg.source("rs-labelled-masks")], {})
    assert [k.status for k in retired] == [SKIPPED] and "retired" in retired[0].detail


def test_listing_loaders_infer_the_bucket_from_the_file_name(tmp_path: Path) -> None:
    pq_path = tmp_path / f"inventory-{OPEN}.parquet"
    pq.write_table(pa.table({"key": ["sources/a/1/CHECKSUMS.sha256"]}), pq_path)
    tsv = tmp_path / "private-listing.tsv"
    tsv.write_text("sources/b/1/CHECKSUMS.sha256\t12\n")
    assert load_listing(pq_path) == {OPEN: {"sources/a/1/CHECKSUMS.sha256"}}
    assert load_listing(tsv, "rs-storage-private") == {
        "rs-storage-private": {"sources/b/1/CHECKSUMS.sha256"}
    }
    with pytest.raises(ValueError):
        load_listing(tsv)


def test_cli_registry_verify_offline_exit_codes(reg: Registry, tmp_path: Path, capsys) -> None:
    a, _, _ = _pick(reg)
    bucket, prefix = _prefix(reg, a)
    snap = tmp_path / "listing.tsv"
    snap.write_text(f"{prefix}CHECKSUMS.sha256\n")
    code = main(["registry", "verify", a, "--listing", f"{bucket}={snap}", "--all"])
    out = capsys.readouterr().out
    assert code == 0 and "ok" in out and a in out
    snap.write_text("unrelated/key\n")
    assert main(["registry", "verify", a, "--listing", f"{bucket}={snap}"]) == 1
    assert main(["registry", "verify", a, "--listing", f"{bucket}={snap}", "--report-only"]) == 0


def test_live_verify_treats_an_unreachable_bucket_as_missing_not_a_crash(reg: Registry) -> None:
    """`--live` first crashed with NoSuchBucket on a third-party bucket (imos-data): a
    ClientError from the 1-key LIST fallback means the prefix is not there."""
    from botocore.exceptions import ClientError

    a, b, _c = _pick(reg)
    ba, pa_ = _prefix(reg, a)

    class FakeClient:
        def head_object(self, Bucket: str, Key: str):
            if Bucket == ba and Key == pa_ + "CHECKSUMS.sha256":
                return {}
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def list_objects_v2(self, **_kw):
            raise ClientError({"Error": {"Code": "NoSuchBucket"}}, "ListObjectsV2")

    checks = {k.source_id: k for k in verify_live([reg.source(a), reg.source(b)], FakeClient())}
    assert checks[a].status == OK
    assert checks[b].status == MISSING


def test_live_sample_flags_a_tree_whose_manifest_survived_but_whose_data_is_gone(
    reg: Registry,
) -> None:
    """RB-2 deleted the data objects of 4 older trees and left their CHECKSUMS: a bare HEAD of the
    manifest still said ok. `--sample N` HEADs listed keys, so the hollow tree must fail."""
    from botocore.exceptions import ClientError

    a, b, _c = _pick(reg)
    (ba, pa_), (bb, pb) = _prefix(reg, a), _prefix(reg, b)
    rels = [f"images/{i:03d}.jpg" for i in range(20)]
    manifest = "".join(f"{'0' * 64}  {r}\n" for r in rels).encode()
    stored = {(ba, pa_ + r) for r in rels} | {(ba, pa_ + "CHECKSUMS.sha256")}
    stored |= {(bb, pb + "CHECKSUMS.sha256")}  # b: manifest only, every data key gone
    gets: list[tuple[str, str]] = []

    class FakeClient:
        def head_object(self, Bucket: str, Key: str):
            if (Bucket, Key) in stored:
                return {}
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

        def get_object(self, Bucket: str, Key: str):
            gets.append((Bucket, Key))
            return {"Body": io.BytesIO(manifest)}

        def list_objects_v2(self, **_kw):
            return {"KeyCount": 0}

    sources = [reg.source(a), reg.source(b)]
    bare = {k.source_id: k.status for k in verify_live(sources, FakeClient())}
    assert bare == {a: OK, b: OK} and not gets  # the gap: both pass without --sample
    checks = {k.source_id: k for k in verify_live(sources, FakeClient(), sample=5)}
    assert checks[a].status == OK and "5 sampled" in checks[a].detail
    assert checks[b].status == HOLLOW and "5/5" in checks[b].detail
    assert [k.source_id for k in failed(checks.values())] == [b]
    assert "hollow=1" in summarise(list(checks.values()))
    assert "hollow" not in summarise(list(verify_live(sources, FakeClient())))
