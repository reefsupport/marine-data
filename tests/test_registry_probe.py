"""``marinedata.registry_probe``: single-key anonymous reads, never a listing (P1.13, P1.14 F8)."""

import hashlib

from marinedata import registry_probe as rp


def test_key_url_names_one_key_on_the_regional_endpoint():
    url = rp.key_url("rs-storage-open", "sources/x/v1/", "CHECKSUMS.sha256")
    assert url == f"https://rs-storage-open.{rp.ENDPOINT}/sources/x/v1/CHECKSUMS.sha256"


def test_read_checksums_digests_the_file_with_one_get(monkeypatch):
    body = b"aa  a.jpg\nbb  b.jpg\n\n"
    urls = []
    monkeypatch.setattr(rp, "get", lambda url, *a, **k: urls.append(url) or body)
    assert rp.read_checksums("rs-storage-open", "p/v1") == (
        hashlib.sha256(body).hexdigest(),
        2,
        len(body),
    )
    assert urls == [rp.key_url("rs-storage-open", "p/v1", rp.CHECKSUMS_NAME)]


def test_an_absent_or_malformed_key_gives_none(monkeypatch):
    monkeypatch.setattr(rp, "get", lambda *a, **k: None)
    assert rp.read_checksums("b", "p") is None
    assert rp.read_ingest_images("b", "p") is None
    monkeypatch.setattr(rp, "get", lambda *a, **k: b'{"images": true}')
    assert rp.read_ingest_images("b", "p") is None
    monkeypatch.setattr(rp, "get", lambda *a, **k: b'{"images": 12}')
    assert rp.read_ingest_images("b", "p") == 12


def test_a_private_bucket_row_is_never_read(monkeypatch):
    monkeypatch.setattr(rp, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("network")))
    row = {"bucket": "rs-storage-private", "source_id": "s", "version": "v1", "bytes": "10"}
    out = rp.probe_tree(row)
    assert out["gets"] == 0 and "private bucket" in out["skipped"]
