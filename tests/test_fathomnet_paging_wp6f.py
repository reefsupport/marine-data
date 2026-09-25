"""WP-6f: FathomNet /api/images is a Spring Pageable (page/size); limit/offset are ignored."""

from __future__ import annotations

import urllib.parse

import pytest

from marinedata.adapters.fathomnet import FathomNetAdapter

TOTAL, SIZE = 25, 10


def _entry(i: int) -> dict:
    sha = "00" * 32 if i in (0, 1) else f"{i:064x}"  # page 0 holds one duplicate sha256
    return {"uuid": f"u{i:03d}", "url": f"https://x/{i}.png", "sha256": sha, "boundingBoxes": []}


def _server(honour_paging: bool):
    calls: list[dict] = []

    def get(path: str) -> dict:
        qs = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(path).query))
        calls.append(qs)
        number = int(qs.get("page", 0)) if honour_paging else 0
        size = int(qs.get("size", 100)) if honour_paging else 100
        content = [_entry(i) for i in range(number * size, min(TOTAL, (number + 1) * size))]
        return {"content": content, "pageNumber": number, "pageSize": size,
                "totalItems": TOTAL if honour_paging else 481126,
                "totalPages": -(-TOTAL // size) if honour_paging else 4812}  # fmt: skip

    return get, calls


def test_capped_run_walks_pages_past_a_duplicate_on_page_0() -> None:
    ad = FathomNetAdapter({"version": "v", "max_items": 12, "page_size": SIZE})
    ad._get, calls = _server(honour_paging=True)  # type: ignore[method-assign]
    keys = [i.key for i in ad.list_items()]
    assert len(keys) == 12 == len(set(keys)) and ad.duplicates == 1
    assert [c["page"] for c in calls] == ["0", "1"] and {c["size"] for c in calls} == {"10"}


def test_full_run_stops_at_total_pages() -> None:
    ad = FathomNetAdapter({"version": "v", "full": True, "page_size": SIZE})
    ad._get, calls = _server(honour_paging=True)  # type: ignore[method-assign]
    assert len(list(ad.list_items())) == TOTAL - 1 and len(calls) == 3


def test_server_ignoring_paging_fails_loudly_instead_of_rereading_page_0() -> None:
    ad = FathomNetAdapter({"version": "v", "max_items": 50, "page_size": SIZE})
    ad._get, calls = _server(honour_paging=False)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="asked for page 1"):
        list(ad.list_items())
    assert len(calls) == 2


def test_version_digest_does_not_depend_on_page_size() -> None:
    versions = set()
    for size in (100, 1000):
        ad = FathomNetAdapter({"max_items": 5, "page_size": size})
        body = {"content": [_entry(i) for i in range(150)], "pageNumber": 0}
        ad._get = lambda path, b=body: b  # type: ignore[method-assign]
        versions.add(ad.resolve_version())
    assert len(versions) == 1


def test_exclude_list_and_manifest_accept_a_pinned_https_url(tmp_path, monkeypatch) -> None:
    import hashlib

    from marinedata.adapters import _http, manifest

    body = ("ab" * 32 + "\n").encode()
    pin = hashlib.sha256(body).hexdigest()
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        dest.write_bytes(body)
        return hashlib.sha256(body).hexdigest(), "", len(body)

    monkeypatch.setattr(_http, "download", fake_download)
    monkeypatch.setenv("MARINEDATA_MANIFEST_CACHE", str(tmp_path))
    url = "https://hel1.example/rs-storage-open/sources/fathomnet/_manifest/x.txt"
    ad = FathomNetAdapter({"version": "v", "exclude_sha256": [url], "exclude_sha256_pin": pin})
    assert ad._exclude_set() == {"ab" * 32}
    assert manifest._resolve(url, pin).read_bytes() == body and len(calls) == 1  # cached
    with pytest.raises(ValueError, match="pinned"):
        manifest._resolve(url.replace("x.txt", "y.txt"), "0" * 64)
