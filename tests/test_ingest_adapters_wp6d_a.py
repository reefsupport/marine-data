"""One network-free fixture test per WP-6d-A web/API source adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from _wp6_fixtures import LocalServer, md5, png, sha256

from marinedata.adapters import AccessRefused, make_adapter


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


def _run(adapter, tmp_path: Path):
    return [(item.key, d) for item, _, d in adapter.samples(tmp_path)]


def test_fathomnet_paginated_boxes_and_licence(server, tmp_path):
    body = png(1)
    server.add(
        "/api/images?page=0&size=1",
        {
            "content": [
                {
                    "uuid": "img-1",
                    "url": f"{server.base}/img1.jpg",
                    "sha256": sha256(body),
                    "latitude": 1.0,
                    "longitude": 2.0,
                    "depthMeters": 5.0,
                    "timestamp": "2021-01-01T00:00:00Z",
                    "boundingBoxes": [{"concept": "fish", "annotationLicense": "CC-BY-4.0"}],
                }
            ],
            "pageNumber": 0,
            "pageSize": 1,
            "totalItems": 1,
            "totalPages": 1,
        },
    )
    server.add("/img1.jpg", body, ctype="image/jpeg")
    adapter = make_adapter(
        "fathomnet", {"endpoint": f"{server.base}/api", "max_items": 1, "page_size": 1}
    )
    assert adapter.resolve_version().startswith("fathomnet-")
    out = _run(adapter, tmp_path)
    assert len(out) == 1
    decoded = out[0][1]
    assert decoded.fields["depth_m"] == 5.0
    assert decoded.labels["concepts"] == "fish"
    assert decoded.labels["licence"] == "CC-BY-4.0"
    assert "img-1.json" in decoded.label_files


def test_figshare_article_files(server, tmp_path):
    body = png(2)
    server.add(
        "/v2/articles/123",
        {
            "id": 123,
            "version": 2,
            "is_embargoed": False,
            "is_confidential": False,
            "files": [
                {
                    "name": "a.png",
                    "size": len(body),
                    "download_url": f"{server.base}/dl/a.png",
                    "computed_md5": md5(body),
                }
            ],
        },
    )
    server.add("/dl/a.png", body)
    adapter = make_adapter("figshare", {"article": 123, "endpoint": f"{server.base}/v2"})
    assert adapter.resolve_version() == "2"
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["123/a.png"]


def test_figshare_embargoed_refuses(server, tmp_path):
    server.add("/v2/articles/9", {"id": 9, "is_embargoed": True, "files": []})
    adapter = make_adapter("figshare", {"article": 9, "endpoint": f"{server.base}/v2"})
    with pytest.raises(AccessRefused):
        adapter.resolve_version()


def test_pangaea_textfile_image_and_hash_columns(server, tmp_path):
    body = png(3)
    text = (
        "/* DATA DESCRIPTION:\n"
        "Citation: someone\n"
        "*/\n"
        "Date/Time\tIMAGE\tIMAGE (Hash)\n"
        f"2021-01-01\timg1.jpg\t{md5(body)}\n"
    ).encode()
    server.add("/10.1594/PANGAEA.987009?format=textfile", text, ctype="text/plain")
    server.add("/dl/987009/files/img1.jpg", body)
    adapter = make_adapter(
        "pangaea",
        {
            "doi": "10.1594/PANGAEA.987009",
            "endpoint": server.base,
            "download_base": f"{server.base}/dl",
        },
    )
    assert adapter.resolve_version() == "PANGAEA.987009"
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["img1.jpg"]


def test_http_index_recursive_html_directory(server, tmp_path):
    img_a, img_b = png(4), png(5)
    server.add(
        "/idx/",
        b'<html><body><a href="sub/">sub/</a><a href="a.jpg">a.jpg</a></body></html>',
        ctype="text/html",
    )
    server.add(
        "/idx/sub/", b'<html><body><a href="b.jpg">b.jpg</a></body></html>', ctype="text/html"
    )
    server.add("/idx/a.jpg", img_a)
    server.add("/idx/sub/b.jpg", img_b)
    adapter = make_adapter(
        "http-index", {"index_url": f"{server.base}/idx/", "version": "v1", "depth": 3}
    )
    assert adapter.resolve_version() == "v1"
    out = sorted(k for k, _ in _run(adapter, tmp_path))
    assert out == ["a.jpg", "b.jpg"]


def test_http_index_csv_url_list(server, tmp_path):
    body = png(6)
    server.add("/f1.jpg", body)
    csv_body = f"url\n{server.base}/f1.jpg\n".encode()
    server.add("/list.csv", csv_body, ctype="text/csv")
    adapter = make_adapter(
        "http-index",
        {"list_url": f"{server.base}/list.csv", "url_column": "url", "version": "v1"},
    )
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["f1.jpg"]


def test_gdrive_public_direct_and_confirm_page(server, tmp_path):
    body = png(7)
    server.add("/uc?export=download&id=abc123", body)
    adapter = make_adapter(
        "gdrive-public", {"file_id": "abc123", "name": "a.png", "endpoint": server.base}
    )
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["a.png"]


def test_gdrive_public_quota_wall_refuses(server, tmp_path):
    server.add(
        "/uc?export=download&id=big1",
        b"<html>Google Drive - Quota exceeded for this file</html>",
        ctype="text/html",
    )
    adapter = make_adapter(
        "gdrive-public", {"file_id": "big1", "name": "b.png", "endpoint": server.base}
    )
    with pytest.raises(AccessRefused):
        list(adapter.samples(tmp_path))


def test_gdrive_usercontent_confirm_form_parsed():
    """Drive's current >100MB interstitial: a GET <form> to usercontent.google.com
    with hidden id/export/confirm/uuid inputs (not the old bare confirm= anchor)."""
    from marinedata.adapters.gdrive import GDriveAdapter

    adapter = GDriveAdapter({"file_id": "big2", "name": "c.zip", "endpoint": "https://drive.google.com"})
    html = (
        '<html><body><form id="download-form" '
        'action="https://drive.usercontent.google.com/download" method="get">'
        '<input type="hidden" name="id" value="big2">'
        '<input type="hidden" name="export" value="download">'
        '<input type="hidden" name="confirm" value="t">'
        '<input type="hidden" name="uuid" value="u-1">'
        "</form></body></html>"
    )
    assert adapter._extract_confirm_url(html) == (
        "https://drive.usercontent.google.com/download?id=big2&export=download&confirm=t&uuid=u-1"
    )


def test_gdrive_legacy_confirm_href_still_works():
    from marinedata.adapters.gdrive import GDriveAdapter

    adapter = GDriveAdapter(
        {"file_id": "big3", "name": "d.zip", "endpoint": "https://drive.google.com"}
    )
    html = (
        '<html><body><a id="uc-download-link" '
        'href="/uc?export=download&amp;id=big3&amp;confirm=t9x">Download anyway</a>'
        "</body></html>"
    )
    assert (
        adapter._extract_confirm_url(html)
        == "https://drive.google.com/uc?export=download&id=big3&confirm=t9x"
    )


def test_gdrive_confirm_page_without_form_needs_yohan(server, tmp_path):
    server.add(
        "/uc?export=download&id=big4",
        b"<html><body>Sorry, no preview is available for this file.</body></html>",
        ctype="text/html",
    )
    adapter = make_adapter(
        "gdrive-public", {"file_id": "big4", "name": "e.png", "endpoint": server.base}
    )
    with pytest.raises(AccessRefused, match="had no resolvable download link"):
        list(adapter.samples(tmp_path))


def test_seafile_share_dir_listing_and_dl(server, tmp_path):
    body = png(8)
    server.add(
        "/api/v2.1/share-links/TOK/dirents/?path=%2F",
        {"dirent_list": [{"is_dir": False, "file_path": "/a.png", "size": len(body)}]},
    )
    server.add("/d/TOK/files/?p=/a.png&dl=1", body)
    adapter = make_adapter("seafile-share", {"share": f"{server.base}/d/TOK"})
    assert adapter.resolve_version() == "share-TOK"
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["a.png"]


def test_girder_folder_item_download(server, tmp_path):
    body = png(9)
    server.add(
        "/folder?parentId=COLL&parentType=collection&limit=0",
        [{"_id": "F1", "name": "images"}],
    )
    server.add("/item?folderId=F1&limit=0", [{"_id": "I1", "name": "a.png", "size": len(body)}])
    server.add("/folder?parentId=F1&parentType=folder&limit=0", [])
    server.add("/item/I1/download", body)
    adapter = make_adapter("girder", {"collection": "COLL", "api": server.base})
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["images/a.png"]


def test_pawsey_portal_swift_json_listing(server, tmp_path):
    body = png(10)
    server.add("/container?format=json", [{"name": "a.png", "bytes": len(body), "hash": md5(body)}])
    server.add("/container?format=json&marker=a.png", [])
    server.add("/container/a.png", body)
    adapter = make_adapter(
        "pawsey-portal", {"container_url": f"{server.base}/container", "version": "list-1"}
    )
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["a.png"]


def test_pawsey_portal_js_shell_enumerates_zero(server, tmp_path):
    """0 items from the Swift listing is a D-R4 ``NoStageableItems`` (INT-ingest2
    merge: this was ``== []`` before WP-6d-B's structural fix landed on the shared
    ``BaseAdapter.enumerate()`` all adapters, including this one, inherit)."""
    from marinedata.adapters import NoStageableItems

    server.add("/container?format=json", b"<html>JS app shell</html>", ctype="text/html")
    adapter = make_adapter(
        "pawsey-portal", {"container_url": f"{server.base}/container", "version": "list-1"}
    )
    with pytest.raises(NoStageableItems, match="pawsey-portal"):
        list(adapter.enumerate())


def test_frdr_https_explicit_manifest(server, tmp_path):
    body = png(11)
    server.add("/file1.jpg", body)
    adapter = make_adapter(
        "frdr-https",
        {
            "urls": [{"url": f"{server.base}/file1.jpg", "key": "file1.jpg"}],
            "version": "manifest-1",
        },
    )
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["file1.jpg"]
