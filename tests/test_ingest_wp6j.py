"""WP-6j: PANGAEA Binary/URL-image columns, pangaea-series enumeration, Deflate64 zips."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest

pytest.importorskip("PIL")

from _wp6_fixtures import LocalServer, md5, png

from marinedata.adapters import Fetched, RemoteItem, make_adapter
from marinedata.adapters.decode import _zip_members
from marinedata.adapters.pangaea import image_column
from marinedata.adapters.zipread import read_member

HDR = "/* DATA DESCRIPTION:\nCitation: x\n*/\n"


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


def _keys(adapter) -> list[str]:
    return [i.key for i in adapter.enumerate()]


def _tab(server, pid: str, body: str) -> None:
    server.add(f"/10.1594/PANGAEA.{pid}?format=textfile", (HDR + body).encode(), ctype="text/plain")


def _search(server, pid: str, kids: list[str], total: int | None = None) -> None:
    results = [{"URI": f"doi:10.1594/PANGAEA.{k}"} for k in kids]
    body = {"totalCount": len(kids) if total is None else total, "results": results}
    server.add(f"/search?q=incollection:{pid}&count=500&offset=0", body)


def _params(server, doi: str) -> dict:
    return {
        "doi": doi,
        "endpoint": server.base,
        "download_base": f"{server.base}/dl",
        "search_url": f"{server.base}/search",
    }


def test_image_column_shapes():
    assert image_column(["Event", "Binary", "Binary (Type)", "Binary (Hash)"]) == (1, "name", 3)
    # "Image no/name" (SO268) must not shadow the real IMAGE parameter.
    assert image_column(["Image no/name", "Date/Time", "IMAGE", "IMAGE (Size) [Bytes]"]) == (
        2,
        "name",
        None,
    )
    assert image_column(["Date/Time", "IMAGE water", "IMAGE water (Hash)"]) == (1, "name", 2)
    assert image_column(["Date/Time", "File name", "URL image", "URL file"]) == (2, "url", None)
    assert image_column(["Date/Time", "Depth water [m]"]) is None


def test_pangaea_binary_column_dedupes_box_rows(server, tmp_path):
    body = png(5)
    _tab(
        server,
        "961314",
        "Event\tFile name\tBinary\tBinary (Type)\tbboxx1 [pixel]\n"
        "e1\tDSC_1.JPG\tDSC_1.JPG\timage/jpeg\t1\ne1\tDSC_1.JPG\tDSC_1.JPG\timage/jpeg\t2\n",
    )
    server.add("/dl/961314/files/DSC_1.JPG", body)
    adapter = make_adapter("pangaea", _params(server, "10.1594/PANGAEA.961314"))
    assert _keys(adapter) == ["DSC_1.JPG"]
    assert [i.key for i, _, _ in adapter.samples(tmp_path)] == ["DSC_1.JPG"]


def test_pangaea_series_walks_children_and_nested(server, tmp_path):
    _search(server, "100", ["101", "102", "103"])
    _tab(server, "101", "Date/Time\tURL image\n2016\t" + f"{server.base}/hs/a/IMG_1.jpg\n")
    _tab(
        server,
        "102",
        f"Date/Time\tIMAGE water\tIMAGE water (Hash)\n2018\tIMG_1.jpg\t{md5(png(2))}\n",
    )
    server.add("/10.1594/PANGAEA.103?format=textfile", b"collection", status=400)
    _search(server, "103", ["104"])
    _tab(server, "104", "Binary\nIMG_9.jpg\n")
    for pid in ("101", "102", "103", "104"):
        _search(server, pid, []) if pid != "103" else None
    server.add("/hs/a/IMG_1.jpg", png(1))
    server.add("/dl/102/files/IMG_1.jpg", png(2))
    adapter = make_adapter("pangaea-series", _params(server, "10.1594/PANGAEA.100"))
    assert adapter.resolve_version() == "PANGAEA.100"
    assert _keys(adapter) == ["101/IMG_1.jpg", "102/IMG_1.jpg", "104/IMG_9.jpg"]
    got = [
        i.key
        for i, _, _ in make_adapter(
            "pangaea-series", {**_params(server, "10.1594/PANGAEA.100"), "max_items": 2}
        ).samples(tmp_path)
    ]
    assert got == ["101/IMG_1.jpg", "102/IMG_1.jpg"]


def test_pangaea_series_plain_dataset_and_short_listing(server):
    _search(server, "200", [])
    _tab(server, "200", "IMAGE\nx.jpg\n")
    assert _keys(make_adapter("pangaea-series", _params(server, "10.1594/PANGAEA.200"))) == [
        "x.jpg"
    ]
    _search(server, "300", ["301"], total=2)
    server.add("/search?q=incollection:300&count=500&offset=1", {"totalCount": 2, "results": []})
    with pytest.raises(RuntimeError, match="lists 2 children, got 1"):
        list(make_adapter("pangaea-series", _params(server, "10.1594/PANGAEA.300")).list_items())


def _deflate64_zip(members: dict[str, bytes]) -> bytes:
    import inflate64

    buf, central = io.BytesIO(), []
    for name, data in members.items():
        d = inflate64.Deflater()
        comp = d.deflate(data) + d.flush()
        crc, off, n = zlib.crc32(data), buf.tell(), name.encode()
        buf.write(
            struct.pack(
                "<4s5H3L2H", b"PK\x03\x04", 20, 0, 9, 0, 0, crc, len(comp), len(data), len(n), 0
            )
        )
        buf.write(n + comp)
        central.append(
            struct.pack(
                "<4s6H3L5H2L",
                b"PK\x01\x02",
                20,
                20,
                0,
                9,
                0,
                0,
                crc,
                len(comp),
                len(data),
                len(n),
                0,
                0,
                0,
                0,
                0,
                off,
            )
            + n
        )
    cd_off, cd = buf.tell(), b"".join(central)
    buf.write(
        cd
        + struct.pack(
            "<4s4H2LH", b"PK\x05\x06", 0, 0, len(central), len(central), len(cd), cd_off, 0
        )
    )
    return buf.getvalue()


def test_deflate64_zip_members(tmp_path: Path):
    pytest.importorskip("inflate64")
    members = {"a/0.jpg": png(7), "a/1.jpg": png(8) * 3}
    path = tmp_path / "d64.zip"
    path.write_bytes(_deflate64_zip(members))
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo("a/0.jpg")
        with pytest.raises(NotImplementedError):
            zf.read(info)
        assert read_member(zf, info) == members["a/0.jpg"]
    fetched = Fetched(RemoteItem(key="d64.zip", url="u"), path=path)
    assert {n: r() for n, r in _zip_members(fetched)} == members


def test_deflate64_crc_mismatch_raises(tmp_path: Path):
    pytest.importorskip("inflate64")
    raw = bytearray(_deflate64_zip({"x.bin": b"hello" * 100}))
    raw[14] ^= 0xFF  # local-header CRC is not what read_member checks; corrupt the central CRC
    cd = raw.rfind(b"PK\x01\x02")
    raw[cd + 16] ^= 0xFF
    path = tmp_path / "bad.zip"
    path.write_bytes(bytes(raw))
    with zipfile.ZipFile(path) as zf, pytest.raises(zipfile.BadZipFile, match="CRC"):
        read_member(zf, zf.getinfo("x.bin"))
