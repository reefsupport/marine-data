"""WP-6i: resumable HF streams (ToL "unexpected end of data") and remote-zip range reads."""

from __future__ import annotations

import io
import json
import os
import struct
import tarfile
import threading
import zipfile
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from marinedata.adapters import Fetched, RemoteItem, make_adapter
from marinedata.adapters._http import HashingReader, open_url
from marinedata.adapters._range import RemoteFile, ResumableStream, open_remote_zip, read_member
from marinedata.adapters.treeoflife import HFMemberFilterAdapter
from marinedata.concurrency import RetriesExhausted

SHA = "b" * 40


class RangeServer:
    """Files with Range support, redirects, one-shot mid-body drops, 403s and a request log."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.redirects: dict[str, str] = {}
        self.drop_first: dict[str, int] = {}  # path -> bytes sent before the socket closes
        self.refuse: set[str] = set()
        self.no_range: set[str] = set()
        self.log: list[tuple[str, str | None]] = []
        srv = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path, rng = self.path, self.headers.get("Range")
                srv.log.append((path, rng))
                if path in srv.redirects:
                    self.send_response(302)
                    self.send_header("Location", srv.redirects[path])
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = srv.files.get(path)
                if path in srv.refuse or body is None:
                    self.send_response(403 if path in srv.refuse else 404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                part, status, extra = body, 200, {}
                if rng and path not in srv.no_range:
                    a, _, b = rng.removeprefix("bytes=").partition("-")
                    lo = len(body) - int(b) if a == "" else int(a)
                    hi = len(body) - 1 if a == "" or b == "" else int(b)
                    part, status = body[lo : hi + 1], 206
                    extra = {"Content-Range": f"bytes {lo}-{hi}/{len(body)}"}
                self.send_response(status)
                self.send_header("Content-Length", str(len(part)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                cut = srv.drop_first.pop(path, None)
                self.wfile.write(part if cut is None else part[:cut])  # then HTTP/1.0 closes

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def srv():
    s = RangeServer()
    yield s
    s.close()


def _tar_gz(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_dropped_connection_is_a_silent_short_read_then_resumes(srv) -> None:
    body = os.urandom(300_000)
    srv.files["/f"] = body
    srv.drop_first["/f"] = 100_000
    with open_url(srv.base + "/f") as resp:  # the mechanism: no IncompleteRead, just b""
        got = b"".join(iter(lambda: resp.read(65536), b""))
    assert len(got) == 100_000
    srv.drop_first["/f"] = 100_000
    rs = ResumableStream(srv.base + "/f")
    assert rs.read() == body and rs.resumes == 1
    assert srv.log[-1] == ("/f", "bytes=100000-")


def test_resume_refused_when_server_ignores_range(srv) -> None:
    srv.files["/f"] = os.urandom(50_000)
    srv.drop_first["/f"] = 10_000
    srv.no_range.add("/f")
    with pytest.raises(RetriesExhausted, match="ignored Range"):
        ResumableStream(srv.base + "/f").read()


def test_tol_shard_stream_survives_a_drop_mid_tar(srv) -> None:
    members = {f"{i}.jpg": os.urandom(40_000) for i in range(6)}
    srv.files["/s.tar.gz"] = shard = _tar_gz(members)
    srv.drop_first["/s.tar.gz"] = len(shard) // 2
    ad = HFMemberFilterAdapter({"repo": "imageomics/TreeOfLife-10M"})
    ad._keep = {"1": {"class": "Anthozoa"}, "5": {"class": "Asteroidea"}}
    item = RemoteItem("dataset/EOL/image_set_01.tar.gz", srv.base + "/s.tar.gz", len(shard))
    fetched = Fetched(item, stream=HashingReader(ResumableStream(item.url)))
    out = list(ad.decode(fetched))
    fetched.close()  # byte count == declared size, else DigestMismatch
    assert [d.labels["treeoflife_id"] for d in out] == ["1", "5"]
    assert out[1].data == members["5.jpg"] and fetched.size == len(shard)


def test_tol_fetch_builds_keep_set_before_opening_the_shard(monkeypatch, tmp_path) -> None:
    import marinedata.adapters.hf as hf

    events: list[str] = []
    ad = HFMemberFilterAdapter({"repo": "imageomics/TreeOfLife-10M"})
    monkeypatch.setattr(ad, "keep_set", lambda: events.append("keep") or {})
    monkeypatch.setattr(hf, "ResumableStream", lambda url: events.append("open") or io.BytesIO())
    ad.fetch(RemoteItem("dataset/EOL/image_set_01.tar.gz", "http://x/s.tar.gz"), tmp_path)
    assert events == ["keep", "open"]


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", allowZip64=True) as zf:
        for i, (name, data) in enumerate(members.items()):
            zf.writestr(name, data, zipfile.ZIP_DEFLATED if i % 2 else zipfile.ZIP_STORED)
    return buf.getvalue()


def test_remote_zip_reads_directory_and_members_by_range_only(srv) -> None:
    members = {f"imgs/{i:03d}.jpg": os.urandom(30_000) for i in range(40)}
    srv.files["/cdn/v1/a.zip"] = srv.files["/cdn/v2/a.zip"] = blob = _zip(members)
    srv.redirects["/resolve/a.zip"] = "/cdn/v1/a.zip"
    zf, rf = open_remote_zip(srv.base + "/resolve/a.zip", len(blob))
    assert len(zf.infolist()) == 40 and rf.requests == 1  # tail load covers this small cd
    info = zf.getinfo("imgs/007.jpg")
    assert read_member(zf, rf, info) == members["imgs/007.jpg"]
    srv.refuse.add("/cdn/v1/a.zip")  # the signed redirect expires -> resolve again
    srv.redirects["/resolve/a.zip"] = "/cdn/v2/a.zip"
    assert read_member(zf, rf, zf.getinfo("imgs/008.jpg")) == members["imgs/008.jpg"]
    assert all(r is not None for _, r in srv.log)  # never a whole-file GET
    assert rf.fetched < len(blob) // 4
    assert sum(p == "/resolve/a.zip" for p, _ in srv.log) == 2  # redirect reused until refused


def test_remote_file_is_seekable_like_a_file(srv) -> None:
    srv.files["/b"] = body = os.urandom(10_000)
    rf = RemoteFile(srv.base + "/b", len(body), block=1024)
    rf.seek(-100, 2)
    assert rf.read() == body[-100:] and rf.tell() == len(body)
    rf.seek(5000)
    assert rf.read(3000) == body[5000:8000]


def test_hf_remote_zip_stages_only_manifest_members(srv, tmp_path: Path) -> None:
    repo = "Coral/VQA"
    members = {f"Images/{n}.jpg": os.urandom(20_000) for n in ("a", "b", "c", "d")}
    blob = _zip(members)
    manifest = "\n".join(
        json.dumps({"question_id": i, "image": f"{n}.jpg"}) for i, n in enumerate("bdb")
    )
    files = {"CoralVQA_Image.zip": blob, "CoralVQA_test.jsonl": manifest.encode()}
    srv.files[f"/api/datasets/{repo}/revision/main"] = json.dumps({"sha": SHA}).encode()
    tree = [{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]
    srv.files[f"/api/datasets/{repo}/tree/{SHA}?recursive=true"] = json.dumps(tree).encode()
    for p, b in files.items():
        srv.redirects[f"/datasets/{repo}/resolve/{SHA}/{p}"] = f"/cdn/{p}"
        srv.files[f"/cdn/{p}"] = b
    ad = make_adapter(
        "hf",
        {
            "repo": repo,
            "endpoint": srv.base,
            "include": list(files),
            "label_patterns": ["*.jsonl"],
            "remote_zip": True,
            "zip_manifest": ["*.jsonl"],
        },
    )
    ad.resolve_version()
    out = [(item.key, d) for item, _, d in ad.samples(tmp_path)]
    keys = [k for k, _ in out]
    assert keys == [
        "CoralVQA_Image.zip#Images/b.jpg",
        "CoralVQA_Image.zip#Images/d.jpg",
        "CoralVQA_test.jsonl",
    ]
    assert out[0][1].data == members["Images/b.jpg"] and out[0][1].suffix == ".jpg"
    assert not any(p == "/cdn/CoralVQA_Image.zip" and r is None for p, r in srv.log)
    assert not (tmp_path / "fetch").exists()  # nothing spooled


def _mixed_zip(entries: list[tuple[str, bytes, int]]) -> bytes:
    """A hand-built zip with a per-entry compression method (0 stored / 8 deflate / 9
    Deflate64) — WP-6k needs one fixture mixing plain members with a Deflate64 member;
    ``zipfile``'s writer only ever emits methods 0/8, so every entry is packed the same
    low-level way as WP-6j's Deflate64-only fixture (``test_ingest_wp6j._deflate64_zip``)."""
    buf, central = io.BytesIO(), []
    for name, data, method in entries:
        if method == 9:
            import inflate64

            d = inflate64.Deflater()
            comp = d.deflate(data) + d.flush()
        elif method == 8:
            co = zlib.compressobj(9, zlib.DEFLATED, -15)
            comp = co.compress(data) + co.flush()
        else:
            comp = data
        crc, off, n = zlib.crc32(data), buf.tell(), name.encode()
        buf.write(
            struct.pack(
                "<4s5H3L2H",
                b"PK\x03\x04",
                20,
                0,
                method,
                0,
                0,
                crc,
                len(comp),
                len(data),
                len(n),
                0,
            )
        )
        buf.write(n + comp)
        central.append(
            struct.pack(
                "<4s6H3L5H2L",
                b"PK\x01\x02",
                20, 20, 0, method, 0, 0, crc, len(comp), len(data), len(n), 0, 0, 0, 0, 0, off,
            )
            + n
        )  # fmt: skip
    cd_off, cd = buf.tell(), b"".join(central)
    buf.write(
        cd
        + struct.pack(
            "<4s4H2LH", b"PK\x05\x06", 0, 0, len(central), len(central), len(cd), cd_off, 0
        )
    )
    return buf.getvalue()


def test_zenodo_adapter_reads_zip_members_by_range_incl_deflate64(srv, tmp_path: Path) -> None:
    """WP-6k (D-AH 4): the zenodo/http adapter never spools a zip container, whatever its
    size — it expands each member into its own item over ``open_remote_zip``/``read_member``,
    Deflate64 (method 9) included."""
    pytest.importorskip("inflate64")
    members = {
        "imgs/000.jpg": os.urandom(4_000),
        "imgs/001.jpg": os.urandom(4_000) * 3,  # deflate64
        "imgs/002.jpg": os.urandom(4_000),  # stored
    }
    blob = _mixed_zip(
        [
            ("imgs/000.jpg", members["imgs/000.jpg"], 8),
            ("imgs/001.jpg", members["imgs/001.jpg"], 9),
            ("imgs/002.jpg", members["imgs/002.jpg"], 0),
        ]
    )
    srv.files["/files/big.zip"] = blob
    srv.files["/api/records/321"] = json.dumps(
        {
            "metadata": {"access_right": "open", "version": "wp6k-1"},
            "files": [
                {
                    "key": "big.zip",
                    "size": len(blob),
                    "checksum": f"md5:{'0' * 32}",
                    "links": {"self": srv.base + "/files/big.zip"},
                }
            ],
        }
    ).encode()
    adapter = make_adapter("zenodo", {"zenodo_record": 321, "endpoint": srv.base})
    out = {item.key: d.data for item, _, d in adapter.samples(tmp_path)}
    assert out == {f"big.zip#{n}": data for n, data in members.items()}
    assert not any(p == "/files/big.zip" and r is None for p, r in srv.log)  # never whole-file
    assert not (tmp_path / "fetch").exists()  # no local spool of the container
