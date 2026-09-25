"""Network-free fixtures for the WP-6 ingest tests: a local HTTP server + tiny images."""

from __future__ import annotations

import hashlib
import io
import json
import random
import tarfile
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

Route = tuple[int, str, bytes, dict[str, str]]


class LocalServer:
    """Serve ``routes[path_with_query] = (status, content_type, body, headers)``."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.seen_headers: list[dict[str, str]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                server.seen_headers.append(dict(self.headers))
                status, ctype, body, extra = server.routes.get(
                    self.path, (404, "text/plain", b"nope", {})
                )
                rng = self.headers.get("Range")
                if status == 200 and rng:  # WP-6k: serve a slice so RemoteFile/open_remote_zip work
                    total = len(body)
                    a, _, b = rng.removeprefix("bytes=").partition("-")
                    lo = total - int(b) if a == "" else int(a)
                    hi = total - 1 if a == "" or b == "" else int(b)
                    body, status = body[lo : hi + 1], 206
                    extra = {**extra, "Content-Range": f"bytes {lo}-{hi}/{total}"}
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                for k, v in extra.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def add(
        self,
        path: str,
        body: bytes | object,
        *,
        status: int = 200,
        ctype: str = "application/octet-stream",
        headers: dict[str, str] | None = None,
    ) -> None:
        if not isinstance(body, bytes):
            body, ctype = json.dumps(body).encode(), "application/json"
        self.routes[path] = (status, ctype, body, headers or {})

    def close(self) -> None:
        self.httpd.shutdown()


def png(seed: int, size: int = 32) -> bytes:
    from PIL import Image

    rnd = random.Random(seed)
    im = Image.frombytes("RGB", (size, size), rnd.randbytes(size * size * 3))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def tar_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def hf_parquet(n: int) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    feats = {
        "image": {"_type": "Image"},
        "label": {"_type": "ClassLabel", "names": ["coral", "sand"]},
    }
    table = pa.table(
        {
            "image": [{"bytes": png(100 + i), "path": f"img{i}.png"} for i in range(n)],
            "label": [i % 2 for i in range(n)],
            "depth": [5.0 + i for i in range(n)],
        }
    ).replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": feats}})})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def s3_listing_xml(entries: list[tuple[str, bytes]]) -> bytes:
    rows = "".join(
        f"<Contents><Key>{k}</Key><Size>{len(b)}</Size><ETag>&quot;{md5(b)}&quot;</ETag></Contents>"
        for k, b in entries
    )
    return (
        f'<?xml version="1.0"?><ListBucketResult>{rows}'
        "<IsTruncated>false</IsTruncated></ListBucketResult>"
    ).encode()
