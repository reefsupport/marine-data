"""A fake staged tree for the depth/pairs producers: a lister and a CHECKSUMS fetch."""

from __future__ import annotations

import hashlib

from marinedata.task_layers.s3_keyed import FetchFailed


def sha_of(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def fake_tree(spec, names: list[str], *, checksums: bool = True):
    base = f"sources/{spec.source_id}/{spec.version}/"

    def lister(prefix: str) -> list[str]:
        return [f"{base}images/{n}" for n in names if f"{base}images/".startswith(prefix)]

    def fetch(key: str) -> bytes:
        if not checksums:
            raise FetchFailed(key, 1, "404")
        return "".join(f"{sha_of(n)}  images/{n}\n" for n in names).encode()

    return lister, fetch
