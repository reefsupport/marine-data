#!/usr/bin/env python3
"""Fetch one pinned file for the release image: download URL, check its sha256, write DEST.

    fetch-pinned.py URL sha256:<64 hex digits> DEST

The Dockerfile runs this where it used to say ``ADD --checksum=... URL DEST``. That flag (and
``--chmod``) exists only in BuildKit, and the release VM builds with the classic Docker builder. The
slim base image has no curl or wget either, so this uses the image's own python and nothing but its
standard library. A digest mismatch or an unreadable URL exits non-zero, which fails the image
build, and DEST is written only after the digest matched. The file mode is the caller's job
(chmod 0444).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import re
import urllib.request
from pathlib import Path

TIMEOUT_S = 60


def digest(text: str) -> str:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", text)
    if not match:
        raise argparse.ArgumentTypeError("expected sha256:<64 lowercase hex digits>")
    return match[1]


def fetch_pinned(url: str, sha256: str, dest: Path) -> None:
    """Write ``url`` to ``dest`` when its bytes hash to ``sha256``; otherwise raise SystemExit."""
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:
            data = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise SystemExit(f"fetch-pinned: cannot read {url}: {exc}") from exc
    got = hashlib.sha256(data).hexdigest()
    if got != sha256:
        raise SystemExit(f"fetch-pinned: sha256 mismatch for {url}: got {got}, pinned {sha256}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    partial.write_bytes(data)
    partial.replace(dest)
    print(f"fetch-pinned: {dest} sha256:{got} ({len(data)} bytes)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download URL, check its sha256, write DEST.")
    parser.add_argument("url")
    parser.add_argument("sha256", type=digest, metavar="sha256:HEX")
    parser.add_argument("dest", type=Path)
    args = parser.parse_args(argv)
    fetch_pinned(args.url, args.sha256, args.dest)


if __name__ == "__main__":
    main()
