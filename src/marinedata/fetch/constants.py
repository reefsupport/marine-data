"""Shared endpoints and byte caps for the fetchers: one place, no second literal."""

from __future__ import annotations

from ..checksums import DOWNLOAD_USER_AGENT

DATASETS_SERVER = "https://datasets-server.huggingface.co"
DEFAULT_LIMIT = 100
USER_AGENT = DOWNLOAD_USER_AGENT
"""Alias, not a second literal (D3a2 tidy) — see :data:`marinedata.checksums.
DOWNLOAD_USER_AGENT`."""

_ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tar", ".zip")

MAX_DOWNLOAD_WITHOUT_RANGE = 200 * 1024 * 1024
"""A full-body GET is only safe up to this size. CoralVQA's real image archive is a
single 26.7 GB zip with no smaller subset published anywhere — downloading that just to
read 100 verification images would be absurd, and it is exactly the shape of "large
upstream archive, needs a bounded sample" that this registry expects to hit again."""

MAX_MEMBER_BYTES = 500 * 1024 * 1024
"""Refuse any single archive member larger than this once extracted (uncompressed). A
verification sample is meant to stay small (~100 items); one member this large is
either the wrong archive entirely or a bomb, not a sample."""

MAX_TOTAL_EXTRACT_BYTES = 2 * 1024 * 1024 * 1024
"""Cumulative uncompressed-byte cap across every member extracted by one archive-
reading call, independent of ``limit`` — ``limit`` bounds member *count*, not how many
bytes each member can expand to. A 100-member archive that expands to hundreds of GB is
otherwise "within policy" purely because member count stayed low."""

MAX_COMPRESSION_RATIO = 100
"""Refuse a member (zip: its own compressed/uncompressed sizes; tar: the whole archive
payload vs. cumulative bytes extracted so far, since tar has no per-member compressed
size once the stream itself is compressed) whose uncompressed size exceeds this
multiple of its compressed size. 100:1 is a generous, conventional zip-bomb threshold —
real photos and label files don't compress anywhere near that hard; a crafted
all-zeros payload does."""

MAX_GET_BYTES = MAX_TOTAL_EXTRACT_BYTES
"""Hard ceiling on a single ``_get()`` response body, enforced while reading rather
than trusted from ``Content-Length`` — a header can under-report, lie, or be absent
while the connection keeps streaming."""
