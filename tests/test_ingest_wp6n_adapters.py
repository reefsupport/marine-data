"""WP-6n: three adapter gaps found by the INT-ingest8 probe.

* split-tar-gz with the ``.partaa``/``.partab`` naming (UW-StereoDepth-40K), not just the
  original ``.aa``/``.ab`` counter -- both collapse into one synthetic ``.tar.gz`` item
  and stream sequentially via the existing ``ConcatReader`` (D-D), no local reassembly.
* a nested HF ``remote_zip`` tree (MarineEVT: ``test/``/``train/`` subfolders) where one
  member zip is genuinely malformed: ``enumerate()`` must not die for the whole tree, and
  every OTHER zip in the tree must still expand and stream correctly.

Fixtures are tiny and built in-test; no network, no upstream bytes."""

from __future__ import annotations

import hashlib
import zipfile

import pytest
from _wp6_fixtures import LocalServer, png, tar_gz_bytes, zip_bytes

from marinedata.adapters import make_adapter


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


# ---------------------------------------------------------------------------
# split-tar-gz: ``.tar.gz.partaa``/``.partab`` (UW-StereoDepth-40K naming)
# ---------------------------------------------------------------------------


def test_split_tar_gz_part_prefix_streams_concatenated(server, tmp_path):
    members = {"pairs/000.png": png(1)}
    whole = tar_gz_bytes(members)
    mid = len(whole) // 2
    halves = [whole[:mid], whole[mid:]]
    urls = []
    for i, chunk in enumerate(halves):
        letter = "a" if i == 0 else "b"
        key = f"output_40000.tar.gz.part{letter}{letter}"
        server.add(f"/{key}", chunk)
        urls.append({"url": f"{server.base}/{key}", "key": key})
    adapter = make_adapter("http", {"urls": urls, "version": "v1"})

    items = list(adapter.enumerate())
    assert len(items) == 1
    assert items[0].key == "output_40000.tar.gz"
    assert len(items[0].parts) == 2

    out = [(item.key, d) for item, _, d in adapter.samples(tmp_path)]
    assert out == [("output_40000.tar.gz", out[0][1])]
    assert out[0][1].data == members["pairs/000.png"]
    assert sha256(out[0][1].data) == sha256(members["pairs/000.png"])


def test_split_tar_gz_old_and_new_naming_both_group(server, tmp_path):
    """The original ``.aa``/``.ab`` counter (D-D, seamapd21) must keep working exactly as
    before -- the broadened regex is additive, never a behaviour change for existing specs."""
    whole = tar_gz_bytes({"a.bin": b"y" * 2000})
    mid = len(whole) // 2
    urls = [
        {"url": f"{server.base}/x.tar.gz.aa", "key": "x.tar.gz.aa"},
        {"url": f"{server.base}/x.tar.gz.ab", "key": "x.tar.gz.ab"},
    ]
    server.add("/x.tar.gz.aa", whole[:mid])
    server.add("/x.tar.gz.ab", whole[mid:])
    adapter = make_adapter("http", {"urls": urls, "version": "v1"})
    items = list(adapter.enumerate())
    assert len(items) == 1 and items[0].key == "x.tar.gz" and len(items[0].parts) == 2


# ---------------------------------------------------------------------------
# nested HF remote_zip tree with one malformed member (MarineEVT)
# ---------------------------------------------------------------------------


def _corrupt_zip(members: dict[str, bytes]) -> bytes:
    """A byte-for-byte valid zip with its tail (central directory + EOCD) sheared off --
    same failure mode as the real MarineEVT member: a real ``PK`` local header at offset
    0, declared size matches what upstream serves, but no EOCD anywhere near the end."""
    blob = zip_bytes(members)
    return blob[: len(blob) // 2] + b"\x00" * (len(blob) - len(blob) // 2)


def test_nested_remote_zip_skips_one_malformed_member_keeps_the_rest(server, tmp_path):
    repo = "AnTo2209/MarineEVT"
    good = {"frames/a.jpg": b"jpgbytes-a" * 50, "frames/b.jpg": b"jpgbytes-b" * 50}
    good_zip = zip_bytes(good)
    bad_zip = _corrupt_zip({"frames/c.jpg": b"jpgbytes-c" * 50})
    qa_json = b'{"q": "how many fish?"}\n'
    files = {
        "test/GoodTopic/videos.zip": good_zip,
        "test/BadTopic/videos.zip": bad_zip,
        "test/qa.json": qa_json,
    }
    sha = "c" * 40
    server.add(f"/api/datasets/{repo}/revision/main", {"sha": sha})
    tree = [{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]
    tree.append({"type": "directory", "path": "test"})
    server.add(f"/api/datasets/{repo}/tree/{sha}?recursive=true", tree)
    for p, b in files.items():
        server.add(f"/datasets/{repo}/resolve/{sha}/{p}", b)
    adapter = make_adapter(
        "hf",
        {
            "repo": repo,
            "endpoint": server.base,
            "include": list(files),
            "label_patterns": ["*.json"],
            "remote_zip": True,
        },
    )
    adapter.resolve_version()

    items = list(adapter.enumerate())
    keys = {i.key for i in items}

    # the good zip expanded into its 2 members ...
    assert "test/GoodTopic/videos.zip#frames/a.jpg" in keys
    assert "test/GoodTopic/videos.zip#frames/b.jpg" in keys
    # ... the malformed one is staged whole instead of crashing enumerate() for the tree ...
    assert "test/BadTopic/videos.zip" in keys
    assert not any(k.startswith("test/BadTopic/videos.zip#") for k in keys)
    # ... and the loose label file is untouched.
    assert "test/qa.json" in keys

    # every OTHER (valid) zip member still fetches and decodes with matching content --
    # the malformed sibling never poisons the rest of the tree. Fetch+decode per item
    # here (not adapter.samples(), which iterates the whole enumerate() including the
    # unexpanded bad item -- D-AF's per-item skip is ingest_missing's job, not the
    # adapter's; the adapter's contract is only "don't crash enumerate()").
    by_key = {i.key: i for i in items}
    good_item = by_key["test/GoodTopic/videos.zip#frames/a.jpg"]
    fetched = adapter.fetch(good_item, tmp_path)
    decoded = next(adapter.decode(fetched))
    assert decoded.data == good["frames/a.jpg"]
    assert sha256(decoded.data) == sha256(good["frames/a.jpg"])

    # the malformed member raises when actually decoded -- proving it is a real,
    # classified failure (D-R4) rather than silently treated as empty/ok.
    bad_item = by_key["test/BadTopic/videos.zip"]
    with pytest.raises(zipfile.BadZipFile):
        list(adapter.decode(adapter.fetch(bad_item, tmp_path)))
