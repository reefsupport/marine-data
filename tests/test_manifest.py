"""WP-3: MANIFEST.tsv build + parse — streaming digest, s3-style ETag, parquet fingerprint."""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq

from marinedata.manifest import build_manifest, digest_file, parquet_fingerprint, read_manifest


def test_digest_file_single_part(tmp_path):
    p = tmp_path / "a.bin"
    data = b"hello world"
    p.write_bytes(data)
    d = digest_file(p, part_size=1 << 20)
    assert d.size == len(data)
    assert d.sha256 == hashlib.sha256(data).hexdigest()
    assert d.s3_etag == hashlib.md5(data, usedforsecurity=False).hexdigest()


def test_digest_file_multipart_etag_matches_md5_of_md5s(tmp_path):
    p = tmp_path / "b.bin"
    part = b"x" * 10
    p.write_bytes(part * 3)
    d = digest_file(p, part_size=10)
    md5s = [hashlib.md5(part, usedforsecurity=False).hexdigest() for _ in range(3)]
    joined = b"".join(bytes.fromhex(m) for m in md5s)
    expected = f"{hashlib.md5(joined, usedforsecurity=False).hexdigest()}-3"
    assert d.s3_etag == expected


def test_parquet_fingerprint_stable_for_same_schema(tmp_path):
    t1 = pa.table({"a": [1, 2], "b": ["x", "y"]})
    t2 = pa.table({"a": [3], "b": ["z"]})
    p1, p2 = tmp_path / "t1.parquet", tmp_path / "t2.parquet"
    pq.write_table(t1, p1)
    pq.write_table(t2, p2)
    r1, r2 = parquet_fingerprint(p1), parquet_fingerprint(p2)
    assert r1[0] == 2 and r2[0] == 1
    assert r1[1] == r2[1]  # same columns/types -> same fingerprint despite different rows


def test_parquet_fingerprint_none_for_non_parquet(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hi")
    assert parquet_fingerprint(p) is None


def test_build_and_read_manifest_roundtrip(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "a.txt").write_text("aaa")
    (tmp_path / "b.txt").write_text("bb")
    pq.write_table(pa.table({"x": [1, 2, 3]}), tmp_path / "c.parquet")

    dest = build_manifest(tmp_path)
    assert dest == tmp_path / "MANIFEST.tsv"
    rows = read_manifest(dest)
    assert set(rows) == {"sub/a.txt", "b.txt", "c.parquet"}
    assert rows["sub/a.txt"].size == 3
    assert rows["c.parquet"].rows == 3
    assert rows["c.parquet"].schema_fingerprint
    assert rows["b.txt"].rows is None

    # MANIFEST.tsv never lists itself, even on a rebuild.
    build_manifest(tmp_path)
    rows2 = read_manifest(dest)
    assert "MANIFEST.tsv" not in rows2
