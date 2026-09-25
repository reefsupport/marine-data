"""One network-free fixture test per WP-6 adapter (local HTTP server)."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("PIL")

from _wp6_fixtures import (
    LocalServer,
    hf_parquet,
    md5,
    png,
    s3_listing_xml,
    sha256,
    tar_bytes,
    zip_bytes,
)

from marinedata.adapters import AccessRefused, SourceAdapter, make_adapter

SHA = "a" * 40


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


def _run(adapter, tmp_path: Path):
    return [(item.key, d) for item, _, d in adapter.samples(tmp_path)]


def test_hf_adapter_parquet_paginated_classlabels_no_auth(server, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_should_never_be_sent")
    shard = hf_parquet(3)
    repo = "reef/open-set"
    server.add(f"/api/datasets/{repo}/revision/main", {"sha": SHA, "gated": False})
    page2 = f"/api/datasets/{repo}/tree/{SHA}?recursive=true&cursor=2"
    server.add(
        f"/api/datasets/{repo}/tree/{SHA}?recursive=true",
        [{"type": "file", "path": "README.md", "size": 10}],
        headers={"Link": f'<{server.base}{page2}>; rel="next"'},
    )
    server.add(
        page2,
        [
            {
                "type": "file",
                "path": "data/train-00000.parquet",
                "size": len(shard),
                "lfs": {"oid": sha256(shard), "size": len(shard)},
            }
        ],
    )
    server.add(f"/datasets/{repo}/resolve/{SHA}/data/train-00000.parquet", shard)

    adapter = make_adapter(
        "hf",
        {
            "repo": repo,
            "endpoint": server.base,
            "label_columns": ["label"],
            "columns": {"depth_m": "depth"},
        },
    )
    assert isinstance(adapter, SourceAdapter)
    assert adapter.resolve_version() == f"rev-{SHA[:12]}"
    out = _run(adapter, tmp_path)
    assert [d.labels["label"] for _, d in out] == ["coral", "sand", "coral"]
    assert out[1][1].fields == {"depth_m": 6.0}
    assert {d.split_hint for _, d in out} == {"train"}
    assert all(d.suffix == ".png" for _, d in out)
    assert not any("Authorization" in h or "authorization" in h for h in server.seen_headers)
    assert not list((tmp_path / "fetch").glob("*")), "spooled parquet must be deleted"


def test_hf_adapter_refuses_gated(server, tmp_path):
    server.add("/api/datasets/o/gated/revision/main", {"sha": SHA, "gated": "manual"})
    with pytest.raises(AccessRefused) as exc:
        make_adapter("hf", {"repo": "o/gated", "endpoint": server.base}).resolve_version()
    assert exc.value.url.endswith("/datasets/o/gated")


def test_zenodo_adapter_zip_with_label_files_md5_checked(server, tmp_path):
    archive = zip_bytes({"imgs/a.png": png(1), "imgs/b.png": png(2), "masks/a_mask.png": png(3)})
    server.add("/files/x.zip", archive)
    server.add(
        "/api/records/42",
        {
            "metadata": {"access_right": "open", "version": "1.2"},
            "files": [
                {
                    "key": "x.zip",
                    "size": len(archive),
                    "checksum": f"md5:{md5(archive)}",
                    "links": {"self": f"{server.base}/files/x.zip"},
                }
            ],
        },
    )
    adapter = make_adapter(
        "zenodo", {"zenodo_record": 42, "endpoint": server.base, "label_patterns": ["masks/*"]}
    )
    assert adapter.resolve_version() == "1.2"
    out = _run(adapter, tmp_path)
    images = [d for _, d in out if d.data]
    labels = [d for _, d in out if not d.data]
    assert [d.upstream_id for d in images] == ["x.zip#imgs/a.png", "x.zip#imgs/b.png"]
    assert list(labels[0].label_files) == ["masks/a_mask.png"]


def _zenodo_record(server, body: bytes, *, size, checksum):
    server.add("/files/y.zip", body)
    server.add(
        "/api/records/7",
        {
            "metadata": {},
            "files": [
                {
                    "key": "y.zip",
                    "size": size,
                    "checksum": checksum,
                    "links": {"self": f"{server.base}/files/y.zip"},
                }
            ],
        },
    )
    return make_adapter("zenodo", {"zenodo_record": 7, "endpoint": server.base})


def test_zenodo_adapter_md5_mismatch_kept_and_recorded(server, tmp_path):
    """D-AG: a declared-md5 mismatch keeps the image and records upstream_match=False."""
    adapter = _zenodo_record(
        server, zip_bytes({"a.png": png(1)}), size=None, checksum="md5:" + "0" * 32
    )
    out = list(adapter.samples(tmp_path))
    assert [item.key for item, _, _ in out] == ["y.zip"]
    assert len(out) == 1
    assert out[0][1].upstream_match is False


def test_zenodo_adapter_md5_match_recorded_true(server, tmp_path):
    body = zip_bytes({"a.png": png(1)})
    checksum = "md5:" + hashlib.md5(body).hexdigest()
    adapter = _zenodo_record(server, body, size=len(body), checksum=checksum)
    out = list(adapter.samples(tmp_path))
    assert len(out) == 1 and out[0][1].upstream_match is True


def test_zenodo_adapter_size_mismatch_still_raises(server, tmp_path):
    """A byte-count mismatch is a short transfer, not a provider-hash disagreement: it raises."""
    body = zip_bytes({"a.png": png(1)})
    adapter = _zenodo_record(server, body, size=len(body) + 1, checksum="md5:" + "0" * 32)
    with pytest.raises(ValueError, match="bytes != declared"):
        _run(adapter, tmp_path)


def test_zenodo_adapter_refuses_restricted(server):
    server.add("/api/records/9", {"metadata": {"access_right": "restricted"}, "files": []})
    with pytest.raises(AccessRefused):
        make_adapter("zenodo", {"zenodo_record": 9, "endpoint": server.base}).resolve_version()


def test_bucket_adapter_anonymous_listing(server, tmp_path):
    objs = [("imgs/val/c1.png", png(5)), ("imgs/val/c2.png", png(6)), ("imgs/notes.txt", b"x")]
    q = urllib.parse.urlencode({"list-type": "2", "prefix": "imgs/"})
    server.add(f"/pub?{q}", s3_listing_xml(objs), ctype="application/xml")
    for key, body in objs:
        server.add(f"/pub/{key}", body)
    adapter = make_adapter("bucket", {"endpoint": server.base, "bucket": "pub", "prefix": "imgs/"})
    version = adapter.resolve_version()
    assert version.startswith("list-") and len(version) == 17
    out = _run(adapter, tmp_path)
    assert [k for k, _ in out] == ["imgs/val/c1.png", "imgs/val/c2.png"]
    assert out[0][1].split_hint == "val"
    assert out[0][1].upstream_url == f"{server.base}/pub/imgs/val/c1.png"


def test_github_release_webdataset_tar_streamed(server, tmp_path):
    shard = tar_bytes(
        {
            "s1.png": png(7),
            "s1.json": json.dumps(
                {"depth_m": 12.5, "depth_source": "metadata", "species": "Acropora"}
            ).encode(),
            "s2.png": png(8),
            "s2.cls": b"3\n",
        }
    )
    server.add(
        "/repos/o/r/releases/tags/v1",
        {
            "tag_name": "v1",
            "assets": [
                {
                    "name": "train-000.tar",
                    "size": len(shard),
                    "digest": f"sha256:{sha256(shard)}",
                    "browser_download_url": f"{server.base}/dl/train-000.tar",
                }
            ],
        },
    )
    server.add("/dl/train-000.tar", shard)
    adapter = make_adapter("github", {"repo": "o/r", "release": "v1", "api": server.base})
    assert adapter.resolve_version() == "v1"
    out = _run(adapter, tmp_path)
    assert len(out) == 2
    first, second = out[0][1], out[1][1]
    assert first.fields == {"depth_m": 12.5, "depth_source": "metadata"}
    assert first.labels == {"species": "Acropora"}
    assert second.labels == {"cls": "3"}
    assert not (tmp_path / "fetch").exists(), "tar must stream, never spool"


def test_github_raw_tree_pinned_to_commit(server, tmp_path):
    server.add("/repos/o/r/commits/main", {"sha": SHA})
    server.add(
        f"/repos/o/r/git/trees/{SHA}?recursive=1",
        {
            "tree": [
                {"path": "data/a.png", "type": "blob", "size": len(png(9))},
                {"path": "README.md", "type": "blob", "size": 3},
            ]
        },
    )
    server.add(f"/raw/o/r/{SHA}/data/a.png", png(9))
    adapter = make_adapter(
        "github",
        {"repo": "o/r", "ref": "main", "api": server.base, "raw_base": f"{server.base}/raw"},
    )
    assert adapter.resolve_version() == f"git-{SHA[:12]}"
    assert [k for k, _ in _run(adapter, tmp_path)] == ["data/a.png"]


def test_http_adapter_requires_version():
    with pytest.raises(ValueError, match="version"):
        make_adapter("http", {"urls": ["https://example.org/a.zip"]}).resolve_version()


def test_hf_loose_yolo_repo_pairs_label_files(server, tmp_path):
    import datetime as dt

    from marinedata.staged_writer import StagedWriter, WriterConfig

    repo = "reef/yolo"
    files = {
        "train/images/a.jpg": png(11),
        "train/labels/a.txt": b"0 0.5 0.5 0.2 0.2\n",
        "valid/images/b.jpg": png(12),
        "valid/labels/b.txt": b"1 0.4 0.4 0.1 0.1\n",
        "data.yaml": b"names: [x, y]\n",
    }
    server.add(f"/api/datasets/{repo}/revision/main", {"sha": SHA, "gated": False})
    server.add(
        f"/api/datasets/{repo}/tree/{SHA}?recursive=true",
        [{"type": "file", "path": k, "size": len(v)} for k, v in files.items()],
    )
    for k, v in files.items():
        server.add(f"/datasets/{repo}/resolve/{SHA}/{k}", v)
    adapter = make_adapter(
        "hf", {"repo": repo, "endpoint": server.base, "label_patterns": ["*/labels/*.txt"]}
    )
    adapter.resolve_version()
    writer = StagedWriter(
        tmp_path / "stage", WriterConfig("s", "v", "CC-BY-4.0", "x", dt.date(2026, 1, 1))
    )
    for item, _, decoded in adapter.samples(tmp_path):
        writer.add(item, decoded)
    writer.finalize()
    assert [r.split_hint for r in writer.rows] == ["train", "val"]
    assert [r.label_refs for r in writer.rows] == [
        ("labels/files/train_labels_a.txt",),
        ("labels/files/valid_labels_b.txt",),
    ]
    assert (tmp_path / "stage/labels/files/train_labels_a.txt").read_bytes().startswith(b"0 ")
