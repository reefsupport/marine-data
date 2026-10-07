"""ADAPTERS 2026-10-02: two new sources, no network, no upstream bytes.

* fathomnet-fgvc23: HF remote_zip + zip_manifest keyed on ``file_name`` (not the
  default ``image`` field) -- images.zip expands to exactly the members
  metadata.jsonl names, and metadata.jsonl itself stages as one label_patterns
  sidecar (the existing COCO-label path, no per-image join).
* deepseagrass: the new minimal ``csiro_dap`` adapter -- lists a DAP v2 collection's
  one-shot file listing and turns each entry into a plain ``RemoteItem``; combined
  with ``format: imagefolder`` the parent folder becomes the class label.
"""

from __future__ import annotations

import hashlib
import json

import pytest
from _wp6_fixtures import LocalServer, png, zip_bytes

from marinedata.adapters import make_adapter


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


# ---------------------------------------------------------------------------
# fathomnet-fgvc23: hf remote_zip + zip_manifest_field: file_name
# ---------------------------------------------------------------------------


def test_fgvc23_zip_manifest_expands_named_members_and_stages_metadata_sidecar(server):
    repo = "justinsiow/fathomnet"
    img_a = png(1)
    img_b = png(2)
    name_a = "3b6f01ae-5bde-434d-9b06-79b269421ed6.png"
    name_b = "dce21f7c-20e5-482b-bd90-c038f8464c03.png"
    images_zip = zip_bytes({name_a: img_a, name_b: img_b})
    metadata = b"\n".join(
        json.dumps(rec).encode()
        for rec in (
            {
                "image_id": 1,
                "file_name": name_a,
                "width": 720,
                "height": 368,
                "objects": [{"id": [1], "category_id": [1.0]}],
            },
            {
                "image_id": 2,
                "file_name": name_b,
                "width": 720,
                "height": 368,
                "objects": [{"id": [2], "category_id": [1.0]}],
            },
        )
    )
    files = {"images.zip": images_zip, "metadata.jsonl": metadata}
    sha = "a" * 40
    server.add(f"/api/datasets/{repo}/revision/main", {"sha": sha})
    tree = [{"type": "file", "path": p, "size": len(b)} for p, b in files.items()]
    server.add(f"/api/datasets/{repo}/tree/{sha}?recursive=true", tree)
    for p, b in files.items():
        server.add(f"/datasets/{repo}/resolve/{sha}/{p}", b)

    adapter = make_adapter(
        "hf",
        {
            "repo": repo,
            "endpoint": server.base,
            "include": list(files),
            "label_patterns": ["metadata.jsonl"],
            "remote_zip": True,
            "zip_manifest": ["metadata.jsonl"],
            "zip_manifest_field": "file_name",
        },
    )
    adapter.resolve_version()
    items = list(adapter.enumerate())
    keys = {i.key for i in items}

    assert f"images.zip#{name_a}" in keys
    assert f"images.zip#{name_b}" in keys
    assert "metadata.jsonl" in keys
    # nothing else leaked out of the zip (only the manifest-named members expand)
    assert sum(1 for k in keys if k.startswith("images.zip#")) == 2

    by_key = {i.key: i for i in items}
    fetched_a = adapter.fetch(by_key[f"images.zip#{name_a}"], None)  # type: ignore[arg-type]
    decoded_a = next(adapter.decode(fetched_a))
    assert decoded_a.data == img_a
    assert sha256(decoded_a.data) == sha256(img_a)

    fetched_meta = adapter.fetch(by_key["metadata.jsonl"], None)  # type: ignore[arg-type]
    decoded_meta = next(adapter.decode(fetched_meta))
    assert decoded_meta.data == b""
    assert decoded_meta.label_files == {"metadata.jsonl": metadata}


# ---------------------------------------------------------------------------
# deepseagrass: csiro_dap adapter
# ---------------------------------------------------------------------------


def test_csiro_dap_lists_files_and_imagefolder_labels_by_parent(server):
    img = png(3)
    txt = b"5-class instructions\n"
    server.add(
        "/dap/ws/v2/collections/47653/data",
        {
            "file": [
                {
                    "filename": "For_5-Class_Case/5ClassInstructions.txt",
                    "fileSize": len(txt),
                    "link": {"href": f"{server.base}/files/instructions.txt"},
                },
                {
                    "filename": "Training/Background/Image1051_Row1_Col0.jpg",
                    "fileSize": len(img),
                    "link": {"href": f"{server.base}/files/img1.jpg"},
                },
            ]
        },
    )
    server.add("/files/instructions.txt", txt)
    server.add("/files/img1.jpg", img)

    adapter = make_adapter(
        "csiro_dap",
        {
            "collection": "47653",
            "endpoint": f"{server.base}/dap/ws/v2",
            "exclude": ["*.txt"],
            "format": "imagefolder",
        },
    )
    adapter.resolve_version()
    items = list(adapter.enumerate())

    assert len(items) == 1
    item = items[0]
    assert item.key == "Training/Background/Image1051_Row1_Col0.jpg"
    assert item.url == f"{server.base}/files/img1.jpg"
    assert item.size == len(img)

    fetched = adapter.fetch(item, None)  # type: ignore[arg-type]
    decoded = next(adapter.decode(fetched))
    assert decoded.data == img
    assert decoded.labels.get("label") == "Background"
