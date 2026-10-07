"""local_dir: already-on-disk manual downloads, no network (adapter: local_dir).

Walks ``params.path`` in sorted order via ``os.walk``, skips symlinks (files and
dirs), and hands every real file to the existing fetch/decode pipeline through a
``file://`` URL -- no adapter-specific fetch/decode code.
"""

from __future__ import annotations

import os

from _wp6_fixtures import png

from marinedata.adapters import make_adapter


def test_lists_images_sorted_skips_symlink_and_excludes_non_matching_suffix(tmp_path):
    root = tmp_path / "data"
    (root / "a").mkdir(parents=True)
    (root / "b").mkdir()
    (root / "a" / "img2.jpg").write_bytes(png(2))
    (root / "a" / "img1.jpg").write_bytes(png(1))
    (root / "b" / "notes.txt").write_text("not an image")
    target = tmp_path / "outside.jpg"
    target.write_bytes(png(3))
    os.symlink(target, root / "b" / "linked.jpg")

    adapter = make_adapter("local_dir", {"path": str(root)})
    items = list(adapter.enumerate())
    keys = [i.key for i in items]

    assert keys == ["a/img1.jpg", "a/img2.jpg"]  # sorted; txt + symlink dropped
    assert all(i.url.startswith("file://") for i in items)

    fetched = adapter.fetch(items[0], tmp_path / "work")
    decoded = next(adapter.decode(fetched))
    assert decoded.data == png(1)


def test_label_patterns_carry_non_image_files_as_plain_label_files(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "img.jpg").write_bytes(png(1))
    (root / "meta.csv").write_text("id,label\n1,fish\n")

    adapter = make_adapter("local_dir", {"path": str(root), "label_patterns": ["*.csv"]})
    items = {i.key: i for i in adapter.enumerate()}
    assert set(items) == {"img.jpg", "meta.csv"}

    fetched = adapter.fetch(items["meta.csv"], tmp_path / "work")
    decoded = next(adapter.decode(fetched))
    assert decoded.label_files == {"meta.csv": b"id,label\n1,fish\n"}


def test_include_exclude_globs_and_version_override(tmp_path):
    root = tmp_path / "data"
    (root / "keep").mkdir(parents=True)
    (root / "skip").mkdir()
    (root / "keep" / "a.jpg").write_bytes(png(1))
    (root / "skip" / "b.jpg").write_bytes(png(2))

    adapter = make_adapter(
        "local_dir", {"path": str(root), "include": ["keep/*"], "version": "zip-abc123"}
    )
    assert adapter.resolve_version() == "zip-abc123"
    keys = [i.key for i in adapter.enumerate()]
    assert keys == ["keep/a.jpg"]


def test_never_yields_outside_path_even_via_symlinked_subdir(tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "x.jpg").write_bytes(png(9))
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.jpg").write_bytes(png(1))
    os.symlink(outside, root / "linked_dir")

    adapter = make_adapter("local_dir", {"path": str(root)})
    keys = [i.key for i in adapter.enumerate()]
    assert keys == ["a.jpg"]  # symlinked dir never walked into
