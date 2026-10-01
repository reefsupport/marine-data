"""WP-6d-B: multipart-tar, rar, video-frame, rosbag, caption/VQA JSON decoders, and the
D-R4 enumeration root-cause fix (0 stageable items -> a real non-zero exit, not a
hand-maintained lookup table). Fixtures are tiny and built in-test."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("PIL")
pytest.importorskip("pyarrow")

from _wp6_fixtures import LocalServer, png, tar_bytes

from marinedata.adapters import (
    NoStageableItems,
    RemoteItem,
    _classify_empty,
    make_adapter,
)


def _run(adapter, tmp_path: Path):
    return [(item.key, d) for item, _, d in adapter.samples(tmp_path)]


# ---------------------------------------------------------------------------
# D-R4 root cause: 0 stageable items is a real exception, classified by kind.
# ---------------------------------------------------------------------------


def test_classify_empty_kinds():
    def item(key: str) -> RemoteItem:
        return RemoteItem(key=key, url=f"http://x/{key}")

    assert _classify_empty([]) == "unknown-empty"
    assert _classify_empty([item("a.mp4")]) == "video"
    assert _classify_empty([item("a.rar")]) == "rar"
    assert _classify_empty([item("a.bag")]) == "rosbag"
    assert _classify_empty([item("a.json")]) == "json-captions"
    assert _classify_empty([item("a.tar.gz.aa"), item("a.tar.gz.ab")]) == "multipart-tar"
    assert _classify_empty([item("a.psd")]) == "unknown-container"


def test_enumerate_raises_nostageableitems_never_a_false_ok(server, tmp_path):
    server.add("/a.psd", b"binary junk")
    adapter = make_adapter(
        "http",
        {"urls": [{"url": f"{server.base}/a.psd", "key": "a.psd"}], "version": "v1"},
    )
    with pytest.raises(NoStageableItems) as exc:
        list(adapter.enumerate())
    assert exc.value.reason == "unknown-container"
    assert "needs_adapter:unknown-container" in str(exc.value)


def test_cli_dry_run_exits_4_on_zero_items(tmp_path):
    from marinedata.cli_ingest_source import _cmd_ingest_source

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        "id: fixture\nadapter: http\nlicense: cc0\nattribution: x\n"
        "params:\n  version: v1\n  urls: []\n"
    )
    ns = type(
        "NS",
        (),
        dict(
            adapter="http",
            spec=str(spec_path),
            dry_run=True,
            work=str(tmp_path),
            max_images=None,
            part_size_mib=64,
            jobs=1,
            max_per_host=4,
            part_jobs=4,
            fetch_only=False,
            limit=None,
        ),
    )()
    assert _cmd_ingest_source(ns) == 4


@pytest.fixture
def server():
    srv = LocalServer()
    yield srv
    srv.close()


# ---------------------------------------------------------------------------
# multipart-tar: N parts concatenated with no full local copy.
# ---------------------------------------------------------------------------


def test_multipart_tar_streams_concatenated_parts(server, tmp_path):
    whole = tar_bytes({"img/a.png": png(1)})
    chunk_size = max(1, len(whole) // 3)
    thirds = [whole[i : i + chunk_size] for i in range(0, len(whole), chunk_size)]
    urls = []
    for i, chunk in enumerate(thirds):
        letter = chr(ord("a") + i)
        key = f"seamapd21.tar.gz.a{letter}"
        server.add(f"/{key}", chunk)
        urls.append({"url": f"{server.base}/{key}", "key": key})
    adapter = make_adapter("http", {"urls": urls, "version": "v1"})
    items = list(adapter.enumerate())
    assert len(items) == 1
    assert items[0].key == "seamapd21.tar.gz"
    assert len(items[0].parts) == len(thirds)
    out = _run(adapter, tmp_path)
    assert out == [("seamapd21.tar.gz", out[0][1])]
    assert out[0][1].data == png(1)


# ---------------------------------------------------------------------------
# rar: extracted member-by-member via bsdtar (libarchive sniffs format from content).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not shutil.which("bsdtar"), reason="bsdtar not on PATH")
def test_rar_extracted_via_bsdtar(server, tmp_path):
    # No RAR-creation tool exists in this sandbox (RAR compression is proprietary); a
    # plain tar's bytes under a `.rar` name still exercises the real bsdtar subprocess
    # path, since libarchive's format detection sniffs content, never the extension.
    payload = tar_bytes({"img/a.png": png(2)})
    server.add("/underwater-images-2542305.rar", payload)
    adapter = make_adapter(
        "http",
        {
            "urls": [{"url": f"{server.base}/underwater-images-2542305.rar"}],
            "version": "v1",
        },
    )
    out = _run(adapter, tmp_path)
    assert len(out) == 1
    assert out[0][1].data == png(2)


# ---------------------------------------------------------------------------
# video -> frames: 1 fps sampling + dHash near-duplicate drop; video_id/frame_ts labels.
# ---------------------------------------------------------------------------


def _make_video(path: Path, *, duration: int, changing: bool) -> None:
    src = "testsrc2=size=32x32:rate=2" if changing else "color=c=red:size=32x32:rate=2"
    subprocess.run(
        [
            shutil.which("ffmpeg"),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"{src}:duration={duration}",
            str(path),
        ],
        check=True,
    )


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not on PATH")
def test_video_frames_sampled_and_deduped(server, tmp_path):
    still = tmp_path / "still.mp4"
    _make_video(still, duration=3, changing=False)
    server.add("/salmon-cage.mp4", still.read_bytes())
    adapter = make_adapter(
        "http", {"urls": [{"url": f"{server.base}/salmon-cage.mp4"}], "version": "v1"}
    )
    out = _run(adapter, tmp_path / "w1")
    # a static-colour source: every 1fps frame is near-identical -> deduped to one
    assert len(out) == 1
    _key, decoded = out[0]
    assert decoded.labels["video_id"] == "salmon-cage"
    assert decoded.labels["frame_ts"] == "0.000"
    assert decoded.suffix == ".jpg"

    changing = tmp_path / "changing.mp4"
    _make_video(changing, duration=3, changing=True)
    server.add("/salmon-cage2.mp4", changing.read_bytes())
    adapter2 = make_adapter(
        "http", {"urls": [{"url": f"{server.base}/salmon-cage2.mp4"}], "version": "v1"}
    )
    out2 = _run(adapter2, tmp_path / "w2")
    assert len(out2) >= 2  # a moving pattern is not deduped away
    assert {d.labels["video_id"] for _, d in out2} == {"salmon-cage2"}


# ---------------------------------------------------------------------------
# rosbag: sensor_msgs/CompressedImage via pure-python `rosbags`; topic/stamp labels.
# ---------------------------------------------------------------------------


def _make_bag(path: Path) -> None:
    import numpy as np
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    ts = get_typestore(Stores.ROS1_NOETIC)
    msg_t = ts.types["sensor_msgs/msg/CompressedImage"]
    header_t = ts.types["std_msgs/msg/Header"]
    time_t = ts.types["builtin_interfaces/msg/Time"]
    # PNG bytes, not real JPEG compression — CompressedImage.format is what decode.py
    # trusts for the suffix; two distinct images so the dHash dedup keeps both.
    frames = [png(3), png(30)]

    with Writer(path) as writer:
        conn = writer.add_connection("/camera/image/compressed", msg_t.__msgtype__, typestore=ts)
        for i, jpg in enumerate(frames):
            msg = msg_t(
                header=header_t(seq=i, stamp=time_t(sec=i, nanosec=0), frame_id="cam"),
                format="png",
                data=np.frombuffer(jpg, dtype=np.uint8),
            )
            writer.write(conn, i * 1_000_000_000, ts.serialize_ros1(msg, msg_t.__msgtype__))


def test_rosbag_compressed_image_extracted(server, tmp_path):
    bag = tmp_path / "fjord_5.bag"
    _make_bag(bag)
    server.add("/fjord_5.bag", bag.read_bytes())
    adapter = make_adapter(
        "http", {"urls": [{"url": f"{server.base}/fjord_5.bag"}], "version": "v1"}
    )
    out = _run(adapter, tmp_path / "w")
    assert len(out) == 2
    for _, d in out:
        assert d.labels["bag_id"] == "fjord_5"
        assert d.labels["topic"] == "/camera/image/compressed"
    assert {d.data for _, d in out} == {png(3), png(30)}


# ---------------------------------------------------------------------------
# caption/VQA JSON: label-only records, image resolved via index or fetched by URL.
# ---------------------------------------------------------------------------


def test_caption_json_resolves_index_and_url_counts_unresolved(server, tmp_path):
    img = png(4)
    server.add("/pics/a.png", img)
    records = [
        {"image": "sha-a", "caption": "a fish"},
        {"image": f"{server.base}/pics/a.png", "caption": "same fish, by url"},
        {"image": "sha-missing", "caption": "no image anywhere"},
    ]
    server.add("/oceaninstruct.json", json.dumps(records).encode(), ctype="application/json")
    adapter = make_adapter(
        "http",
        {
            "urls": [{"url": f"{server.base}/oceaninstruct.json"}],
            "version": "v1",
            "image_index": {"sha-a": f"{server.base}/pics/a.png"},
        },
    )
    out = _run(adapter, tmp_path)
    assert len(out) == 3
    resolved = [d for _, d in out if d.labels["unresolved"] == "false"]
    unresolved = [d for _, d in out if d.labels["unresolved"] == "true"]
    assert len(resolved) == 2
    assert all(d.data == img for d in resolved)
    assert len(unresolved) == 1
    assert unresolved[0].labels["caption"] == "no image anywhere"


def test_caption_json_coco_images_list_and_own_split(server, tmp_path):
    """BENCH-fix3 B: a COCO-detection file (``images``/``annotations``/``categories``,
    no ``data``/``records`` key — e.g. fathomnet-vme's ``coco_*.json``) used to fall
    back to ``[whole_dict]``: one always-unresolved, empty-bytes record per file. The
    fix tries ``images`` as the record list and reads each record's own ``split``."""
    img = png(4)
    server.add("/pics/a.jpg", img)
    coco = {
        "images": [
            {
                "id": 1,
                "file_name": "a.jpg",
                "source_url": f"{server.base}/pics/a.jpg",
                "split": "test",
            }
        ],
        "annotations": [{"image_id": 1, "category_id": 0}],
        "categories": [{"id": 0, "name": "coral"}],
    }
    server.add("/coco_test.json", json.dumps(coco).encode(), ctype="application/json")
    adapter = make_adapter(
        "http",
        {
            "urls": [{"url": f"{server.base}/coco_test.json"}],
            "version": "v1",
            "caption_image_field": "source_url",
        },
    )
    out = _run(adapter, tmp_path)
    assert len(out) == 1
    _, d = out[0]
    assert d.data == img
    assert d.labels["unresolved"] == "false"
    assert d.split_hint == "test"


def test_caption_json_eval_splits_skips_before_fetch(server, tmp_path):
    """BENCH-vmfix: when ``eval_splits`` is set, a record outside it is skipped
    BEFORE the image fetch (no network request for its image), not fetched-then-
    discarded. ``coco_train.json``'s image is never requested; val + test yield."""

    def coco(image_path: str) -> dict:
        return {"images": [{"id": 1, "file_name": image_path, "source_url": image_path}]}

    server.add("/pics/train.jpg", png(1))
    server.add("/pics/val.jpg", png(2))
    server.add("/pics/test.jpg", png(3))
    server.add("/coco_train.json", coco(f"{server.base}/pics/train.jpg"))
    server.add("/coco_val.json", coco(f"{server.base}/pics/val.jpg"))
    server.add("/coco_test.json", coco(f"{server.base}/pics/test.jpg"))
    adapter = make_adapter(
        "http",
        {
            "urls": [
                {"url": f"{server.base}/coco_train.json"},
                {"url": f"{server.base}/coco_val.json"},
                {"url": f"{server.base}/coco_test.json"},
            ],
            "version": "v1",
            "caption_image_field": "source_url",
            "eval_splits": ["test", "val"],
        },
    )
    out = _run(adapter, tmp_path)
    splits = {d.split_hint for _, d in out}
    assert splits == {"val", "test"}
    # 3 json GETs + 2 image GETs (val, test) — train's image is never requested.
    assert len(server.seen_headers) == 5
