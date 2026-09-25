"""WP-10 dedup v2: features, multi-index Hamming search, grouping, confirmation, gate."""

from __future__ import annotations

import csv
import inspect
import io
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
pa = pytest.importorskip("pyarrow")

from PIL import Image, ImageOps  # noqa: E402

from marinedata.cli import main  # noqa: E402
from marinedata.dedup.confirm import ConfirmRules, confirm  # noqa: E402
from marinedata.dedup.corpus import run_corpus  # noqa: E402
from marinedata.dedup.crop import Thumb, crop_refine  # noqa: E402
from marinedata.dedup.embed import match_patch  # noqa: E402
from marinedata.dedup.features import (  # noqa: E402
    features_from_bytes,
    features_from_image,
    phash_distance,
)
from marinedata.dedup.groups import GroupRecord, dup_clusters, run_gate, split_groups  # noqa: E402
from marinedata.dedup.mih import MultiIndexHamming, probe_masks, unique_pairs  # noqa: E402
from marinedata.neardup import dhash_file  # noqa: E402

CROP_AUDIT_TSV = Path(__file__).resolve().parents[1] / "docs" / "dedup-crop-audit-2026-09-25.tsv"


def _image(seed: int = 0, size: tuple[int, int] = (160, 120)) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size[1] // 8, size[0] // 8, 3), dtype=np.uint8)
    return Image.fromarray(base).resize(size, Image.Resampling.BICUBIC)


def _encode(img: Image.Image, fmt: str, **kw: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return buf.getvalue()


def test_dhash_is_bit_identical_to_v1(tmp_path: Path) -> None:
    path = tmp_path / "a.png"
    _image(1).save(path)
    assert features_from_bytes(path.read_bytes()).dhash == dhash_file(path)


def test_flip_variants_match_flipped_image() -> None:
    img = _image(2)
    f = features_from_image(img, "x")
    fh = features_from_image(ImageOps.mirror(img), "y")
    fv = features_from_image(ImageOps.flip(img), "z")
    assert bin(f.dhash_h ^ fh.dhash).count("1") <= 2
    assert bin(f.dhash_v ^ fv.dhash).count("1") <= 2
    assert phash_distance(f.phash_h, fh.phash) <= 8


def test_pixel_sha_survives_lossless_reencode_and_jpeg_stays_close() -> None:
    img = _image(3)
    png, bmp = features_from_bytes(_encode(img, "PNG")), features_from_bytes(_encode(img, "BMP"))
    assert png.sha256 != bmp.sha256 and png.pixel_sha256 == bmp.pixel_sha256
    jpg = features_from_bytes(_encode(img, "JPEG", quality=40))
    assert jpg.pixel_sha256 != png.pixel_sha256
    assert phash_distance(jpg.phash, png.phash) <= 40
    other = features_from_bytes(_encode(_image(99), "PNG"))
    assert phash_distance(other.phash, png.phash) > 60


def test_flat_image_is_lowtex() -> None:
    flat = Image.new("RGB", (64, 48), (10, 80, 140))
    assert features_from_image(flat, "f").lowtex
    assert not features_from_image(_image(4), "t").lowtex


def test_probe_masks_counts() -> None:
    assert [len(probe_masks(r)) for r in (0, 1, 2)] == [1, 17, 137]


@pytest.mark.parametrize("radius", [3, 7, 9])
def test_mih_matches_brute_force(radius: int) -> None:
    rng = np.random.default_rng(radius)
    base = rng.integers(0, 2**62, 300, dtype=np.int64).astype(np.uint64)
    near = base[:100] ^ (np.uint64(1) << rng.integers(0, 64, 100).astype(np.uint64))
    far = base[100:150] ^ np.uint64(0b1011 << 20)
    codes = np.concatenate([base, near, far])
    index = MultiIndexHamming.build(codes)
    a, b, d = unique_pairs(index.search(codes, radius, self_join=True, max_pairs=37), len(codes))
    got = set(zip(a.tolist(), b.tolist(), d.tolist(), strict=True))
    want = {
        (i, j, int(np.bitwise_count(codes[i] ^ codes[j])))
        for i, j in combinations(range(len(codes)), 2)
        if int(np.bitwise_count(codes[i] ^ codes[j])) <= radius
    }
    assert got == want and len(want) >= 100


def test_groups_union_dups_and_declared_keys() -> None:
    clusters = dup_clusters(["a1", "b2", "c3", "d4"], [("a1", "b2")])
    assert clusters["a1"] == clusters["b2"] != clusters["c3"]
    records = [
        GroupRecord("r1", "a1", ("seq:x/1",), "train"),
        GroupRecord("r2", "b2", (), None),
        GroupRecord("r3", "c3", ("seq:x/1",), "test"),
        GroupRecord("r4", "d4", (), None),
    ]
    groups, upstream = split_groups(records, clusters)
    assert groups["a1"] == groups["b2"] == groups["c3"] != groups["d4"]
    assert upstream[groups["a1"]] == ["test", "train"]
    assert groups["a1"] == "sg-a1"


def test_gate_flags_spanning_and_upstream_test() -> None:
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3"}
    upstream = {"g2": ["test"]}
    ok = run_gate([("a", "train"), ("b", "train"), ("d", "test")], groups, upstream)
    assert ok.ok
    bad = run_gate([("a", "train"), ("b", "test"), ("c", "train"), ("e", "val")], groups, upstream)
    assert not bad.ok
    assert set(bad.spanning) == {"g1"} and bad.upstream_test_in_train == ["g2"]
    assert bad.ungrouped == ["e"]
    assert run_gate([("e", "train")], groups, {}, allow_ungrouped=True).ok


def test_confirm_lowtex_guard() -> None:
    scores = {
        "cos": np.array([0.8, 0.8, 0.9, 0.2, 0.55]),
        "d_dhash": np.array([2, 2, 2, 0, 30]),
        "d_phash": np.array([10, 10, 10, 0, 90]),
        "exact": np.array([False, False, False, True, False]),
        "pixel": np.array([False, False, False, True, False]),
        "lowtex": np.array([False, True, True, True, False]),
    }
    _, kind = confirm(scores, ConfirmRules())
    assert kind.tolist() == ["copy", "", "lowtex", "exact", ""]


def test_match_patch_finds_crop() -> None:
    parent = _image(5, (400, 300))
    patch = parent.crop((120, 90, 220, 165))
    m = match_patch(patch, parent)
    assert m.score > 0.9
    assert abs(m.box[0] - 120) <= 6 and abs(m.box[1] - 90) <= 6


def _thumb_pair(img: Image.Image) -> Thumb:
    return Thumb(gray=img.convert("L"), rgb=img.convert("RGB"))


def test_run_corpus_dedup_crop_flag_defaults_off() -> None:
    """WP-10c: v2 dedup ships with the crop channel off unless --dedup-crop is passed."""
    assert inspect.signature(run_corpus).parameters["dedup_crop"].default is False


def test_cli_dedup_crop_flag_defaults_off() -> None:
    import argparse

    from marinedata.cli_dedup import add_dedup_subparser

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    add_dedup_subparser(sub)
    ns = parser.parse_args(["dedup", "run", "--out", "/tmp/wp10c-cli-default-check"])
    assert ns.dedup_crop is False


def test_confirm_alone_rejects_known_false_cross_corpus_pairs() -> None:
    """WP-10b audit (2026-09-25): C008/C023/C103 are unrelated cross-corpus scenes that only
    the crop channel ever confirmed (cos/d_dhash recorded in the audit). With the crop channel
    off, copy/agree/pixel/lowtex alone must not merge them."""
    scores = {
        "cos": np.array([0.5033, 0.5065, 0.5344], dtype=np.float32),
        "d_dhash": np.array([26, 36, 21], dtype=np.int16),
        "d_phash": np.array([200, 200, 200], dtype=np.int16),
        "exact": np.array([False, False, False]),
        "pixel": np.array([False, False, False]),
        "lowtex": np.array([False, False, False]),
    }
    ok, kind = confirm(scores, ConfirmRules())
    assert not ok.any()
    assert kind.tolist() == ["", "", ""]


def test_crop_audit_fixture_composition() -> None:
    """The 185-pair WP-10c audit set (hashes/ids only, no images): composition frozen so a
    future re-run notices drift. 3 derivative + 33 same_scene + 149 different = 185."""
    with CROP_AUDIT_TSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    assert len(rows) == 185
    assert Counter(r["verdict"] for r in rows) == {
        "different": 149,
        "same_scene": 33,
        "derivative": 3,
    }
    cross = [r for r in rows if r["cross_corpus"] == "1"]
    assert {r["pair_id"] for r in cross} == {"C008", "C023", "C103"}
    assert all(r["verdict"] == "different" for r in cross)


def test_crop_refine_scale_floor_can_reject_a_true_crop() -> None:
    """``crop_scale_min`` restricts the scales match_patch tries; excluding the true scale
    starves the NCC peak, so a genuine crop can miss confirmation while the floor stands."""
    parent = _image(6, (400, 300))
    patch = parent.crop((120, 90, 280, 210))  # 160x120 at scale 160/400 = 0.40
    pt, pr = _thumb_pair(patch), _thumb_pair(parent)
    scores = {"cos": np.array([0.6], dtype=np.float32), "lowtex": np.array([False])}
    ok, kind = np.array([False]), np.array([""], dtype=object)
    area_a = np.array([patch.width * patch.height])
    area_b = np.array([parent.width * parent.height])

    rules = ConfirmRules(ncc_crop=0.8, crop_scale_min=0.0)
    _, kind1, ncc1 = crop_refine(
        scores, ok, kind, area_a, area_b, lambda k: pt, lambda k: pr, rules
    )
    assert kind1[0] == "crop" and ncc1[0] > 0.8

    floored = ConfirmRules(ncc_crop=0.8, crop_scale_min=0.6)  # excludes the true 0.40 scale
    _, kind2, _ = crop_refine(scores, ok, kind, area_a, area_b, lambda k: pt, lambda k: pr, floored)
    assert kind2[0] != "crop"


def test_crop_refine_requires_box_cos_confirmation() -> None:
    """The NCC peak only locates the box; ``box_cos`` re-verifies its content and can veto or
    confirm a pair NCC alone would have accepted."""
    parent = _image(7, (400, 300))
    patch = parent.crop((120, 90, 280, 210))
    pt, pr = _thumb_pair(patch), _thumb_pair(parent)
    scores = {"cos": np.array([0.6], dtype=np.float32), "lowtex": np.array([False])}
    ok, kind = np.array([False]), np.array([""], dtype=object)
    area_a = np.array([patch.width * patch.height])
    area_b = np.array([parent.width * parent.height])
    rules = ConfirmRules(ncc_crop=0.8, crop_scale_min=0.0, cos_box_crop=0.9)

    rejected, _, _ = crop_refine(
        scores, ok, kind, area_a, area_b, lambda k: pt, lambda k: pr, rules, box_cos=lambda *a: 0.1
    )
    assert not rejected.any()

    accepted, akind, _ = crop_refine(
        scores, ok, kind, area_a, area_b, lambda k: pt, lambda k: pr, rules, box_cos=lambda *a: 0.95
    )
    assert accepted.all() and akind[0] == "crop"


def test_cli_gate(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    rel = tmp_path / "rel"
    (rel / "manifests").mkdir(parents=True)
    (rel / "manifests" / "t.tsv").write_text("image_sha256\tsplit\na\ttrain\nb\ttest\n")
    groups = tmp_path / "groups.parquet"
    pq.write_table(
        pa.table(
            {"sha256": ["a", "b"], "split_group_id": ["g", "g"], "group_upstream_splits": [[], []]}
        ),
        groups,
    )
    assert main(["dedup", "gate", str(rel), "--groups", str(groups), "--write"]) == 1
    assert (rel / "DEDUP_GATE.json").exists()
    pq.write_table(
        pa.table(
            {"sha256": ["a", "b"], "split_group_id": ["g", "h"], "group_upstream_splits": [[], []]}
        ),
        groups,
    )
    assert main(["dedup", "gate", str(rel), "--groups", str(groups)]) == 0
