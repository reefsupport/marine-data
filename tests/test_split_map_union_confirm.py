"""Rule B's confirm-hash (v1.1.0): a dHash pair within the union radius is only a candidate.

On v1.1.0 (80,518 images, 17 sources) dHash-4 unions chained 886 groups into one 5,151-image
component — Red Sea Coralscapes sites joined to Hawaiian NOAA PIFSC sites — because a 9x8 dHash of
a low-information frame is decided by the illumination (a falloff, a gradient) or, on a flat frame,
by nothing at all. These tests stage real images: three "sites" chained through such frames stay
separate, a genuine re-encode is still one component, and the chain guard still fires on it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_neardup import (
    _config,
    _encode,
    _picture,
    _registry,
    _source,
    _stage,
    _two_sources_with_twin,
)

from marinedata.decon import _entropy_bits
from marinedata.neardup import (
    NearDupChainError,
    NearDupConfig,
    NearDupError,
    UnionConfirm,
    compute_confirm_features,
    compute_dhashes,
    grey_entropy_bits,
    near_dup_record,
    near_pairs,
)
from marinedata.release import generate_split_map
from marinedata.splitmap import load_split_map

NOW = "2026-10-16T00:00:00Z"
V1_MAP = Path(__file__).resolve().parents[1] / "registry" / "SPLIT_MAP.json"


def _lit_frame(seed: int) -> Image.Image:
    """A left-to-right light falloff over fine, site-specific texture: the 9x8 dHash sees only the
    falloff (every bit ``right > left``), the 64x64 pHash-256 sees the texture."""
    rng = np.random.default_rng(seed)
    ramp = np.linspace(40.0, 200.0, 256)[None, :].repeat(256, axis=0)
    cells = rng.uniform(-30.0, 30.0, (64, 64)).repeat(4, axis=0).repeat(4, axis=1)
    return Image.fromarray(np.clip(ramp + cells, 0, 255).astype(np.uint8)).convert("RGB")


def _flat_frame(grey: int) -> Image.Image:
    return Image.new("RGB", (96, 96), (grey // 3, grey // 2, grey))


def _three_sites(tmp_path: Path):  # type: ignore[no-untyped-def]
    roots, sources = {}, {}
    for n, sid in enumerate(("src-a", "src-b", "src-c")):
        rows = [(f"p{i}", f"{sid}/site", _encode(_picture(10 * n + i), "PNG")) for i in range(4)]
        rows += [("lit", f"{sid}/site", _encode(_lit_frame(n), "PNG"))]
        rows += [("flat", f"{sid}/site", _encode(_flat_frame(60 + 7 * n), "PNG"))]
        rows += [
            (f"u{i}", f"{sid}/g{i}", _encode(_picture(100 + 10 * n + i), "PNG")) for i in range(6)
        ]
        roots[sid] = _stage(tmp_path / sid, rows)
        sources[sid] = _source(sid)
    return _registry(sources), roots


def test_three_sites_chained_by_low_information_frames_stay_separate(tmp_path: Path) -> None:
    registry, roots = _three_sites(tmp_path)
    frames = {
        f"{s}/{k}": roots[s] / "images" / "p" / f"{k}.img" for s in roots for k in ("lit", "flat")
    }
    hashes = compute_dhashes(frames, cache_dir=tmp_path / "probe")
    bridges = {(a, b) for a, b, _ in near_pairs(hashes, 4)}
    assert {("src-a/lit", "src-b/lit"), ("src-b/lit", "src-c/lit")} <= bridges  # dHash chains them
    assert {("src-a/flat", "src-b/flat"), ("src-b/flat", "src-c/flat")} <= bridges

    with pytest.raises(NearDupChainError, match="chain guard"):  # v1.0's rule B: one component
        generate_split_map(
            registry, out=tmp_path / "v1.json", roots=roots, near_dup=_config(tmp_path)
        )

    out = tmp_path / "SPLIT_MAP.json"
    stats = generate_split_map(
        registry,
        out=out,
        roots=roots,
        now=NOW,
        near_dup=NearDupConfig(cache_dir=tmp_path / "dhash", union_confirm=UnionConfirm()),
    )
    assert stats.near_dup_candidates == 6
    assert (stats.near_dup_rejected_phash, stats.near_dup_rejected_entropy) == (3, 3)
    assert (stats.near_dup_unions, stats.near_dup_max_component) == (0, 0)
    split_map = load_split_map(out)
    assert split_map is not None
    assert split_map.near_dup["union_confirm"] == UnionConfirm().record()
    assert split_map.near_dup["union_max_hamming"] == 4  # the candidate radius is unchanged


def test_genuine_reencode_is_confirmed_and_a_real_chain_still_trips_the_guard(
    tmp_path: Path,
) -> None:
    registry, roots = _two_sources_with_twin(tmp_path)
    confirm = UnionConfirm()
    stats = generate_split_map(
        registry,
        out=tmp_path / "SPLIT_MAP.json",
        roots=roots,
        now=NOW,
        near_dup=NearDupConfig(cache_dir=tmp_path / "d", chain_fraction=0.5, union_confirm=confirm),
    )
    assert (stats.near_dup_candidates, stats.near_dup_unions, stats.near_dup_max_component) == (
        1,
        1,
        2,
    )
    split_map = load_split_map(tmp_path / "SPLIT_MAP.json")
    assert (
        split_map is not None and split_map.assignments["a/twin"] == split_map.assignments["b/twin"]
    )
    with pytest.raises(NearDupChainError, match=r"chain guard.*2 images"):
        generate_split_map(
            registry,
            out=tmp_path / "M2.json",
            roots=roots,
            near_dup=NearDupConfig(cache_dir=tmp_path / "d", union_confirm=confirm),
        )
    assert not (tmp_path / "M2.json").exists()


def test_v1_record_keeps_the_frozen_map_keys_and_v11_adds_only_union_confirm(
    tmp_path: Path,
) -> None:
    frozen = json.loads(V1_MAP.read_text())["near_dup"]
    v1 = near_dup_record(NearDupConfig(cache_dir=tmp_path))
    assert v1.keys() == frozen.keys()
    assert {k: v for k, v in v1.items() if k != "pil_version"} == {
        k: v for k, v in frozen.items() if k != "pil_version"
    }
    v11 = near_dup_record(NearDupConfig(cache_dir=tmp_path, union_confirm=UnionConfirm()))
    assert v11.keys() - v1.keys() == {"union_confirm"}
    assert (v11["union_confirm"]["phash256_max"], v11["union_confirm"]["min_entropy_bits"]) == (
        32,
        4.0,
    )


def test_entropy_is_decons_s3_definition(tmp_path: Path) -> None:
    for img in (_lit_frame(3), _flat_frame(90), _picture(7)):
        data = _encode(img, "PNG")
        with Image.open(__import__("io").BytesIO(data)) as decoded:
            assert grey_entropy_bits(decoded) == _entropy_bits(lambda d=data: d)


def test_confirm_features_are_cached_by_sha_and_fail_closed(tmp_path: Path) -> None:
    good = tmp_path / "good.img"
    good.write_bytes(_encode(_picture(1), "JPEG"))
    first = compute_confirm_features({"s1": good}, cache_dir=tmp_path / "c")
    good.unlink()  # a cache hit never re-reads the file
    assert compute_confirm_features({"s1": good}, cache_dir=tmp_path / "c") == first
    assert len(first["s1"][0]) == 32 and first["s1"][1] > 4.0
    bad = tmp_path / "bad.img"
    bad.write_bytes(b"not an image")
    with pytest.raises(NearDupError, match="1 image"):
        compute_confirm_features({"s2": bad}, cache_dir=tmp_path / "c")


def test_rejected_new_prior_pairs_keep_the_prior_release_band_constraint() -> None:
    """A rejected pair inside the union radius never constrains less than one in the band: a new
    image it pairs with a prior-release image is never-eval / never-train exactly as at 5-8 bits."""
    from marinedata.neardup import UnionConfirm, confirm_pairs
    from marinedata.release.split_map_inherit import unconfirmed_band_pairs
    from marinedata.splitmap_inherit import TRAIN, band_constraints

    feats = {"a": (bytes(32), 8.0), "b": (bytes(32), 8.0), "c": (b"\xff" * 32, 8.0)}
    feats["d"] = (bytes(32), 1.0)
    got = confirm_pairs([("a", "b", 0), ("a", "c", 1), ("a", "d", 2)], feats, UnionConfirm())
    assert got.links == [("a", "b")]
    assert got.rejected == [("a", "c", 1), ("a", "d", 2)]
    assert UnionConfirm().record()["rejected"].startswith("links nothing")

    prior = {"p-train": TRAIN, "p-test": "test"}
    rejected = [("p-train", "n-1", 2), ("n-2", "p-test", 0), ("n-3", "n-4", 1)]
    rejected += [("p-train", "p-test", 3), ("n-5", "p-train", 4), ("p-test", "n-5", 1)]
    band = unconfirmed_band_pairs(rejected, prior)
    assert band == [
        ("n-1", "p-train", 2),
        ("n-2", "p-test", 0),
        ("n-5", "p-train", 4),
        ("n-5", "p-test", 1),
    ]
    forced, excluded = band_constraints(
        band, v1_split=prior, component_of={s: f"c-{s}" for s in ("n-1", "n-2", "n-5")}
    )
    assert forced == {"c-n-1": TRAIN, "c-n-2": "test"}
    # n-5 hits a prior train and a prior test image: it keeps no anchor, so it alone is excluded
    assert excluded == {"n-5": (1, "p-test", "test")}
