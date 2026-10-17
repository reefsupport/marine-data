"""The gates' DCT pHash-64 rule in split-map generation (v1.1.0).

The dedup and splits gates fail a cross-split pair within 8 bits of ``gates.phash.phash``; run 5
shipped 70 of them because rule B only unions dHash <= 4 pairs that pass the confirm-hash. These
tests stage real images: a PNG and its JPEG re-encode, and check the hash, the cache and the link.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from test_neardup import _config, _two_sources_with_twin

from marinedata.gates.phash import ALGORITHM, phash
from marinedata.neardup import PHASH_ALGORITHM, UnionConfirm
from marinedata.neardup_phash import compute_phashes, phash_links
from marinedata.release import generate_split_map
from marinedata.splitmap import load_split_map


def test_the_hash_is_the_gates_function_and_is_cached_by_sha(tmp_path: Path) -> None:
    _, roots = _two_sources_with_twin(tmp_path)
    path = roots["src-a"] / "images/p/t.img"
    assert PHASH_ALGORITHM == ALGORITHM
    got = compute_phashes({"s": path}, cache_dir=tmp_path / "c")
    with Image.open(path) as image:
        assert got["s"] == phash(image)
    path.unlink()  # a cache hit never re-reads the file
    assert compute_phashes({"s": path}, cache_dir=tmp_path / "c") == got


def test_phash_links_pair_the_reencoded_twin_and_nothing_else(tmp_path: Path) -> None:
    _, roots = _two_sources_with_twin(tmp_path)
    paths = {
        "a": roots["src-a"] / "images/p/t.img",
        "b": roots["src-b"] / "images/p/t.img",
        "c": roots["src-a"] / "images/p/u0.img",
    }
    hashes = compute_phashes(paths, cache_dir=tmp_path / "c")
    twin, other = hashes["a"] ^ hashes["b"], hashes["a"] ^ hashes["c"]
    assert bin(twin).count("1") <= 8 < bin(other).count("1")
    assert phash_links(paths, cache_dir=tmp_path / "c", max_distance=8) == [("a", "b")]


def test_split_map_ties_a_phash_pair_that_rule_b_left_apart(tmp_path: Path) -> None:
    registry, roots = _two_sources_with_twin(tmp_path)
    flat = UnionConfirm(min_entropy_bits=99.0)  # every dHash candidate is rejected: B links nothing
    kw = {"registry": registry, "roots": roots, "now": "2026-10-17T00:00:00Z"}
    base = _config(tmp_path, union_confirm=flat, chain_fraction=0.5)
    plain = generate_split_map(out=tmp_path / "plain.json", near_dup=base, **kw)
    assert (plain.near_dup_pairs, plain.near_dup_phash_pairs) == (0, 0)
    gates_rule = _config(tmp_path, union_confirm=flat, chain_fraction=0.5, phash_max=8)
    out = tmp_path / "SPLIT_MAP.json"
    stats = generate_split_map(out=out, near_dup=gates_rule, **kw)
    got = (stats.near_dup_phash_pairs, stats.near_dup_unions, stats.near_dup_max_component)
    assert got == (1, 1, 2)
    split_map = load_split_map(out)
    assert split_map is not None
    assert split_map.assignments["a/twin"] == split_map.assignments["b/twin"]
    assert split_map.near_dup["phash_max_hamming"] == 8
    assert split_map.near_dup["phash_algorithm"] == ALGORITHM
