"""Near-duplicate-safe splitting (WS-D S47): dHash, banded search, rules (A) and (B).

sha256 only merges byte-identical copies; S46 found re-encoded copies of eval images in
the never-eval ``coralscop-masks-rs`` pool at dHash distance 1. These tests stage real,
decodable images — including a JPEG re-encode of a PNG, i.e. different bytes, same photo.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from marinedata.enums import (
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import Access, Coverage, Licence, LoaderSpec, Profile, Source, Verification
from marinedata.neardup import (
    NearDupChainError,
    NearDupConfig,
    NearDupError,
    compute_dhashes,
    dhash_file,
    near_pairs,
    pil_version,
)
from marinedata.registry import Registry
from marinedata.release import build_release, generate_split_map
from marinedata.schema import Axis, LabelSchema
from marinedata.splitmap import SplitMap, load_split_map, save_split_map
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec


def _picture(seed: int) -> Image.Image:
    rng = random.Random(seed)
    small = Image.new("L", (9, 8))
    small.putdata([rng.randrange(256) for _ in range(72)])
    return small.resize((72, 64), Image.Resampling.BICUBIC).convert("RGB")


def _encode(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {}))
    return buf.getvalue()


def _source(source_id: str, tags: tuple[str, ...] = ()) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the near-dup tests.",
        version="v1",
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 9, 24), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(method=AccessMethod.HTTP, uri="https://example.invalid"),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout="staged-tree", params={}),
        annotations=(),
        tags=tags,
    )


def _stage(root: Path, rows: list[tuple[str, str, bytes]]) -> Path:
    for stem, _, data in rows:
        path = root / "images" / "p" / f"{stem}.img"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    write_metadata_table(
        root / "metadata.parquet",
        [
            StagedImage(
                stem=stem,
                partition="p",
                upstream_path=f"orig/{stem}",
                upstream_split=None,
                width=72,
                height=64,
                split_group=group,
            )
            for stem, group, _ in rows
        ],
    )
    return root


def _registry(sources: dict[str, Source]) -> Registry:
    return Registry(
        sources=sources,
        licences={},
        profiles={
            "research": Profile(
                id="research", description="fixture", allow_tiers=(Tier.OWN, Tier.PERMISSIVE)
            )
        },
        schemas={
            "rs-benthic-v1": LabelSchema(id="rs-benthic-v1", name="F", axes=(Axis.TAXON,), nodes=())
        },
        tasks={"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)},
    )


def _config(tmp_path: Path, **kw: float) -> NearDupConfig:
    return NearDupConfig(cache_dir=tmp_path / "dhash", **kw)  # type: ignore[arg-type]


def test_dhash_is_the_s46_definition(tmp_path: Path) -> None:
    path = tmp_path / "x.png"
    path.write_bytes(_encode(_picture(7), "PNG"))
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    expected = 0
    for bit in (arr[:, 1:] > arr[:, :-1]).flatten():
        expected = (expected << 1) | int(bit)
    assert dhash_file(path) == expected


def test_near_pairs_is_exact_against_brute_force() -> None:
    rng = random.Random(3)
    base = [rng.getrandbits(64) for _ in range(300)]
    planted = [h ^ (1 << rng.randrange(64)) ^ (1 << rng.randrange(64)) for h in base[:40]]
    hashes = {f"k{i:04d}": h for i, h in enumerate(base + planted)}
    for k in (0, 4, 8):
        brute = sorted(
            (a, b, (hashes[a] ^ hashes[b]).bit_count())
            for a in hashes
            for b in hashes
            if a < b and (hashes[a] ^ hashes[b]).bit_count() <= k
        )
        assert near_pairs(hashes, k) == brute
    query = {k: v for k, v in hashes.items() if k < "k0200"}
    target = {k: v for k, v in hashes.items() if k >= "k0150"}
    cross = sorted(
        (a, b, (query[a] ^ target[b]).bit_count())
        for a in query
        for b in target
        if (query[a] ^ target[b]).bit_count() <= 8
    )
    assert near_pairs(query, 8, target) == cross
    assert any(a == b for a, b, _ in cross)  # a key on both sides matches itself


def test_compute_dhashes_caches_by_sha_and_fails_closed(tmp_path: Path) -> None:
    good = tmp_path / "good.png"
    good.write_bytes(_encode(_picture(1), "PNG"))
    first = compute_dhashes({"sha-good": good}, cache_dir=tmp_path / "c")
    good.unlink()  # a cache hit never re-reads the file
    assert compute_dhashes({"sha-good": good}, cache_dir=tmp_path / "c") == first
    assert (tmp_path / "c" / f"dhash-pillow-{pil_version()}.sqlite").is_file()
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    with pytest.raises(NearDupError, match="could not be dHashed"):
        compute_dhashes({"sha-bad": bad}, cache_dir=tmp_path / "c")


def _two_sources_with_twin(tmp_path: Path) -> tuple[Registry, dict[str, Path]]:
    twin = _picture(100)
    rows_a = [("t", "a/twin", _encode(twin, "PNG"))]
    rows_b = [("t", "b/twin", _encode(twin, "JPEG"))]  # same photo, different bytes
    rows_a += [(f"u{i}", f"a/g{i}", _encode(_picture(i), "PNG")) for i in range(8)]
    rows_b += [(f"u{i}", f"b/g{i}", _encode(_picture(50 + i), "PNG")) for i in range(8)]
    roots = {"src-a": _stage(tmp_path / "a", rows_a), "src-b": _stage(tmp_path / "b", rows_b)}
    return _registry({"src-a": _source("src-a"), "src-b": _source("src-b")}), roots


def test_reencoded_twin_across_sources_is_one_component(tmp_path: Path) -> None:
    registry, roots = _two_sources_with_twin(tmp_path)
    assert dhash_file(roots["src-a"] / "images/p/t.img") != 0
    out = tmp_path / "SPLIT_MAP.json"
    stats = generate_split_map(
        registry,
        out=out,
        roots=roots,
        now="2026-09-24T00:00:00Z",
        near_dup=_config(tmp_path, chain_fraction=0.5),
    )
    assert (stats.near_dup_pairs, stats.near_dup_unions, stats.near_dup_max_component) == (1, 1, 2)
    split_map = load_split_map(out)
    assert split_map is not None
    assert split_map.assignments["a/twin"] == split_map.assignments["b/twin"]
    assert split_map.near_dup["pil_version"] == pil_version()
    assert split_map.near_dup["union_max_hamming"] == 4
    assert split_map.near_dup["never_eval_exclude_max_hamming"] == 8


def test_chain_guard_refuses_before_writing(tmp_path: Path) -> None:
    registry, roots = _two_sources_with_twin(tmp_path)
    out = tmp_path / "SPLIT_MAP.json"
    with pytest.raises(NearDupChainError, match=r"chain guard.*2 images"):
        generate_split_map(registry, out=out, roots=roots, near_dup=_config(tmp_path))
    assert not out.exists()


def test_never_eval_twin_is_dropped_from_every_split(tmp_path: Path) -> None:
    twin = _picture(100)
    real_root = _stage(tmp_path / "real", [("t", "real/twin", _encode(twin, "PNG"))])
    pseudo_rows = [
        ("t", "pseudo/twin", _encode(twin, "JPEG")),
        ("u", "pseudo/u", _encode(_picture(9), "PNG")),
    ]
    pseudo_root = _stage(tmp_path / "pseudo", pseudo_rows)
    registry = _registry(
        {
            "real-src": _source("real-src"),
            "pseudo-src": _source("pseudo-src", ("pseudo-label", "never-eval")),
        }
    )
    split_map = tmp_path / "SPLIT_MAP.json"
    save_split_map(
        split_map,
        SplitMap(
            by="group",
            seed=0,
            ratios={"train": 0.7, "val": 0.15, "test": 0.15},
            assignments={"real/twin": "val", "pseudo/twin": "train", "pseudo/u": "train"},
        ),
    )
    result = build_release(
        registry,
        release="r47",
        split_map=split_map,
        out_dir=tmp_path / "out",
        roots={"real-src": real_root, "pseudo-src": pseudo_root},
        near_dup=_config(tmp_path),
    )
    twin_sha = hashlib.sha256(pseudo_rows[0][2]).hexdigest()
    kept_sha = hashlib.sha256(pseudo_rows[1][2]).hexdigest()
    shas = {sha for sha, _ in result.tasks[0].rows}
    assert twin_sha not in shas and kept_sha in shas
    assert result.never_eval_near_dup_excluded == (twin_sha,)
    assert result.never_eval_near_dup_rows == 1
    release_json = json.loads((result.out_dir / "RELEASE.json").read_text())
    assert release_json["never_eval_near_dup_excluded"] == {"count": 1, "sha256": [twin_sha]}
    assert release_json["near_dup"]["pil_version"] == pil_version()
