"""WP-R2c: ``generate_split_map`` honours an upstream TEST set (default on, per-source switch).

Real writers throughout (``StagedImage`` + ``write_metadata_table``, real image bytes). The fixtures
make roughly half of all groups upstream-test, far above the 15% test quota: a stratified lottery
could not put them all in ``test``, so "all in test" can only come from the forced rule.
"""

from __future__ import annotations

import io
import random
from datetime import date
from pathlib import Path

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
from marinedata.neardup import NearDupConfig
from marinedata.registry import Registry
from marinedata.release import generate_split_map
from marinedata.schema import Axis, LabelSchema
from marinedata.splitmap import load_split_map
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec

NOW = "2026-10-05T00:00:00Z"
Spec = tuple[str, str, str | None, bytes]  # stem, split_group, upstream_split, image bytes


def _source(source_id: str, tags: tuple[str, ...] = ()) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the upstream-test split rule.",
        version="v1",
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 10, 5), verified_by="synthetic fixture", method="licence-file"
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


def _registry(sources: dict[str, Source]) -> Registry:
    return Registry(
        sources=sources,
        licences={},
        profiles={
            "research": Profile(id="research", description="f", allow_tiers=(Tier.PERMISSIVE,))
        },
        schemas={
            "rs-benthic-v1": LabelSchema(id="rs-benthic-v1", name="F", axes=(Axis.TAXON,), nodes=())
        },
        tasks={"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)},
    )


def _stage(root: Path, rows: list[Spec]) -> Path:
    for stem, _, _, data in rows:
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
                upstream_split=upstream,
                width=72,
                height=64,
                split_group=group,
            )
            for stem, group, upstream, _ in rows
        ],
    )
    return root


def _fixture_rows(prefix: str, *, test_groups: int, other_groups: int) -> list[Spec]:
    """``test_groups`` groups with TEST rows (one also holds a train row), then plain train ones."""
    rows: list[Spec] = []
    for g in range(test_groups):
        rows.append((f"t{g}a", f"{prefix}/t{g}", "TEST", f"{prefix}-t{g}-a".encode()))
        # group 0 mixes upstream train + test: the whole group must still go to test
        second = "train" if g == 0 else "TEST"
        rows.append((f"t{g}b", f"{prefix}/t{g}", second, f"{prefix}-t{g}-b".encode()))
    for g in range(other_groups):
        for k in "ab":
            rows.append((f"o{g}{k}", f"{prefix}/o{g}", "train", f"{prefix}-o{g}-{k}".encode()))
    return rows


def _map(tmp_path: Path, sources: dict[str, Source], roots: dict[str, Path], **kw):  # type: ignore[no-untyped-def]
    tmp_path.mkdir(parents=True, exist_ok=True)
    out = tmp_path / "SPLIT_MAP.json"
    stats = generate_split_map(
        _registry(sources), out=out, roots=roots, now=NOW, seed=3, min_groups=3, **kw
    )
    loaded = load_split_map(out)
    assert loaded is not None
    return stats, loaded.assignments


def test_upstream_test_rows_all_land_in_test(tmp_path: Path) -> None:
    root = _stage(tmp_path / "a", _fixture_rows("a", test_groups=12, other_groups=12))
    stats, assigned = _map(tmp_path, {"src-a": _source("src-a")}, {"src-a": root})
    assert {assigned[f"a/t{g}"] for g in range(12)} == {"test"}
    assert stats.upstream_test_groups == 12 and stats.upstream_test_components == 12
    # the balancer still works the remainder: the plain train groups are not all forced to test
    assert {assigned[f"a/o{g}"] for g in range(12)} != {"test"}


def test_group_mixing_upstream_train_and_test_goes_whole_to_test(tmp_path: Path) -> None:
    root = _stage(tmp_path / "a", _fixture_rows("a", test_groups=12, other_groups=12))
    _, assigned = _map(tmp_path, {"src-a": _source("src-a")}, {"src-a": root})
    assert assigned["a/t0"] == "test"  # one row says train, one says TEST


def test_switch_off_per_source_and_globally(tmp_path: Path) -> None:
    root = _stage(tmp_path / "a", _fixture_rows("a", test_groups=12, other_groups=12))
    forced = _map(tmp_path / "on", {"src-a": _source("src-a")}, {"src-a": root})[1]
    for name, kw in (
        ("off-src", {"upstream_test_off": ["src-a"]}),
        ("off-all", {"upstream_test_off": ["*"]}),
        ("off", {"honour_upstream_test": False}),
    ):
        stats, assigned = _map(tmp_path / name, {"src-a": _source("src-a")}, {"src-a": root}, **kw)
        assert stats.upstream_test_groups == 0
        assert {assigned[f"a/t{g}"] for g in range(12)} != {"test"}, name
    assert {forced[f"a/t{g}"] for g in range(12)} == {"test"}


def test_near_dup_of_an_upstream_test_image_in_another_group_goes_to_test(tmp_path: Path) -> None:
    def picture(seed: int) -> Image.Image:
        rng = random.Random(seed)
        small = Image.new("L", (9, 8))
        small.putdata([rng.randrange(256) for _ in range(72)])
        return small.resize((72, 64), Image.Resampling.BICUBIC).convert("RGB")

    def encode(img: Image.Image, fmt: str) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=fmt, **({"quality": 85} if fmt == "JPEG" else {}))
        return buf.getvalue()

    twin = picture(100)
    rows_a: list[Spec] = [("tw", "a/twin", "TEST", encode(twin, "PNG"))]
    rows_a += [(f"u{i}", f"a/g{i}", "train", encode(picture(i), "PNG")) for i in range(14)]
    # same photo re-encoded under ANOTHER source/group, upstream says train: a leak if not pulled
    rows_b: list[Spec] = [("tw", "b/twin", "train", encode(twin, "JPEG"))]
    rows_b += [(f"u{i}", f"b/g{i}", "train", encode(picture(50 + i), "PNG")) for i in range(14)]
    roots = {"src-a": _stage(tmp_path / "a", rows_a), "src-b": _stage(tmp_path / "b", rows_b)}
    stats, assigned = _map(
        tmp_path,
        {"src-a": _source("src-a"), "src-b": _source("src-b")},
        roots,
        near_dup=NearDupConfig(cache_dir=tmp_path / "dhash", chain_fraction=0.5),
    )
    assert stats.near_dup_unions == 1 and stats.upstream_test_groups == 1
    assert assigned["a/twin"] == "test" and assigned["b/twin"] == "test"


def test_never_eval_only_component_stays_train(tmp_path: Path) -> None:
    root = _stage(tmp_path / "a", _fixture_rows("a", test_groups=12, other_groups=12))
    stats, assigned = _map(
        tmp_path, {"src-a": _source("src-a", tags=("never-eval",))}, {"src-a": root}
    )
    assert {assigned[f"a/t{g}"] for g in range(12)} == {"train"}
    assert stats.upstream_test_groups == 12 and stats.upstream_test_components == 0
