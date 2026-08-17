"""Mirror-legality and sharding tests.

This is the one module in the project where a bug is a licence breach rather than a bad
metric, so the matrix is exhaustive: every tier against every target, plus each absolute
bar.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from conftest import make_source

from marinedata import Registry
from marinedata.builder import Dataset
from marinedata.enums import LegalBasis, Tier
from marinedata.labelindex import LabelIndex
from marinedata.lineage import build_lineage
from marinedata.mirror import (
    MirrorTarget,
    MirrorViolation,
    attribution_document,
    evaluate_mirror,
    plan_mirror,
)
from marinedata.models import Licence, LicenceFlags
from marinedata.sample import LabelValue, Sample
from marinedata.schema import Axis
from marinedata.shard import ShardError, read_shard, write_shards

pytest.importorskip("pandas")

TIERS = (
    Tier.OWN,
    Tier.PERMISSIVE,
    Tier.COPYLEFT,
    Tier.NONCOMMERCIAL,
    Tier.TDM_ONLY,
    Tier.PROHIBITED,
)

# target                       T0    T1    T2    T3    T4    TX
EXPECTED = {
    MirrorTarget.PRIVATE_CACHE: (True, True, True, True, False, False),
    MirrorTarget.TRAINING_SHARD: (True, True, True, True, False, False),
    MirrorTarget.PUBLIC_MIRROR: (True, True, True, False, False, False),
}


def _source(tier: Tier, **kwargs) -> object:
    source = make_source("flat-images")
    flags = kwargs.pop("flags", LicenceFlags())
    basis = kwargs.pop("legal_basis", None)
    if basis is None:
        basis = LegalBasis.TDM if tier is Tier.TDM_ONLY else LegalBasis.LICENCE
    return source.model_copy(
        update={
            "licence": Licence(id="TEST", name="Test", tier=tier, flags=flags),
            "legal_basis": basis,
            "notes": "fixture",
        }
    )


@pytest.mark.parametrize("target", sorted(EXPECTED, key=lambda t: t.value))
def test_tier_target_matrix(target: MirrorTarget) -> None:
    """The core table. Changing it should require changing this test deliberately."""
    for tier, should_allow in zip(TIERS, EXPECTED[target], strict=True):
        decision = evaluate_mirror(_source(tier), target)
        assert decision.allowed is should_allow, (
            f"{target.value} / {tier.value}: expected {should_allow}, "
            f"got {decision.allowed} — {decision.reason}"
        )


def test_noncommercial_may_be_cached_but_never_published() -> None:
    """The distinction the whole module exists for: caching is not distribution."""
    source = _source(Tier.NONCOMMERCIAL)
    assert evaluate_mirror(source, MirrorTarget.PRIVATE_CACHE).allowed
    published = evaluate_mirror(source, MirrorTarget.PUBLIC_MIRROR)
    assert not published.allowed
    assert "never republished" in published.reason


@pytest.mark.parametrize("target", sorted(EXPECTED, key=lambda t: t.value))
def test_no_derivatives_never_copied(target: MirrorTarget) -> None:
    decision = evaluate_mirror(
        _source(Tier.NONCOMMERCIAL, flags=LicenceFlags(no_derivatives=True)), target
    )
    assert not decision.allowed
    assert "no_derivatives" in decision.reason


@pytest.mark.parametrize("target", sorted(EXPECTED, key=lambda t: t.value))
def test_provenance_defective_never_copied(target: MirrorTarget) -> None:
    decision = evaluate_mirror(
        _source(Tier.PERMISSIVE, flags=LicenceFlags(provenance_defective=True)), target
    )
    assert not decision.allowed


@pytest.mark.parametrize("target", sorted(EXPECTED, key=lambda t: t.value))
def test_tdm_corpus_never_stored(target: MirrorTarget) -> None:
    """Art. 4(2) permits retention only while mining needs it — a lake is not that."""
    decision = evaluate_mirror(_source(Tier.PERMISSIVE, legal_basis=LegalBasis.TDM), target)
    assert not decision.allowed
    assert "retention" in decision.reason


def test_unknown_basis_never_copied() -> None:
    decision = evaluate_mirror(
        _source(Tier.PERMISSIVE, legal_basis=LegalBasis.UNKNOWN),
        MirrorTarget.PRIVATE_CACHE,
    )
    assert not decision.allowed


def test_share_alike_obligation_surfaces_on_public_mirror() -> None:
    source = _source(Tier.COPYLEFT, flags=LicenceFlags(share_alike=True))
    decision = evaluate_mirror(source, MirrorTarget.PUBLIC_MIRROR)
    assert decision.allowed
    assert any("share-alike" in o for o in decision.obligations)


def test_share_alike_propagates_to_shards_and_weights() -> None:
    source = _source(Tier.COPYLEFT, flags=LicenceFlags(share_alike=True))
    decision = evaluate_mirror(source, MirrorTarget.TRAINING_SHARD)
    assert any("weights inherit" in o for o in decision.obligations)


def test_decision_can_raise() -> None:
    with pytest.raises(MirrorViolation):
        evaluate_mirror(_source(Tier.PROHIBITED), MirrorTarget.PRIVATE_CACHE).raise_if_denied()


# ── plans and attribution ─────────────────────────────────────────────────


def test_plan_partitions_and_collects_obligations() -> None:
    sources = [
        _source(Tier.PERMISSIVE, flags=LicenceFlags(attribution_required=True)),
        _source(Tier.NONCOMMERCIAL),
        _source(Tier.PROHIBITED),
    ]
    plan = plan_mirror(sources, MirrorTarget.PUBLIC_MIRROR)
    assert len(plan.included) == 1
    assert len(plan.excluded) == 2
    assert any("attribution" in o for o in plan.obligations)
    assert "mirror plan" in plan.summary()


def test_attribution_document_lists_every_included_source() -> None:
    sources = [_source(Tier.PERMISSIVE, flags=LicenceFlags(attribution_required=True))]
    plan = plan_mirror(sources, MirrorTarget.PUBLIC_MIRROR)
    text = attribution_document(plan, sources)
    assert "# Attribution" in text
    assert "not a relicensing" in text
    assert sources[0].name in text


def test_real_registry_public_mirror_excludes_nc_and_prohibited(registry: Registry) -> None:
    plan = plan_mirror(list(registry), MirrorTarget.PUBLIC_MIRROR)
    excluded = {d.source_id for d in plan.excluded}
    assert "marineinst20m" in excluded
    assert "seatizen-atlas" in excluded, "ND must never be publicly mirrored"
    assert "coralscop-masks-rs" in excluded, "NC must never be publicly mirrored"
    assert plan.included, "some permissive sources should be mirrorable"


# ── sharding ──────────────────────────────────────────────────────────────


@pytest.fixture
def shardable(registry: Registry, tmp_path: Path) -> Dataset:
    """A dataset whose images exist on disk, from an own-tier source."""
    images = tmp_path / "img"
    images.mkdir()
    samples = []
    for i in range(6):
        path = images / f"f{i}.jpg"
        path.write_bytes(b"\xff\xd8\xff" + bytes([i]) * 512)
        samples.append(
            Sample(
                source_id="reef-support-benthic",
                key=f"f{i}.jpg",
                image=path,
                labels={Axis.TAXON: LabelValue("HC")},
                supervised=frozenset({Axis.TAXON}),
                meta={"partition": f"site{i % 2}"},
            )
        )
    schema = registry.label_schema("rs-benthic-v1")
    return Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )


def test_write_and_read_shards_round_trip(shardable: Dataset, tmp_path: Path) -> None:
    result = write_shards(shardable, tmp_path / "shards")
    assert result.samples == 6
    assert result.shards

    manifest = json.loads((result.output_dir / "SHARD_MANIFEST.json").read_text())
    assert manifest["samples"] == 6
    assert manifest["format"] == "webdataset"
    assert manifest["ignore_index"] == -100
    assert (result.output_dir / "ATTRIBUTION.md").is_file()

    entries = list(read_shard(result.output_dir / result.shards[0]))
    assert len(entries) == 6
    first = entries[0]
    assert "image" in first and "metadata" in first
    assert first["metadata"]["labels"]["taxon"] == "HC"
    assert first["metadata"]["supervised"] == ["taxon"]


def test_shard_keys_are_self_describing(shardable: Dataset, tmp_path: Path) -> None:
    """A shard found detached from its manifest should still say where it came from."""
    result = write_shards(shardable, tmp_path / "shards")
    with tarfile.open(result.output_dir / result.shards[0]) as tar:
        names = tar.getnames()
    assert any(n.startswith("reef-support-benthic__site0__") for n in names)


def test_small_shard_target_produces_multiple_shards(shardable: Dataset, tmp_path: Path) -> None:
    result = write_shards(shardable, tmp_path / "shards", shard_bytes=600)
    assert len(result.shards) > 1
    assert result.samples == 6


def test_shards_are_deterministic(shardable: Dataset, tmp_path: Path) -> None:
    """mtime is zeroed so identical content yields byte-identical shards."""
    a = write_shards(shardable, tmp_path / "a")
    b = write_shards(shardable, tmp_path / "b")
    assert (a.output_dir / a.shards[0]).read_bytes() == (b.output_dir / b.shards[0]).read_bytes()


def test_missing_images_are_skipped_not_fatal(shardable: Dataset, tmp_path: Path) -> None:
    Path(shardable.samples[0].image).unlink()
    result = write_shards(shardable, tmp_path / "shards")
    assert result.skipped == 1
    assert result.samples == 5


def test_strict_mode_raises_on_missing_image(shardable: Dataset, tmp_path: Path) -> None:
    Path(shardable.samples[0].image).unlink()
    with pytest.raises(ShardError, match="image missing"):
        write_shards(shardable, tmp_path / "shards", strict=True)


def test_sharding_a_prohibited_source_raises(registry: Registry, tmp_path: Path) -> None:
    """⭐ Sharding is a derivative act, so the mirror gate applies to it."""
    image = tmp_path / "x.jpg"
    image.write_bytes(b"\xff\xd8\xff")
    samples = [
        Sample(
            source_id="marineinst20m",
            key="x.jpg",
            image=image,
            labels={},
            supervised=frozenset(),
        )
    ]
    schema = registry.label_schema("rs-benthic-v1")
    ds = Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )
    with pytest.raises(ShardError, match="may not be sharded"):
        write_shards(ds, tmp_path / "shards")


def test_no_images_on_disk_gives_actionable_error(registry: Registry, tmp_path: Path) -> None:
    samples = [
        Sample(
            source_id="reef-support-benthic",
            key="k",
            image=Path("/nope.jpg"),
            labels={},
            supervised=frozenset(),
        )
    ]
    schema = registry.label_schema("rs-benthic-v1")
    ds = Dataset(
        samples=samples,
        label_index=LabelIndex.from_samples(samples, schema),
        lineage=build_lineage([], registry.profile("research")),
    )
    with pytest.raises(ShardError, match="not present on disk"):
        write_shards(ds, tmp_path / "shards")
