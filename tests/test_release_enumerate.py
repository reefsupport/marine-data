"""``enumerate_release_rows`` / ``generate_split_map`` — the metadata-only guard (WS-D
S12c).

``enumerate_release_rows`` used to assume every admitted source's ``metadata.parquet``
was the ``staged-tree`` schema (``partition``/``stem`` columns). A ``loader.layout:
metadata-only`` source (e.g. the pinned Roboflow v3i classification set: bespoke
``path``/``class``/``upstream_split``/``width``/``height``/``split_group`` columns, no
``partition``) crashed with ``KeyError: 'partition'`` instead. Whether classification
sources like these join a release at all is still Yohan's open question ("Roboflow
labels: parquet vs metadata") — until it's answered, a non-``staged-tree`` source is
skipped and reported, never silently dropped and never allowed to crash the enumerator.

Fixtures reuse the same real-writer pattern as ``test_release.py`` (``StagedImage`` +
``write_metadata_table``), kept in this file rather than imported from there since
pytest test modules aren't meant to import each other's private helpers.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from marinedata.enums import (
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import (
    Access,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    SplitGroupRule,
    Verification,
)
from marinedata.registry import Registry
from marinedata.release import enumerate_release_rows, generate_split_map
from marinedata.splitmap import load_split_map
from marinedata.tables import StagedImage, write_metadata_table


def _source(
    source_id: str,
    layout: str,
    *,
    tags: tuple[str, ...] = (),
    split_group: SplitGroupRule | None = None,
) -> Source:
    kwargs = {} if split_group is None else {"split_group": split_group}
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture source for the release-enumerator layout guard.",
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
        loader=LoaderSpec(layout=layout, params={}),
        annotations=(),
        tags=tags,
        **kwargs,
    )


def _registry(sources: dict[str, Source]) -> Registry:
    profile = Profile(
        id="research",
        description="fixture",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )
    return Registry(
        sources=sources, licences={}, profiles={"research": profile}, schemas={}, tasks={}
    )


def _stage_staged_tree(
    root: Path, stem: str, group: str, data: bytes, *, partition: str = "p"
) -> Path:
    """A real ``staged-tree`` root: one image under ``images/<partition>/`` plus a
    matching ``metadata.parquet`` row, written with the repo's own writer."""
    image_path = root / "images" / partition / f"{stem}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(data)
    write_metadata_table(
        root / "metadata.parquet",
        [
            StagedImage(
                stem=stem,
                partition=partition,
                upstream_path=f"orig/{stem}.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group=group,
            )
        ],
    )
    return root


def _stage_metadata_only(root: Path, stem: str, group: str) -> Path:
    """A ``metadata-only`` root: a bespoke ``metadata.parquet`` schema with no
    ``partition``/``stem``-indexed image tree at all — v3i's real shape."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "path": [f"{stem}.jpg"],
            "class": ["healthy"],
            "upstream_split": ["train"],
            "width": [4],
            "height": [4],
            "split_group": [group],
        }
    )
    pq.write_table(table, root / "metadata.parquet")
    return root


def test_metadata_only_source_skipped_and_reported(tmp_path: Path) -> None:
    """A ``metadata-only`` root never crashes the enumerator, contributes no rows, and
    is reported in ``skipped`` — never silently dropped."""
    staged_root = _stage_staged_tree(tmp_path / "staged", "a", "g/a", b"AAAA")
    meta_root = _stage_metadata_only(tmp_path / "meta", "v3i-a", "g/v3i")
    registry = _registry(
        {"staged-src": _source("staged-src", "staged-tree"), "v3i": _source("v3i", "metadata-only")}
    )
    roots = {"staged-src": staged_root, "v3i": meta_root}

    skipped: dict[str, str] = {}
    rows = list(enumerate_release_rows(registry, roots, "research", skipped=skipped))

    assert {source_id for _, _, source_id in rows} == {"staged-src"}
    assert "v3i" in skipped
    assert "not in release" in skipped["v3i"]


def test_needs_attribution_source_skipped_and_reported(tmp_path: Path) -> None:
    """A ``staged-tree`` source tagged ``needs-attribution`` (WS-D S23) never crashes
    or contributes rows — it is skipped with a reason naming the tag, the same
    never-silently-dropped contract as the ``metadata-only`` layout guard above. Tagged
    before the layout check even matters here: this source IS staged-tree, proving the
    attribution gate is independent of layout."""
    staged_root = _stage_staged_tree(tmp_path / "staged", "a", "g/a", b"AAAA")
    gated_root = _stage_staged_tree(tmp_path / "gated", "b", "g/b", b"BBBB")
    registry = _registry(
        {
            "staged-src": _source("staged-src", "staged-tree"),
            "gated-src": _source("gated-src", "staged-tree", tags=("needs-attribution",)),
        }
    )
    roots = {"staged-src": staged_root, "gated-src": gated_root}

    skipped: dict[str, str] = {}
    rows = list(enumerate_release_rows(registry, roots, "research", skipped=skipped))

    assert {source_id for _, _, source_id in rows} == {"staged-src"}
    assert "needs-attribution" in skipped["gated-src"]


def test_missing_split_group_resolved_by_registry_rule_at_build_time(tmp_path: Path) -> None:
    """D-V2: a ``metadata.parquet`` cached before the ``split_group`` column was
    backfilled at ingest (e.g. coralscapes, mermaid-aws before WP-2) has ``None`` for
    some rows. The release enumerator must not raise — it applies the registry's own
    per-source :class:`~marinedata.models.SplitGroupRule` at build time instead, the
    same rule ``ingest.py`` would have applied had staging run after the rule existed.
    Never rewrites the cached ``metadata.parquet`` — the row stays ``None`` on disk."""
    rule = SplitGroupRule(pattern=r"(?i)(site[0-9]+)", match_field="stem", template="stale/{group}")
    root = _stage_staged_tree(tmp_path / "staged", "site7-a", None, b"AAAA")
    registry = _registry({"stale-src": _source("stale-src", "staged-tree", split_group=rule)})

    rows = list(enumerate_release_rows(registry, {"stale-src": root}, "research"))

    assert rows == [(rows[0][0], "stale/site7", "stale-src")]
    # the cache itself is never touched (D-V2: a changed mtime must not happen)
    import pyarrow.parquet as pq

    on_disk = pq.read_table(root / "metadata.parquet").to_pylist()
    assert on_disk[0]["split_group"] is None


def test_missing_image_still_raises(tmp_path: Path) -> None:
    """A ``metadata.parquet`` row whose image is absent from the partition dir still
    raises — the stem→path index (WS-D S45 perf) must not turn a real missing file into
    a silent skip."""
    root = tmp_path / "staged"
    write_metadata_table(
        root / "metadata.parquet",
        [
            StagedImage(
                stem="ghost",
                partition="p",
                upstream_path="orig/ghost.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group="g/ghost",
            )
        ],
    )
    registry = _registry({"staged-src": _source("staged-src", "staged-tree")})

    with pytest.raises(ValueError, match="not found under"):
        list(enumerate_release_rows(registry, {"staged-src": root}, "research"))


def test_two_candidate_extensions_picks_the_sorted_first(tmp_path: Path) -> None:
    """A stem with two files under the same partition (e.g. ``a0.jpg`` and ``a0.png``)
    keeps the pre-index tie-break: ``sorted(...)[0]`` — alphabetically first extension —
    same as the old ``glob(f"{stem}.*")`` call this index replaces."""
    root = tmp_path / "staged"
    images_dir = root / "images" / "p"
    images_dir.mkdir(parents=True)
    (images_dir / "a0.png").write_bytes(b"PNG-BYTES")
    (images_dir / "a0.jpg").write_bytes(b"JPG-BYTES")
    write_metadata_table(
        root / "metadata.parquet",
        [
            StagedImage(
                stem="a0",
                partition="p",
                upstream_path="orig/a0.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group="g/a0",
            )
        ],
    )
    registry = _registry({"staged-src": _source("staged-src", "staged-tree")})

    from marinedata.checksums import file_digest

    rows = list(enumerate_release_rows(registry, {"staged-src": root}, "research"))
    assert len(rows) == 1
    digest, group, source_id = rows[0]
    assert digest == file_digest(images_dir / "a0.jpg")
    assert (group, source_id) == ("g/a0", "staged-src")


def test_staged_tree_missing_partition_still_raises(tmp_path: Path) -> None:
    """A ``staged-tree`` source is held to the real schema: a row with no ``partition``
    key is corruption, not a layout to skip, and must still raise ``KeyError``."""
    root = tmp_path / "broken"
    root.mkdir()
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({"stem": ["a"], "split_group": ["g/a"]}), root / "metadata.parquet")
    registry = _registry({"broken-src": _source("broken-src", "staged-tree")})

    with pytest.raises(KeyError):
        list(enumerate_release_rows(registry, {"broken-src": root}, "research"))


def test_generate_split_map_excludes_metadata_only_rows(tmp_path: Path) -> None:
    """A split map generated with a ``metadata-only`` source present in ``roots``
    contains no group from that source — and the caller is told it was skipped."""
    staged_root = _stage_staged_tree(tmp_path / "staged", "a", "shared/a", b"AAAA")
    meta_root = _stage_metadata_only(tmp_path / "meta", "v3i-a", "shared/v3i")
    registry = _registry(
        {"staged-src": _source("staged-src", "staged-tree"), "v3i": _source("v3i", "metadata-only")}
    )
    roots = {"staged-src": staged_root, "v3i": meta_root}

    out = tmp_path / "SPLIT_MAP.json"
    skipped: dict[str, str] = {}
    generate_split_map(
        registry,
        out=out,
        roots=roots,
        profile="research",
        ratios={"train": 1.0, "val": 0.0, "test": 0.0},
        seed=0,
        min_groups=1,
        now="2026-09-24T00:00:00Z",
        release="r12c-test",
        skipped=skipped,
    )

    split_map = load_split_map(out)
    assert split_map is not None
    assert "shared/v3i" not in split_map.assignments
    assert "shared/a" in split_map.assignments
    assert skipped == {"v3i": "metadata-only: not in release until labels format decided"}
