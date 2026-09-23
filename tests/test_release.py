"""``marinedata release build`` — the DoD leakage test (WS-D S7k).

Fixtures are staged with the repo's own writer (``marinedata.tables``), the same
pattern ``tests/test_loader_staged_tree.py`` uses, not hand-rolled parquet or samples.
Two sources share one byte-identical image under one ``split_group`` — a real
cross-source duplicate, image identity = sha256 — and three registry tasks (standing in
for a segmentation set, a points set and a pretraining set; see the module docstring
below for why all three are declared ``self_supervised`` here) all build from them
against one pre-written, frozen ``SPLIT_MAP.json``. The guarantee under test: no sha256
ever appears in two different splits, in any task's manifest.

All three tasks are ``self_supervised`` rather than one of each real kind because
``DatasetBuilder`` only needs a projector (and therefore a real crosswalk/vocabulary) for
a *supervised* task — see :meth:`marinedata.task.TaskSpec` — and these fixture sources
carry no labels at all (``annotations=()``, so ``_check_mappable`` never flags them
either). The property this file proves — one frozen group-keyed map, read by several
independently-built ``Dataset``s, never disagreeing on a shared image's split — does not
depend on task kind or label content, only on ``split_group``/sha256 identity.
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
from marinedata.models import Access, Coverage, Licence, LoaderSpec, Profile, Source, Verification
from marinedata.registry import Registry
from marinedata.release import build_release
from marinedata.schema import Axis, LabelSchema
from marinedata.splitmap import SplitMap, SplitMapError, save_split_map
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _source(source_id: str, version: str = "v1") -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the release-build DoD test.",
        version=version,
        licence=Licence(id="CC-BY-4.0", name="CC BY 4.0", tier=Tier.PERMISSIVE),
        verification=Verification(
            verified_on=date(2026, 8, 17), verified_by="synthetic fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(method=AccessMethod.HTTP, uri="https://example.invalid"),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        loader=LoaderSpec(layout="staged-tree", params={}),
        annotations=(),
    )


def _stage(root: Path, rows: list[tuple[str, str, bytes]]) -> Path:
    """``rows``: ``(stem, split_group, image_bytes)``. Writes real image bytes and a
    real ``metadata.parquet`` via the repo's own writer — see the module docstring."""
    staged = []
    for stem, group, data in rows:
        image_path = root / "images" / "p" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data)
        staged.append(
            StagedImage(
                stem=stem,
                partition="p",
                upstream_path=f"orig/{stem}.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group=group,
            )
        )
    write_metadata_table(root / "metadata.parquet", staged)
    return root


def _registry(sources: dict[str, Source]) -> Registry:
    # Named to match `release.DEFAULT_SCHEMA_ID` — every task below is self-supervised
    # (`schema_id=None`), so `build_release` falls back to this id for `LabelIndex`'s
    # sake even though nothing here has a vocabulary to project onto.
    schema = LabelSchema(id="rs-benthic-v1", name="Fixture", axes=(Axis.TAXON,), nodes=())
    tasks = {
        task_id: TaskSpec(id=task_id, kind=TaskKind.SELF_SUPERVISED)
        for task_id in ("segmentation-set", "points-set", "pretrain-set")
    }
    profile = Profile(
        id="research",
        description="fixture",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )
    return Registry(
        sources=sources,
        licences={},
        profiles={"research": profile},
        schemas={"rs-benthic-v1": schema},
        tasks=tasks,
    )


def _two_source_fixture(tmp_path: Path) -> tuple[Registry, dict[str, Path]]:
    """Two staged sources sharing one byte-identical, same-``split_group`` image — the
    cross-source duplicate — plus one unique image each."""
    dup_bytes = b"DUPLICATE-IMAGE-BYTES"
    root_a = _stage(
        tmp_path / "src-a",
        [("dup", "shared/dup", dup_bytes), ("only_a", "src-a/only", b"AAAA-UNIQUE")],
    )
    root_b = _stage(
        tmp_path / "src-b",
        [("dup", "shared/dup", dup_bytes), ("only_b", "src-b/only", b"BBBB-UNIQUE")],
    )
    registry = _registry({"src-a": _source("src-a"), "src-b": _source("src-b")})
    return registry, {"src-a": root_a, "src-b": root_b}


def _write_map(path: Path, assignments: dict[str, str]) -> None:
    save_split_map(
        path,
        SplitMap(
            by="group",
            seed=0,
            ratios=RATIOS,
            assignments=assignments,
            generated_at="2026-09-23T00:00:00Z",
            release="r7k",
        ),
    )


def test_release_no_image_in_two_splits_across_tasks(tmp_path: Path) -> None:
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(
        split_map,
        {"shared/dup": "test", "src-a/only": "train", "src-b/only": "val"},
    )

    result = build_release(
        registry,
        release="r7k",
        split_map=split_map,
        roots=roots,
        out_dir=tmp_path / "out",
    )

    assert {t.task_id for t in result.tasks} == {"segmentation-set", "points-set", "pretrain-set"}

    split_of: dict[str, str] = {}
    for task in result.tasks:
        for sha256, split_name in task.rows:
            prior = split_of.get(sha256)
            assert prior is None or prior == split_name, (
                f"{sha256} landed in both {prior!r} and {split_name!r} across tasks"
            )
            split_of[sha256] = split_name

    # The shared duplicate really was hashed identically and really did appear in
    # every task's manifest — otherwise the assertion above would pass vacuously.
    dup_sha256 = {sha for sha, split_name in result.tasks[0].rows if split_name == "test"}
    assert dup_sha256, "the duplicate's assigned split must be reachable"
    for task in result.tasks:
        assert dup_sha256 <= {sha for sha, _ in task.rows}


def test_cross_source_duplicate_lands_in_one_split(tmp_path: Path) -> None:
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(
        split_map,
        {"shared/dup": "val", "src-a/only": "train", "src-b/only": "test"},
    )

    result = build_release(
        registry, release="r7k", split_map=split_map, roots=roots, out_dir=tmp_path / "out"
    )

    task = next(t for t in result.tasks if t.task_id == "segmentation-set")
    rows_by_sha = {}
    for sha256, split_name in task.rows:
        rows_by_sha.setdefault(sha256, set()).add(split_name)
    for sha256, splits in rows_by_sha.items():
        assert len(splits) == 1, f"{sha256} split across {splits}"
    # Two samples (one per source) really did share the duplicate's sha256.
    assert sum(1 for sha256, _ in task.rows if rows_by_sha[sha256] == {"val"}) == 2


def test_frozen_map_raises_on_new_group(tmp_path: Path) -> None:
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    # "shared/dup" is missing — a group an admitted source contributes but the frozen
    # map never recorded.
    _write_map(split_map, {"src-a/only": "train", "src-b/only": "val"})
    before = split_map.read_text()

    with pytest.raises(SplitMapError, match="frozen"):
        build_release(
            registry, release="r7k", split_map=split_map, roots=roots, out_dir=tmp_path / "out"
        )

    assert split_map.read_text() == before, "a frozen release build must never write to the map"
