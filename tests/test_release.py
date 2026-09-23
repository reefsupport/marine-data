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

import hashlib
from datetime import date
from pathlib import Path

import pytest

from marinedata.enums import (
    AccessMethod,
    AnnotationKind,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.models import (
    Access,
    Annotation,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    Verification,
)
from marinedata.registry import Registry
from marinedata.release import build_release
from marinedata.schema import Axis, Crosswalk, CrosswalkEdge, LabelNode, LabelSchema
from marinedata.splitmap import SplitMap, SplitMapError, save_split_map
from marinedata.tables import PointRow, StagedImage, write_metadata_table, write_points_table
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
    """No sha256 ever appears in two different splits across tasks — and, since every
    task here is self-supervised, the shared map's `test` split must never surface at
    all (WS-D S7m): a map `val` group demotes to `probe`, a map `test` group is excluded
    outright rather than appended to any pretrain manifest."""
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(
        split_map,
        {"shared/dup": "val", "src-a/only": "train", "src-b/only": "test"},
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
    all_split_names: set[str] = set()
    for task in result.tasks:
        for sha256, split_name in task.rows:
            all_split_names.add(split_name)
            prior = split_of.get(sha256)
            assert prior is None or prior == split_name, (
                f"{sha256} landed in both {prior!r} and {split_name!r} across tasks"
            )
            split_of[sha256] = split_name

    assert "test" not in all_split_names, "no self-supervised task may ever emit a `test` row"

    # The `val`-mapped duplicate really was hashed identically and really did appear in
    # every task's manifest, renamed `probe` — otherwise the assertion above would pass
    # vacuously.
    dup_sha256 = {sha for sha, split_name in result.tasks[0].rows if split_name == "probe"}
    assert dup_sha256, "the duplicate's assigned split must be reachable as probe"
    for task in result.tasks:
        assert dup_sha256 <= {sha for sha, _ in task.rows}

    # src-b/only was mapped `test` — excluded from every self-supervised task, not just
    # renamed or dropped from one.
    only_b_sha256 = hashlib.sha256(b"BBBB-UNIQUE").hexdigest()
    for task in result.tasks:
        assert only_b_sha256 not in {sha for sha, _ in task.rows}


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
    # Two samples (one per source) really did share the duplicate's sha256, and it
    # surfaces as `probe` — this task is self-supervised, so the map's `val` demotes.
    assert sum(1 for sha256, _ in task.rows if rows_by_sha[sha256] == {"probe"}) == 2
    # src-b/only was mapped `test` and must not appear in this self-supervised task at all.
    only_b_sha256 = hashlib.sha256(b"BBBB-UNIQUE").hexdigest()
    assert only_b_sha256 not in rows_by_sha


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


# ── Part B DoD: a real SUPERVISED task alongside a pretrain task (WS-D S7m) ─────────


def _labelled_registry(sources: dict[str, Source]) -> Registry:
    """The smallest crosswalk/vocabulary the repo accepts: one canonical node, one edge —
    just enough for ``DatasetBuilder``'s "no crosswalk" guard and a supervised
    ``TaskSpec`` to both pass."""
    schema = LabelSchema(
        id="rs-benthic-v1",
        name="Fixture",
        axes=(Axis.TAXON,),
        nodes=(LabelNode(id="HC", name="Hard Coral", axis=Axis.TAXON),),
        canonical=True,
    )
    crosswalk = Crosswalk(
        id="fixture-crosswalk",
        source_schema="fixture-native",
        target_schema="rs-benthic-v1",
        edges=(CrosswalkEdge(source_label="hard_coral", targets={Axis.TAXON: "HC"}),),
    )
    tasks = {
        "bleaching-task": TaskSpec(
            id="bleaching-task",
            kind=TaskKind.SUPERVISED,
            schema_id="rs-benthic-v1",
            axis=Axis.TAXON,
            classes=("HC",),
        ),
        "pretrain-task": TaskSpec(id="pretrain-task", kind=TaskKind.SELF_SUPERVISED),
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
        crosswalks={"fixture-crosswalk": crosswalk},
    )


def _labelled_source(source_id: str) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture labelled staged source for the Part B DoD test.",
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
        loader=LoaderSpec(
            layout="staged-tree",
            params={},
            schema_id="fixture-native",
            crosswalk_id="fixture-crosswalk",
        ),
        annotations=(Annotation(kind=AnnotationKind.POINT_LABEL, supervises=("taxon",)),),
    )


def _stage_labelled(root: Path, rows: list[tuple[str, str, bytes]]) -> Path:
    """Like ``_stage``, plus one ``hard_coral`` point label per image so the source
    actually supervises ``taxon`` — a supervised task's projector needs real supervision,
    not just a crosswalk that is never exercised."""
    staged = []
    points = []
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
        points.append(
            PointRow(
                stem=stem,
                partition="p",
                row=1,
                col=1,
                label="hard_coral",
                schema_id="fixture-native",
            )
        )
    write_metadata_table(root / "metadata.parquet", staged)
    write_points_table(root / "labels" / "points.parquet", points)
    return root


def test_supervised_and_pretrain_share_no_split_and_pretrain_excludes_test(tmp_path: Path) -> None:
    """Part B DoD (WS-D S7m). One SUPERVISED task and one pretrain (self-supervised) task
    build from the same labelled staged trees over overlapping images, including a
    cross-source duplicate. No sha256 ever lands in two different splits across the two
    tasks — ``probe`` counts as the same split as ``val`` — and no image the supervised
    task calls ``test`` ever reaches the pretrain manifest."""
    dup_bytes = b"LABELLED-DUPLICATE-BYTES"
    root_a = _stage_labelled(
        tmp_path / "lab-a",
        [("dup", "shared/dup", dup_bytes), ("only_a", "lab-a/only", b"AAAA-LABELLED")],
    )
    root_b = _stage_labelled(
        tmp_path / "lab-b",
        [("dup", "shared/dup", dup_bytes), ("only_b", "lab-b/only", b"BBBB-LABELLED")],
    )
    registry = _labelled_registry(
        {"lab-a": _labelled_source("lab-a"), "lab-b": _labelled_source("lab-b")}
    )
    roots = {"lab-a": root_a, "lab-b": root_b}

    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(split_map, {"shared/dup": "val", "lab-a/only": "train", "lab-b/only": "test"})

    result = build_release(
        registry, release="r7m", split_map=split_map, roots=roots, out_dir=tmp_path / "out"
    )

    assert {t.task_id for t in result.tasks} == {"bleaching-task", "pretrain-task"}
    supervised = next(t for t in result.tasks if t.task_id == "bleaching-task")
    pretrain = next(t for t in result.tasks if t.task_id == "pretrain-task")

    # The supervised task keeps the map's own train/val/test vocabulary...
    assert {name for _, name in supervised.rows} <= {"train", "val", "test"}
    # ...and the pretrain task never sees `test`, only `train`/`probe`.
    assert {name for _, name in pretrain.rows} <= {"train", "probe"}

    # No sha256 in two different splits across tasks — `probe` is `val` under another name.
    def _canonical(name: str) -> str:
        return "val" if name == "probe" else name

    split_of: dict[str, str] = {}
    for task in (supervised, pretrain):
        for sha256, name in task.rows:
            canon = _canonical(name)
            prior = split_of.get(sha256)
            assert prior is None or prior == canon, f"{sha256}: {prior!r} vs {canon!r}"
            split_of[sha256] = canon

    # The duplicate really is reachable under both tasks, and really is val/probe.
    dup_sha256 = hashlib.sha256(dup_bytes).hexdigest()
    assert split_of[dup_sha256] == "val"
    assert dup_sha256 in {sha for sha, _ in supervised.rows}
    assert dup_sha256 in {sha for sha, _ in pretrain.rows}

    # lab-b/only was mapped `test` by the shared vocabulary — it must appear as `test`
    # in the supervised manifest, and never appear in the pretrain manifest at all.
    only_b_sha256 = hashlib.sha256(b"BBBB-LABELLED").hexdigest()
    assert (only_b_sha256, "test") in supervised.rows
    assert only_b_sha256 not in {sha for sha, _ in pretrain.rows}
