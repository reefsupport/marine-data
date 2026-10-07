"""``build_release`` excludes ``never-eval``-tagged rows from any non-train split (WS-D
S15c) — the registry tag `generate_split_map`/`rows_to_counts` used to enforce nowhere
(``grep -rn never.eval src/`` was empty before this).

Reuses ``tests/test_release.py``'s real-writer fixture pattern (``StagedImage`` +
``write_metadata_table``, real image bytes, real ``Registry``/``TaskSpec``).
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

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
from marinedata.splitmap import SplitMap, save_split_map
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec

RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def _source(source_id: str, *, tags: tuple[str, ...] = ()) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="Fixture staged source for the never-eval exclusion test.",
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
    staged = [
        StagedImage(
            stem=stem,
            partition="p",
            upstream_path=f"orig/{stem}.jpg",
            upstream_split=None,
            width=4,
            height=4,
            split_group=group,
        )
        for stem, group, _ in rows
    ]
    for stem, _, data in rows:
        image_path = root / "images" / "p" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(data)
    write_metadata_table(root / "metadata.parquet", staged)
    return root


def _registry(sources: dict[str, Source]) -> Registry:
    schema = LabelSchema(id="rs-benthic-v1", name="Fixture", axes=(Axis.TAXON,), nodes=())
    tasks = {"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)}
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


def test_mixed_merged_component_in_val_excludes_never_eval_copy(tmp_path: Path) -> None:
    """A real source and a never-eval-tagged source share one byte-identical image under
    two different `split_group` values (the S15c duplicate-content shape). The frozen map
    (as `generate_split_map` would have written it after merging, D1) sends that shared
    group to `val`. The real source's row stays in `val`; the never-eval source's copy of
    the same content is dropped and counted — never moved to `train`."""
    shared_bytes = b"SHARED-DUPLICATE-CONTENT"
    real_root = _stage(
        tmp_path / "real",
        [("shared", "real/shared", shared_bytes), ("only", "real/only", b"REAL-UNIQUE")],
    )
    pseudo_root = _stage(
        tmp_path / "pseudo",
        [
            ("shared", "pseudo/shared", shared_bytes),
            ("only", "pseudo/only", b"PSEUDO-UNIQUE"),
        ],
    )
    registry = _registry(
        {
            "real-src": _source("real-src"),
            "pseudo-src": _source("pseudo-src", tags=("pseudo-label", "never-eval")),
        }
    )
    roots = {"real-src": real_root, "pseudo-src": pseudo_root}

    split_map = tmp_path / "SPLIT_MAP.json"
    save_split_map(
        split_map,
        SplitMap(
            by="group",
            seed=0,
            ratios=RATIOS,
            generated_at="2026-09-24T00:00:00Z",
            release="r15c-test",
            assignments={
                # Both members of the merged component carry the same split — what
                # `resolve_splits`'s member-expansion (D1) writes.
                "real/shared": "val",
                "pseudo/shared": "val",
                "real/only": "train",
                "pseudo/only": "train",
            },
        ),
    )

    result = build_release(
        registry, release="r15c-test", split_map=split_map, roots=roots, out_dir=tmp_path / "out"
    )

    assert len(result.tasks) == 1
    rows = dict(result.tasks[0].rows)
    shared_sha = hashlib.sha256(shared_bytes).hexdigest()
    # The task is self-supervised (see `_registry`): a shared map's `val` demotes to
    # `probe` for a pretrain task (WS-D S7m) — orthogonal to the never-eval rule under
    # test here, which is why the row survives at all (the real-src copy, not excluded).
    assert rows[shared_sha] == "probe"
    # Exactly one row for the shared sha256 — the pseudo-label copy was dropped, not
    # duplicated or moved to train.
    assert sum(1 for sha, _ in result.tasks[0].rows if sha == shared_sha) == 1
    assert result.never_eval_excluded == 1


def test_never_eval_row_in_train_is_kept(tmp_path: Path) -> None:
    """A never-eval row assigned `train` is ordinary output, not excluded — only a
    non-train landing drops it. A second never-eval group assigned `val` is included
    here purely so `Dataset.split()`'s own non-empty-split invariant has a `probe` group
    to work with; it is expected to itself be excluded (never_eval_excluded == 1)."""
    root = _stage(
        tmp_path / "pseudo",
        [("only", "pseudo/only", b"PSEUDO-UNIQUE"), ("v1", "pseudo/v1", b"PSEUDO-VAL")],
    )
    registry = _registry({"pseudo-src": _source("pseudo-src", tags=("never-eval",))})
    split_map = tmp_path / "SPLIT_MAP.json"
    save_split_map(
        split_map,
        SplitMap(
            by="group",
            seed=0,
            ratios=RATIOS,
            generated_at="2026-09-24T00:00:00Z",
            release="r15c-test",
            assignments={"pseudo/only": "train", "pseudo/v1": "val"},
        ),
    )

    result = build_release(
        registry,
        release="r15c-test",
        split_map=split_map,
        roots={"pseudo-src": root},
        out_dir=tmp_path / "out",
    )

    assert result.tasks[0].rows == ((hashlib.sha256(b"PSEUDO-UNIQUE").hexdigest(), "train"),)
    assert result.never_eval_excluded == 1
