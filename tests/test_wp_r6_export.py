"""WP-R6 items 3-5: fail closed without a split_group, per-source task exclusion, split column."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_global_split_map_flavours import NOW, _registry, _rows, _src
from test_upstream_test_split import _stage

from marinedata.builder import DatasetBuilder
from marinedata.enums import AnnotationKind
from marinedata.hf_export import HFExportError, collect_rows
from marinedata.metadata_release import ImageRef, build_rows
from marinedata.models import Annotation
from marinedata.neardup import NearDupConfig
from marinedata.release import SplitGroupError, build_release, generate_split_map
from marinedata.splitmap import load_split_map

pytest.importorskip("pyarrow")


def _build(tmp_path: Path, sources, monkeypatch=None, **kw):  # type: ignore[no-untyped-def]
    registry = _registry(sources)
    roots = {s.id: _stage(tmp_path / s.id, _rows(s.id, f"shared-{s.id}".encode())) for s in sources}
    split_map = tmp_path / "SPLIT_MAP.json"
    generate_split_map(
        registry, out=split_map, roots=roots, profile="ship-open", now=NOW, seed=0, min_groups=3
    )
    result = build_release(
        registry,
        release="r1",
        split_map=split_map,
        roots=roots,
        out_dir=tmp_path / "rel",
        profile="ship-open",
        flavour="open",
        near_dup=NearDupConfig(cache_dir=tmp_path / "dh", workers=1),  # writes the near-dup record
        **kw,
    )
    return registry, roots, result


class _NoGroup(DatasetBuilder):
    """A loader whose samples carry no ``split_group`` (and no release enumeration entry)."""

    def build(self, **kw):  # type: ignore[no-untyped-def]
        dataset = super().build(**kw)
        dataset.samples = [
            replace(s, meta={k: v for k, v in s.meta.items() if k != "split_group"})
            for s in dataset.samples
        ]
        return dataset


def test_build_and_hf_export_fail_closed_naming_the_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = [_src("src-open", "open")]
    monkeypatch.setattr("marinedata.release.release_label_gate", lambda *a, **k: {})
    registry, roots, _ = _build(tmp_path, sources, allow_unmapped=True)
    release_dir = tmp_path / "rel" / "releases" / "r1" / "open"

    monkeypatch.setattr("marinedata.hf_export.DatasetBuilder", _NoGroup)
    with pytest.raises(HFExportError, match="src-open"):
        collect_rows(registry, roots, release_dir, "ship-open")

    monkeypatch.setattr("marinedata.release.DatasetBuilder", _NoGroup)
    with pytest.raises(SplitGroupError, match="src-open"):
        build_release(
            registry,
            release="r2",
            split_map=release_dir / "SPLIT_MAP.json",
            roots=roots,
            out_dir=tmp_path / "rel2",
            profile="ship-open",
            flavour="open",
            allow_unmapped=True,
        )


def test_source_without_a_crosswalk_is_excluded_from_the_task_not_the_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("marinedata.release.release_label_gate", lambda *a, **k: {})
    lab = _src("src-lab", "open")
    lab = lab.model_copy(
        update={
            "annotations": (Annotation(kind=AnnotationKind.IMAGE_LABEL, supervises=("taxon",)),)
        }
    )
    registry, roots, result = _build(tmp_path, [_src("src-ok", "open"), lab])
    assert [t.task_id for t in result.tasks] == ["pretrain-set"]  # the task is not skipped
    assert result.skipped_tasks == ()
    release_dir = tmp_path / "rel" / "releases" / "r1" / "open"
    release = json.loads((release_dir / "RELEASE.json").read_text())
    assert release["task_source_exclusions"] == [
        {
            "task": "pretrain-set",
            "source": "src-lab",
            "reason": "no crosswalk into schema 'rs-benthic-v1' for task 'pretrain-set'",
        }
    ]
    rows = (release_dir / "tasks" / "pretrain-set.tsv").read_text().splitlines()[1:]
    assert rows  # src-ok's images are in the manifest
    # hf_export rebuilds the same rows (the exclusion is applied there too)
    rebuilt = collect_rows(registry, roots, release_dir, "ship-open")["pretrain-set"]
    assert {r.source_id for r in rebuilt} == {"src-ok"}


def test_metadata_carries_the_frozen_map_split_of_an_upstream_test_image(tmp_path: Path) -> None:
    registry = _registry([_src("src-open", "open")])
    rows = [
        *_rows("src-open", b"open-shared"),
        ("up", "src-open/up", "TEST", b"upstream-test-image"),
    ]
    root = _stage(tmp_path / "src-open", rows)
    out = tmp_path / "M.json"
    generate_split_map(
        registry,
        out=out,
        roots={"src-open": root},
        profile="ship-open",
        now=NOW,
        seed=0,
        min_groups=3,
    )
    split_map = load_split_map(out)
    assert split_map is not None and split_map.assignments["src-open/up"] == "test"
    refs = [
        ImageRef("a" * 64, "src-open", root / "images" / "p" / "up.img", "test"),
        ImageRef("b" * 64, "src-open", root / "images" / "p" / "g0a.img", "train"),
    ]
    built = build_rows(refs, registry, Path("."), {}, source_roots={"src-open": root})
    assert {r["image_sha256"]: r["split"] for r in built} == {"a" * 64: "test", "b" * 64: "train"}
    assert {r["image_sha256"]: r["split_group"] for r in built}["a" * 64] == "src-open/up"
