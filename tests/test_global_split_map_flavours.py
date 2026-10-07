"""WP-R2d: one global split map serves both flavours.

The map is generated ONCE over open + nc sources; the open and the nc build both read it,
so an image shared across an open and an nc source has the same split in both repos, and
an nc build that roots a source the map cannot cover (flat-images) no longer raises.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from test_upstream_test_split import _source, _stage

from marinedata.enums import AccessClass, Tier
from marinedata.models import LoaderSpec, Profile
from marinedata.registry import Registry
from marinedata.release import build_release, generate_split_map
from marinedata.schema import Axis, LabelSchema
from marinedata.task import TaskKind, TaskSpec

NOW = "2026-10-05T00:00:00+00:00"


def _src(source_id: str, access: str, layout: str = "staged-tree"):  # type: ignore[no-untyped-def]
    base = _source(source_id)
    return base.model_copy(
        update={"access_class": AccessClass(access), "loader": LoaderSpec(layout=layout, params={})}
    )


def _registry(sources) -> Registry:  # type: ignore[no-untyped-def]
    both = (Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT)
    profiles = {
        "ship-open": Profile(
            id="ship-open", description="f", allow_tiers=both, allow_access_classes=("open",)
        ),
        "ship-noncommercial": Profile(
            id="ship-noncommercial",
            description="f",
            allow_tiers=both,
            allow_access_classes=("open", "restricted-nc"),
        ),
    }
    return Registry(
        sources={s.id: s for s in sources},
        licences={},
        profiles=profiles,
        schemas={
            "rs-benthic-v1": LabelSchema(id="rs-benthic-v1", name="F", axes=(Axis.TAXON,), nodes=())
        },
        tasks={"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)},
    )


def _rows(prefix: str, shared: bytes) -> list:  # type: ignore[no-untyped-def]
    rows = [
        (f"g{g}{k}", f"{prefix}/g{g}", "train", f"{prefix}-{g}-{k}".encode())
        for g in range(12)
        for k in "ab"
    ]
    return [*rows, ("shared", f"{prefix}/shared", "train", shared)]


def _manifest(out: Path, flavour: str) -> dict[str, str]:
    path = out / "releases" / "r1" / flavour / "tasks" / "pretrain-set.tsv"
    return dict(line.split("\t") for line in path.read_text().splitlines()[1:])


def test_one_map_two_flavours_same_split_and_nc_build_does_not_raise(tmp_path: Path) -> None:
    registry = _registry(
        [
            _src("src-open", "open"),
            _src("src-nc", "restricted-nc"),
            _src("src-vqa", "restricted-nc", layout="metadata-only").model_copy(
                update={"release_skip_reason": "context-layer"}
            ),
        ]
    )
    shared = b"the-same-image-in-an-open-and-an-nc-source"
    vqa_root = tmp_path / "vqa"
    vqa_root.mkdir()
    roots = {
        "src-open": _stage(tmp_path / "open", _rows("src-open", shared)),
        "src-nc": _stage(tmp_path / "nc", _rows("src-nc", shared)),
        "src-vqa": vqa_root,
    }
    split_map = tmp_path / "SPLIT_MAP.json"
    skipped: dict[str, str] = {}
    generate_split_map(
        registry,
        out=split_map,
        roots=roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=0,
        min_groups=3,
        skipped=skipped,
    )
    assert "src-vqa" in skipped  # a registry reason: the map never enumerates it

    out = tmp_path / "rel"
    results = {
        flavour: build_release(
            registry,
            release="r1",
            split_map=split_map,
            roots=roots,
            out_dir=out,
            profile=profile,
            flavour=flavour,
            allow_unmapped=True,
        )
        for flavour, profile in (("open", "ship-open"), ("nc", "ship-noncommercial"))
    }
    assert results["open"].sources == ("src-open",)
    assert results["nc"].sources == ("src-nc",)  # src-vqa (registry reason) dropped, no raise
    release_nc = json.loads((out / "releases" / "r1" / "nc" / "RELEASE.json").read_text())
    assert [s["id"] for s in release_nc["skipped_sources"]] == ["src-vqa"]
    sha = hashlib.sha256(shared).hexdigest()
    assert _manifest(out, "open")[sha] == _manifest(out, "nc")[sha]
    maps = [(out / "releases" / "r1" / f / "SPLIT_MAP.json").read_bytes() for f in ("open", "nc")]
    assert maps[0] == maps[1] == split_map.read_bytes()
