"""``marinedata release build`` — the ``--split-map`` / ``--generate-split-map`` gate.

A release must never silently allocate the permanent, append-only ``SPLIT_MAP.json``.
``release build`` refuses to run against a missing ``--split-map`` unless
``--generate-split-map`` is also passed, and refuses to overwrite an existing map even
then — see the module docstring in :mod:`marinedata.cli_release`.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

import marinedata.cli_release as cli_release
from marinedata.cli import main
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
    Verification,
)
from marinedata.registry import Registry
from marinedata.schema import Axis, LabelSchema
from marinedata.tables import StagedImage, write_metadata_table
from marinedata.task import TaskKind, TaskSpec

SOURCE_ID = "src-a"


def _source() -> Source:
    return Source(
        id=SOURCE_ID,
        name=SOURCE_ID,
        description="Fixture staged source for the release-build gate test.",
        version="v1",
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


def _registry() -> Registry:
    schema = LabelSchema(id="rs-benthic-v1", name="Fixture", axes=(Axis.TAXON,), nodes=())
    tasks = {"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)}
    profile = Profile(
        id="research",
        description="fixture",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
    )
    return Registry(
        sources={SOURCE_ID: _source()},
        licences={},
        profiles={"research": profile},
        schemas={"rs-benthic-v1": schema},
        tasks=tasks,
    )


def _image_bytes(seed: int) -> bytes:
    """A real, decodable image — ``release build`` dHashes every staged image (WS-D S47)
    — smooth random noise, so no two seeds are near-duplicates of each other."""
    import io
    import random

    from PIL import Image

    rng = random.Random(seed)
    small = Image.new("L", (9, 8))
    small.putdata([rng.randrange(256) for _ in range(72)])
    buf = io.BytesIO()
    small.resize((72, 64), Image.Resampling.BICUBIC).save(buf, format="PNG")
    return buf.getvalue()


def _stage(root: Path) -> Path:
    # Enough distinct groups (> DEFAULT_MIN_GROUPS) that a real 70/15/15 split can
    # populate every split non-empty — this test is about the --split-map gate, not
    # about exercising the splitter's edge cases.
    staged = []
    for i in range(10):
        stem = f"a{i}"
        image_path = root / "images" / "p" / f"{stem}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(_image_bytes(i))
        staged.append(
            StagedImage(
                stem=stem,
                partition="p",
                upstream_path=f"orig/{stem}.jpg",
                upstream_split=None,
                width=4,
                height=4,
                split_group=f"{SOURCE_ID}/group{i}",
            )
        )
    write_metadata_table(root / "metadata.parquet", staged)
    return root


def _run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, split_map: Path, extra: list[str]) -> int:
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: _registry()))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))
    root = _stage(tmp_path / "src")
    return main(
        [
            "release",
            "build",
            "--release",
            "r1",
            "--split-map",
            str(split_map),
            "--out",
            str(tmp_path / "out"),
            "--profile",
            "research",
            "--local",
            f"{SOURCE_ID}={root}",
            *extra,
        ]
    )


def test_missing_split_map_without_flag_refuses_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    split_map = tmp_path / "SPLIT_MAP.json"
    code = _run(monkeypatch, tmp_path, split_map, [])
    assert code != 0
    assert not split_map.exists()
    err = capsys.readouterr().err
    assert str(split_map) in err
    assert "--generate-split-map" in err


def test_missing_split_map_with_flag_generates_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_map = tmp_path / "SPLIT_MAP.json"
    code = _run(monkeypatch, tmp_path, split_map, ["--generate-split-map"])
    assert code == 0
    assert split_map.exists()


def test_generate_flag_with_existing_path_refuses_and_leaves_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    split_map = tmp_path / "SPLIT_MAP.json"
    code = _run(monkeypatch, tmp_path, split_map, ["--generate-split-map"])
    assert code == 0
    before = split_map.read_bytes()

    code = _run(monkeypatch, tmp_path, split_map, ["--generate-split-map"])
    assert code != 0
    assert split_map.read_bytes() == before
    err = capsys.readouterr().err
    assert str(split_map) in err


def test_existing_split_map_without_flag_is_unchanged_behaviour(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_map = tmp_path / "SPLIT_MAP.json"
    code = _run(monkeypatch, tmp_path, split_map, ["--generate-split-map"])
    assert code == 0
    before = split_map.read_bytes()

    code = _run(monkeypatch, tmp_path, split_map, [])
    assert code == 0
    assert split_map.read_bytes() == before, "a frozen release build must never write to the map"


def test_pillow_mismatch_against_a_frozen_map_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A split map's near-dup dHashes are only valid under the Pillow version that
    computed them (LANCZOS resize is a Pillow implementation detail, WS-D S49). A
    frozen map built under a different Pillow than the running one must fail the
    build closed, naming both versions, rather than silently trust stale dHashes."""
    split_map = tmp_path / "SPLIT_MAP.json"
    code = _run(monkeypatch, tmp_path, split_map, ["--generate-split-map"])
    assert code == 0
    before = split_map.read_bytes()

    monkeypatch.setattr(cli_release, "pil_version", lambda: "99.0.0")
    code = _run(monkeypatch, tmp_path, split_map, [])
    assert code != 0
    assert split_map.read_bytes() == before
    err = capsys.readouterr().err
    assert "99.0.0" in err
    from marinedata.neardup import pil_version as _real_pil_version

    assert _real_pil_version() in err


def test_fetch_failure_for_admitted_source_fails_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An admitted source with no ``--local`` override whose fetch raises must fail the
    whole build (non-zero exit, the source id and error named on stderr) rather than
    silently produce a smaller release — see :class:`marinedata.cli_release.ReleaseFetchError`."""
    monkeypatch.setattr(cli_release, "_is_pinned_staged_tree", lambda source: True)

    def _boom(source: object, *, limit: int) -> object:
        raise cli_release.FetchError(f"{SOURCE_ID}: connection reset")

    monkeypatch.setattr(cli_release, "fetch_sample", _boom)
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: _registry()))

    out_dir = tmp_path / "out"
    split_map = tmp_path / "SPLIT_MAP.json"
    code = main(
        [
            "release",
            "build",
            "--release",
            "r1",
            "--split-map",
            str(split_map),
            "--out",
            str(out_dir),
            "--profile",
            "research",
            "--generate-split-map",
        ]
    )
    assert code != 0
    err = capsys.readouterr().err
    assert SOURCE_ID in err
    assert "connection reset" in err
    assert not split_map.exists()
    assert not out_dir.exists()
