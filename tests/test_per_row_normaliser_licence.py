"""WP-R2b: the release path's per-row licence comes from the ``metadata_norm`` per-row
normaliser (FathomNet staged label JSON), not from a new ingest column."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cli_release import _image_bytes, _registry, _source
from test_per_row_licence import _manifest

from marinedata.checksums import file_digest
from marinedata.cli import main
from marinedata.enums import AccessClass, Tier
from marinedata.metadata_norm.local import staged_row_licences
from marinedata.models import Licence
from marinedata.registry import Registry
from marinedata.tables import StagedImage, write_metadata_table

BOX_LICENCES = {"u-by": "CC-BY-4.0", "u-nc": "CC-BY-NC-4.0", "u-nd": "CC-BY-ND-4.0", "u-none": None}


def _stage_fathomnet(root: Path) -> dict[str, str]:
    rows, shas = [], {}
    (root / "labels" / "files").mkdir(parents=True)
    for i, (uuid, lic) in enumerate(BOX_LICENCES.items()):
        img = root / "images" / "default" / f"{uuid}.jpg"
        img.parent.mkdir(parents=True, exist_ok=True)
        img.write_bytes(_image_bytes(700 + i))
        shas[uuid] = file_digest(img)
        box = {"annotationLicense": lic} if lic else {"concept": "x"}
        rec = {"uuid": uuid, "boundingBoxes": [box], "contributorsEmail": "a@mbari.org"}
        (root / "labels" / "files" / f"{uuid}.json").write_text(json.dumps(rec))
        rows.append(
            StagedImage(stem=uuid, partition="default", upstream_path=f"o/{uuid}.jpg",
                        upstream_split=None, width=4, height=4, split_group=f"fn/g{i}")
        )  # fmt: skip
    write_metadata_table(root / "metadata.parquet", rows)  # NO license column
    return shas


def test_staged_row_licences_reads_the_label_json(tmp_path) -> None:
    _stage_fathomnet(tmp_path)
    got = staged_row_licences("fathomnet", tmp_path)
    assert got == {"u-by": "CC-BY-4.0", "u-nc": "CC-BY-NC-4.0", "u-nd": "CC-BY-ND-4.0"}


def test_inat_row_licence_comes_from_the_staged_column(tmp_path) -> None:
    rows = [
        StagedImage(stem="111", partition="default", upstream_path="p/111.jpg", upstream_split=None,
                    width=4, height=4, split_group="o/1", license="CC0-1.0"),
        StagedImage(stem="222", partition="default", upstream_path="p/222.jpg", upstream_split=None,
                    width=4, height=4, split_group="o/2"),
    ]  # fmt: skip
    write_metadata_table(tmp_path / "metadata.parquet", rows)
    assert staged_row_licences("inat-marine", tmp_path) == {"111": "CC0-1.0"}


@pytest.fixture
def fathomnet_build(monkeypatch, tmp_path):
    base = _registry()
    lic = Licence(id="PER-SOURCE-VARIES", name="varies", tier=Tier.TDM_ONLY)
    src = _source().model_copy(
        update={"id": "fathomnet", "name": "fathomnet", "licence": lic, "licence_per_row": True,
                "access_class": AccessClass.RESTRICTED_ND}
    )  # fmt: skip
    reg = type(base)(
        sources={"fathomnet": src}, licences={}, profiles={p.id: p for p in base.profiles},
        schemas={s.id: s for s in base.schemas}, tasks={t.id: t for t in base.tasks},
    )  # fmt: skip
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: reg))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "src" / "fathomnet"
    shas = _stage_fathomnet(root)

    def run(*extra: str) -> int:
        args = ["release", "build", "--release", "r1", "--split-map", str(tmp_path / "sm.json")]
        args += ["--out", str(tmp_path / "out"), "--local-only", f"--local=fathomnet={root}"]
        return main([*args, *extra])

    return run, shas, tmp_path / "out" / "releases" / "r1"


def test_fathomnet_cc_by_label_json_reaches_the_open_flavour(fathomnet_build) -> None:
    run, shas, rel = fathomnet_build
    assert run("--flavour", "open", "--generate-split-map") == 0
    assert run("--flavour", "nc") == 0
    assert _manifest(rel, "open") == {shas["u-by"]}
    assert _manifest(rel, "nc") == {shas["u-nc"]}
    assert shas["u-nd"] not in _manifest(rel, "open") | _manifest(rel, "nc")
    assert shas["u-none"] not in _manifest(rel, "open") | _manifest(rel, "nc")
