"""WP-R2: a ``licence_per_row`` source (FathomNet, iNat, ...) ships row by row.

CC0 / CC-BY rows -> open flavour, BY-NC rows -> nc flavour, ND / unknown / missing -> never."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_cli_release import _image_bytes, _registry, _source

from marinedata.checksums import file_digest
from marinedata.cli import main
from marinedata.enums import AccessClass, Tier
from marinedata.hf_export import SampleRow
from marinedata.licence_class import flavour_filter
from marinedata.models import Licence
from marinedata.registry import Registry
from marinedata.tables import StagedImage, write_metadata_table

ROW_LICENCES = {
    "cc0": "CC0-1.0",
    "by": "CC-BY-4.0",
    "by2": "CC-BY-4.0",
    "nc": "CC-BY-NC-4.0",
    "nd": "CC-BY-ND-4.0",
    "arr": "All rights reserved",
    "none": None,
}
OPEN_STEMS, NC_STEMS = {"cc0", "by", "by2"}, {"nc"}


def _stage_per_row(root: Path) -> dict[str, str]:
    rows, shas = [], {}
    for i, (stem, lic) in enumerate(ROW_LICENCES.items()):
        p = root / "images" / "p" / f"{stem}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_image_bytes(500 + i))
        shas[stem] = file_digest(p)
        rows.append(
            StagedImage(stem=stem, partition="p", upstream_path=f"o/{stem}.jpg",
                        upstream_split=None, width=4, height=4, split_group=f"row/g{i}",
                        license=lic)
        )  # fmt: skip
    write_metadata_table(root / "metadata.parquet", rows)
    return shas


@pytest.fixture
def per_row_build(monkeypatch, tmp_path):
    base = _registry()
    lic = Licence(id="PER-SOURCE-VARIES", name="varies", tier=Tier.TDM_ONLY)
    row = _source().model_copy(
        update={"id": "row1", "name": "row1", "licence": lic, "licence_per_row": True,
                "access_class": AccessClass.RESTRICTED_ND}
    )  # fmt: skip
    reg = type(base)(
        sources={"row1": row},
        licences={},
        profiles={p.id: p for p in base.profiles},
        schemas={s.id: s for s in base.schemas},
        tasks={t.id: t for t in base.tasks},
    )
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: reg))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "src" / "row1"
    shas = _stage_per_row(root)

    def run(*extra: str) -> int:
        args = ["release", "build", "--release", "r1", "--split-map", str(tmp_path / "sm.json")]
        args += ["--out", str(tmp_path / "out"), "--local-only", f"--local=row1={root}"]
        return main([*args, *extra])

    return run, shas, tmp_path / "out" / "releases" / "r1"


def _manifest(rel: Path, flavour: str) -> set[str]:
    path = rel / flavour / "tasks" / "pretrain-set.tsv"
    return {ln.split("\t")[0] for ln in path.read_text().splitlines()[1:]}


def test_per_row_licences_land_in_the_right_flavour(per_row_build) -> None:
    run, shas, rel = per_row_build
    assert run("--flavour", "open", "--generate-split-map") == 0
    assert run("--flavour", "nc") == 0
    open_shas, nc_shas = _manifest(rel, "open"), _manifest(rel, "nc")
    assert open_shas == {shas[s] for s in OPEN_STEMS}
    assert nc_shas == {shas[s] for s in NC_STEMS}
    dropped = {shas[s] for s in ("nd", "arr", "none")}
    assert not (dropped & (open_shas | nc_shas))
    assert len(open_shas) == 3 and len(nc_shas) == 1


def test_sample_rows_carry_a_per_row_class_the_flavour_filter_uses() -> None:
    def row(sid: str, cls: str | None) -> SampleRow:
        return SampleRow(task_id="t", raw_split="train", image_sha256=sid, source_id="fathomnet",
                         sample_key=sid, image=Path(sid), licence_class=cls)  # fmt: skip

    rows = [row("a", "open"), row("b", "restricted-nc"), row("c", "restricted-nd"), row("d", None)]
    assert [r.image_sha256 for r in flavour_filter(rows, "open")] == ["a"]
    assert [r.image_sha256 for r in flavour_filter(rows, "nc")] == ["b"]  # None -> unknown, dropped
