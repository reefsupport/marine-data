"""PR-E, E3/E4: the v1.1.0 dataset card, root files and Croissant of a synthetic export."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from test_hf_export_v11 import _layout, _rows
from test_hf_export_v11 import build as _build
from test_hf_export_v11 import registry as _registry

from marinedata import card_v11_text as txt
from marinedata import hf_card
from marinedata.cli import main as cli_main
from marinedata.hf_card_v11 import write_card_files
from marinedata.hf_export_v11 import PRIVATE_NAMES, V11ExportError, export_v11
from marinedata.release_card import load_context
from marinedata.release_croissant import build_croissant, validate

build, registry = _build, _registry  # the synthetic 12-table build of test_hf_export_v11
REPO = {"open": "reefsupport/marine-data", "nc": "reefsupport/marine-data-nc"}
FILES = ("README.md", "LICENSE", "NOTICE", "CITATION.cff")
V1_OPEN = ["coral-masks", "scene-masks", "instance-masks", "fish-boxes"]
FIXTURE = Path(__file__).parent / "data" / "v1.0-card-front-matter.yaml"
CFF = {
    "cff-version": "1.2.0",
    "message": "If you use this dataset, please cite it as below.",
    "title": "Reef Support Marine Data",
    "type": "dataset",
    "version": "1.0.0",
    "date-released": "2026-10-07",
    "authors": [{"family-names": "Runhaar", "given-names": "Yohan"}],
    "url": "https://huggingface.co/datasets/reefsupport/marine-data",
}


def _export(registry, build, tmp_path, flavour="open"):
    layout, _ = _layout(registry, build, flavour=flavour, shard_target_bytes=2000)
    out = tmp_path / f"export-{flavour}"
    export_v11(layout, out)
    write_card_files(
        out, registry, flavour=flavour, repo_id=REPO[flavour], cff=CFF, released="2026-10-16"
    )
    return out


def _front(readme: str) -> tuple[list[str], dict]:
    head = readme.split("---\n", 2)[1]
    return head.splitlines(), yaml.safe_load(head)


def _cff_file(tmp_path: Path) -> Path:
    path = tmp_path / "repo-CITATION.cff"
    path.write_text(yaml.safe_dump(CFF))
    return path


def _croissant(out, build, tmp_path, flavour="open"):
    ctx = load_context(
        out, release_dir=build[0], citation=_cff_file(tmp_path), repo_id=REPO[flavour]
    )
    return build_croissant(out, ctx)


def test_front_matter_keeps_the_v10_configs_byte_for_byte(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    lines, _ = _front((out / "README.md").read_text())
    fixture = FIXTURE.read_text().splitlines()
    start = lines.index("configs:")
    assert lines[start : start + 33] == fixture[37:70]


def test_front_matter_adds_the_new_configs_with_metadata_as_default(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    _, front = _front((out / "README.md").read_text())
    names = [c["config_name"] for c in front["configs"]]
    new = sorted(p.name for p in (out / "data").iterdir())
    assert names[:4] == V1_OPEN
    assert sorted(names[4:]) == new
    assert [c["config_name"] for c in front["configs"] if c.get("default")] == ["metadata"]
    assert sorted(d["config_name"] for d in front["dataset_info"]) == new
    assert front["license"] == "other"
    assert front["license_link"] == "LICENSE"
    assert "extra_gated_prompt" not in front
    for entry in front["configs"][4:]:
        assert all(
            f["path"].startswith(f"data/{entry['config_name']}/") for f in entry["data_files"]
        )


def test_card_has_every_section_and_the_measured_tables(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    text = (out / "README.md").read_text()
    assert re.findall(r"^## (.+)$", text, flags=re.M) == [
        "Dataset summary",
        "Supported tasks",
        "Quick start",
        "Dataset structure",
        "Label space",
        "Splits",
        "Dataset creation",
        "Personal and sensitive information",
        "Limitations and known defects",
        "Licensing",
        "Excluded sources",
        "Takedown",
        "Versions and changelog",
        "Citation",
        "Contact",
    ]
    for needle in (
        "| Config | Task |",
        "test_ood",
        "| Taxonomy node |",
        "### Mapped share per source",
        "| Source | Image licence |",
        "`seaview-survey-imagery`",
        "load_dataset(",
        'revision="v1.1.0"',
        "@misc{",
        "unmapped",
    ):
        assert needle in text, needle
    assert text.count("\n---\n") == 1  # one front matter block, closed once


@pytest.mark.parametrize("flavour", ["open", "nc"])
def test_no_public_file_names_a_private_resource(registry, build, tmp_path, flavour):
    out = _export(registry, build, tmp_path, flavour)
    banned = PRIVATE_NAMES if flavour == "open" else PRIVATE_NAMES[1:]
    for name in FILES:
        text = (out / name).read_text()
        assert not [b for b in banned if b in text], (flavour, name)


def test_nc_card_is_gated_and_says_so(registry, build, tmp_path):
    out = _export(registry, build, tmp_path, "nc")
    text = (out / "README.md").read_text()
    _, front = _front(text)
    names = [c["config_name"] for c in front["configs"]]
    assert names[:2] == ["coral-masks", "coral-masks-machine"]
    assert "taxonomy" not in names
    assert front["license_name"] == "mixed-non-commercial"
    assert front["extra_gated_prompt"]
    assert "NON-COMMERCIAL" in text
    assert "non-commercial" in (out / "LICENSE").read_text().lower()


def test_open_card_refuses_to_name_the_private_repo(registry, build, tmp_path):
    layout, _ = _layout(registry, build, shard_target_bytes=2000)
    out = tmp_path / "export"
    export_v11(layout, out)
    with pytest.raises(V11ExportError):
        write_card_files(out, registry, flavour="open", repo_id=REPO["nc"], cff=CFF)
    assert not any((out / name).exists() for name in FILES)


def test_license_notice_and_citation_files(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    license_text, notice = (out / "LICENSE").read_text(), (out / "NOTICE").read_text()
    for row in _rows(out, "sources"):
        assert row["source_id"] in license_text
        assert row["source_id"] in notice
    assert "Marine Ecoregions of the World" in notice
    assert "marinespecies.org" in notice
    cff = yaml.safe_load((out / "CITATION.cff").read_text())
    assert cff["version"] == "1.1.0"
    assert str(cff["date-released"]) == "2026-10-16"
    assert cff["title"] == CFF["title"]
    assert cff["authors"] == CFF["authors"]
    assert CFF["version"] == "1.0.0"  # the input is not mutated


def test_card_stage_delegates_a_v11_export(registry, build, tmp_path):
    layout, _ = _layout(registry, build, shard_target_bytes=2000)
    out = tmp_path / "export"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(export_v11(layout, out)))
    argv = ["--release-dir", str(build[0]), "--summary", str(summary), "--out", str(out)]
    argv += ["--citation", str(_cff_file(tmp_path))]
    assert hf_card.main(argv) == 0
    assert all((out / name).is_file() for name in FILES)
    assert "## Dataset summary" in (out / "README.md").read_text()


def test_croissant_has_splits_keys_references_and_rai(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    doc = _croissant(out, build, tmp_path)
    sets = {r["@id"]: r for r in doc["recordSet"]}
    for cfg in sorted(p.name for p in (out / "data").iterdir()):
        splits = sets[f"{cfg}_splits"]
        assert splits["dataType"] == "cr:Split"
        assert any(f["@id"] == f"{cfg}/hub_split" for f in sets[cfg]["field"])
        assert [d[f"{cfg}_splits/split_name"] for d in splits["data"]]
    assert sets["metadata"]["key"] == {"@id": "metadata/image_sha256"}
    keys = [k["@id"] for k in sets["points"]["key"]]
    assert keys[:2] == ["points/image_sha256", "points/source_id"]
    refs = {
        f["@id"]: f["references"]["field"]["@id"]
        for r in doc["recordSet"]
        for f in r["field"]
        if f["@id"].endswith("/image_sha256") and "references" in f
    }
    assert refs
    assert set(refs.values()) == {"metadata/image_sha256"}
    assert "metadata/image_sha256" not in refs
    assert doc["version"] == "1.1.0"
    assert doc["datePublished"]
    assert all(doc[k] for k in txt.RAI)
    assert not [n for n in PRIVATE_NAMES if n in json.dumps(doc)]


def test_croissant_validates_with_mlcroissant(registry, build, tmp_path):
    pytest.importorskip("mlcroissant", reason="mlcroissant is a dev/croissant extra")
    out = _export(registry, build, tmp_path)
    status, detail = validate(_croissant(out, build, tmp_path))
    assert status == "valid", detail


def test_nc_croissant_builds_and_validates(registry, build, tmp_path):
    pytest.importorskip("mlcroissant", reason="mlcroissant is a dev/croissant extra")
    out = _export(registry, build, tmp_path, "nc")
    (build[0] / "RELEASE.json").write_text(json.dumps({"flavour": "nc", "release": "v1.1"}))
    doc = _croissant(out, build, tmp_path, "nc")
    assert validate(doc)[0] == "valid"
    assert "s3://" not in json.dumps(doc)


def test_release_croissant_cli_contract(registry, build, tmp_path):
    out = _export(registry, build, tmp_path)
    argv = ["release", "croissant", str(out), "--release-dir", str(build[0])]
    argv += ["--citation", str(_cff_file(tmp_path)), "--repo-id", REPO["open"]]
    assert cli_main(argv) == 0
    doc = json.loads((out / "croissant.json").read_text())
    assert doc["version"] == "1.1.0"
    assert "metadata_splits" in {r["@id"] for r in doc["recordSet"]}
