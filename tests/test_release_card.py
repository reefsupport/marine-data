"""P1.9: the dataset card and Croissant generators (offline; fixture builds only)."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

import pytest
import yaml
from test_e2e_release import _pipeline

from marinedata import card_body as cb
from marinedata import release_croissant
from marinedata.card_fields import field_doc
from marinedata.cli import main
from marinedata.licence_class import unflavoured_for_tests
from marinedata.metadata_norm.resolvers.join import POINTS_COLUMNS
from marinedata.metadata_release import METADATA_COLUMNS
from marinedata.registry import Registry
from marinedata.release_card import front_matter, render_card, source_info
from marinedata.release_croissant import validate
from marinedata.task_layers.mask_configs import INSTANCE_COLUMNS, SEMANTIC_COLUMNS

HERE = Path(__file__).parent
DATA, GOLDEN, REPO = HERE / "data", HERE / "golden", HERE.parent
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"


def _names(columns) -> list[str]:
    return [c if isinstance(c, str) else getattr(c, "name", None) or c[0] for c in columns]


def _golden(name: str, text: str) -> None:
    path = GOLDEN / name
    if UPDATE:
        path.write_text(text)
    assert text == path.read_text(), f"{name} changed; rerun with UPDATE_GOLDEN=1 to accept"


def _tables(lines: list[str], only: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    heading = None
    for line in lines:
        found = re.match(r"### (.+)$", line)
        if found:
            heading = found[1]
        elif line.startswith("## "):
            heading = None
        elif heading in only and line.startswith("|"):
            out.setdefault(heading, []).append(
                "|".join(cell.strip() for cell in line.strip().strip("|").split("|"))
            )
    return out


@pytest.fixture(scope="module")
def v1() -> tuple[dict, dict]:
    facts = json.loads((DATA / "v1.0-card-facts.json").read_text())
    registry = Registry.load()
    sids = dict.fromkeys(s for c in facts["configs"].values() for s in c["sources"])
    return facts, {sid: source_info(registry, sid) for sid in sids}


def test_v1_front_matter_is_exactly_the_hand_written_one(v1) -> None:
    facts, infos = v1
    got = front_matter(
        facts,
        infos,
        flavour="open",
        repo_id="reefsupport/marine-data",
        pretty_name=facts["pretty_name"],
    )
    assert got == (DATA / "v1.0-card-front-matter.yaml").read_text().rstrip("\n")


def test_v1_tables_match_the_hand_written_card_modulo_whitespace(v1) -> None:
    facts, infos = v1
    configs = facts["configs"]
    shown = {
        sid: {**i, "_licence_display": facts["licence_display"].get(sid)}
        for sid, i in infos.items()
    }
    keep = {"Configurations", *configs}
    gold = _tables((DATA / "v1.0-card-tables.md").read_text().splitlines(), keep)
    got_text = cb.configurations_table(configs)
    got = _tables(
        [
            "### Configurations",
            *got_text.splitlines(),
            *cb.config_sections(configs, shown).splitlines(),
        ],
        keep,
    )
    assert set(gold) == {"Configurations", *configs} - {"scene-masks"}
    assert got == gold


def test_v1_card_renders_all_generated_sections(v1) -> None:
    facts, infos = v1
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    card = render_card(facts, infos, repo_id="reefsupport/marine-data", cff=cff)
    assert card.startswith("---\n" + (DATA / "v1.0-card-front-matter.yaml").read_text() + "---\n")
    for heading in (
        "## Dataset summary",
        "### Configurations",
        "### Data fields",
        "## Licensing",
        "## Takedown requests and corrections",
        "## Versions and changelog",
        "## Citation",
    ):
        assert heading in card
    assert "58,803" in card and "### points" not in card


def test_every_layout_column_has_a_one_line_description() -> None:
    layout = {
        *_names(SEMANTIC_COLUMNS),
        *_names(INSTANCE_COLUMNS),
        *_names(METADATA_COLUMNS),
        *_names(POINTS_COLUMNS),
    }
    assert len(_names(POINTS_COLUMNS)) == 15
    missing = sorted(c for c in layout if not field_doc(c))
    assert not missing, f"columns without a description in card_fields.FIELD_DOCS: {missing}"
    assert all("\n" not in field_doc(c) for c in layout)


def test_points_config_columns_appear_in_the_data_fields() -> None:
    features = [{"name": n, "dtype": "string"} for n in _names(POINTS_COLUMNS)]
    configs = {"points": {"features": features, "splits": {}, "download_size": 0}}
    text, missing = cb.data_fields(configs)
    assert not missing
    assert all(f"| `{n}` |" in text for n in _names(POINTS_COLUMNS))


def test_gates_table_summarises_gates_json() -> None:
    table = cb.gates_table(
        {
            "card": {"passed": True, "counts": {"configs": 4}},
            "dedup": {"passed": False, "failures": ["a", "b"]},
            "nc": {"skipped": "no nc build"},
        }
    )
    assert "| `card` | passed | configs=4 |" in table
    assert "| `dedup` | failed | 2 failure(s) |" in table
    assert "| `nc` | skipped | no nc build |" in table
    assert "No `GATES.json`" in cb.gates_table({})


@pytest.fixture(scope="module")
def e2e(tmp_path_factory) -> dict:
    tmp = tmp_path_factory.mktemp("card-e2e")
    with (
        pytest.MonkeyPatch.context() as mp,
        unflavoured_for_tests(),
    ):  # module scope: the autouse guard is off
        run = _pipeline(tmp, mp)  # the shared e2e fixture: lock, release build, hf_export
        export_dir, release_dir = run.export_dir, run.release_dir
        common = ["--release-dir", str(release_dir), "--citation", str(REPO / "CITATION.cff")]
        readme, croissant = tmp / "README.md", tmp / "croissant.json"
        assert main(["release", "card", str(export_dir), *common, "--out", str(readme)]) == 0
        assert (
            main(["release", "croissant", str(export_dir), *common, "--out", str(croissant)]) == 0
        )
    return {"export_dir": export_dir, "readme": readme, "croissant": croissant}


def test_e2e_card_matches_golden(e2e) -> None:
    _golden("card-e2e-README.md", e2e["readme"].read_text())


def test_e2e_card_describes_every_exported_column(e2e) -> None:
    text = e2e["readme"].read_text()
    section = text.split("### Data fields")[1].split("\n## ")[0]
    assert not re.findall(r"^\| `\w+` \| [^|]+ \| – \|$", section, re.M)


def test_e2e_card_parses_with_huggingface_hub_and_datasets(e2e) -> None:
    hub = pytest.importorskip("huggingface_hub")
    metadata = pytest.importorskip("datasets.utils.metadata")
    card = hub.DatasetCard.load(e2e["readme"])
    data = card.data.to_dict()
    names = [c["config_name"] for c in data["configs"]]
    assert names and names == [i["config_name"] for i in data["dataset_info"]]
    assert list(metadata.MetadataConfigs.from_dataset_card_data(card.data)) == names


def test_e2e_croissant_matches_golden(e2e) -> None:
    _golden("croissant-e2e.json", e2e["croissant"].read_text())


def test_e2e_croissant_shape(e2e) -> None:
    doc = json.loads(e2e["croissant"].read_text())
    assert doc["conformsTo"] == "http://mlcommons.org/croissant/1.0"
    assert doc["version"] == "wp4-e2e" and doc["citeAs"].startswith("@misc{")
    shards = [
        d for d in doc["distribution"] if d["@type"] == "cr:FileObject" and d["@id"] != "repo"
    ]
    assert shards and all(re.fullmatch(r"[0-9a-f]{64}", s["sha256"]) for s in shards)
    assert all(s["contentSize"].endswith(" B") for s in shards)
    configs = list(dict.fromkeys(s["@id"].split("/")[0] for s in shards))
    assert [r["@id"] for r in doc["recordSet"]] == configs
    kinds = {f["dataType"] for r in doc["recordSet"] for f in r["field"] if "dataType" in f}
    assert "sc:ImageObject" in kinds and "sc:Text" in kinds


def test_e2e_croissant_validates_with_mlcroissant(e2e) -> None:
    pytest.importorskip("mlcroissant")
    status, detail = validate(json.loads(e2e["croissant"].read_text()))
    assert status == "valid", detail


def test_validate_reports_a_missing_validator(monkeypatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "mlcroissant", None)
    status, detail = validate({})
    assert status == "not-validated" and "mlcroissant" in detail


def test_validate_does_not_mutate_the_document_or_the_shared_context(e2e) -> None:
    """mlcroissant adds `@base` to the dict it is given. It once leaked into every later document
    through the shared module-level CONTEXT, so the golden passed alone and failed in the suite."""
    doc = json.loads(e2e["croissant"].read_text())
    before = copy.deepcopy(doc)
    release_croissant.validate(doc)
    assert doc == before
    assert "@base" not in release_croissant.CONTEXT
