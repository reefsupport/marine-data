"""v1.2.0 card text: the Coralscapes V2 section, its numbers and its hook in the v1.1 renderer."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml
from test_hf_export_v11 import _layout
from test_hf_export_v11 import build as _build
from test_hf_export_v11 import registry as _registry

from marinedata import card_v11_text as txt
from marinedata import card_v12_text as v12
from marinedata.hf_card_v11 import V1_CONFIGS, card_version, write_card_files
from marinedata.hf_export_v11 import assert_public_safe, export_v11

build, registry = _build, _registry  # the synthetic 12-table build of test_hf_export_v11


V2 = [{"source_id": "coralscapes-v2", "n_images": 2433}]
ROOT = Path(__file__).resolve().parents[1]
MATCHING_JSON = ROOT / "results" / "label-matching-v1.2.0.json"
MANIFEST = ROOT / "registry" / "benchmarks" / "manifests" / "coralscapes-v2.parquet"
D7 = "V2 supersedes V1 for Coralscapes tasks; V1 configs are kept frozen for reproducibility"
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


REPO = {"open": "reefsupport/marine-data", "nc": "reefsupport/marine-data-nc"}


def _render(registry, build, root: Path, flavour: str = "open") -> tuple[str, dict]:
    layout, _ = _layout(registry, build, flavour=flavour, shard_target_bytes=2000)
    export_v11(layout, root)
    write_card_files(
        root, registry, flavour=flavour, repo_id=REPO[flavour], cff=CFF, released="2026-10-29",
    )  # fmt: skip
    return (root / "README.md").read_text(), yaml.safe_load((root / "CITATION.cff").read_text())


def _limitations(readme: str) -> str:
    return readme.split("## Limitations and known defects", 1)[1].split("\n## ", 1)[0]


def test_applies_only_for_the_v2_source():
    assert v12.applies(V2)
    assert not v12.applies([{"source_id": "coralscapes"}, {"source_id": "seaview-survey-imagery"}])
    assert not v12.applies([])
    assert card_version("open", V2) == v12.VERSION
    assert card_version("nc", V2) == txt.VERSION


def test_numbers_agree_with_each_other():
    assert sum(v12.V2_SPLITS.values()) == v12.V2_FRAMES == 2_433
    assert v12.TWINS + sum(v12.V2_NEW.values()) == v12.V2_FRAMES
    assert v12.TWINS + v12.V1_ONLY == 2_075
    assert sum(v12.EDGES.values()) == sum(v12.LEGACY.values()) == 95
    assert sorted(v12.NEW_NODES.values()) == [22549, 125269, 146762]


def test_section_states_the_release_facts():
    text = v12.section(V2)
    for needle in (
        v12.SUPERSEDES, "2,433 frames", "2,433 images in this release", "arXiv:2609.12826",
        v12.V2_REVISION, "Apache-2.0", "taxonomy 3.1.0", "CN_ANTIPATHARIA", "22549", "125269",
        "146762", "docs/label-matching.md", "2,074", "site30", "uint16", "1,790",
        "Downstream model pins are unchanged", "josauder", "@inproceedings{sauder2026",
        "- `mask_ref` is the object key in the public bucket",
        "(in v1.2.0: Coralscapes V2, `sources/<source_id>/<version>/labels/...`)",
        "the path relative to the source's staged tree `sources/<source_id>/<version>/`",
        "for the earlier sources.",
    ):  # fmt: skip
        assert needle in text, needle
    assert "<<" not in text and ">>" not in text
    assert "images in this release" not in v12.section([{"source_id": v12.SOURCE_ID}])
    assert_public_safe(text, "open", where="v1.2.0 section")


def test_limitations_replace_only_the_coralscapes_bullet():
    out = v12.limitations(txt.LIMITATIONS)
    assert out[0] == v12.LIMITATION
    shortened = [
        b.replace("sources table with its share.", "sources table.") for b in txt.LIMITATIONS[1:]
    ]
    assert list(out[1:]) == shortened  # the other bullets only lose the share clause
    with pytest.raises(ValueError, match="anchor"):
        v12.limitations(("something else", *txt.LIMITATIONS))


def test_card_without_v2_is_the_v110_card(registry, build, tmp_path):
    card, cff = _render(registry, build, tmp_path / "plain")
    assert "Coralscapes V2" not in card and 'revision="v1.1.0"' in card
    assert "Using labels across sources" not in card and v12.TEST_PAIR[0] not in card
    assert txt.LIMITATIONS[0] in _limitations(card)
    assert cff["version"] == "1.1.0"


def test_card_with_v2_gets_the_section_and_the_v120_statements(
    registry, build, tmp_path, monkeypatch
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, cff = _render(registry, build, tmp_path / "v2")
    start, mid, end = (
        card.index(h) for h in ("## Dataset structure", "## Coralscapes V2", "## Label space")
    )
    assert start < mid < end
    assert 'revision="v1.2.0"' in card and cff["version"] == "1.2.0"
    assert v12.SUPERSEDES in card and v12.SUMMARY in card and v12.VERSION_NOTE in card
    assert v12.LIMITATION in _limitations(card) and txt.LIMITATIONS[0] not in _limitations(card)
    assert "**v1.2.0**" in card and "**v1.1.0**" in card
    assert all(f"- {c}" in card for c in v12.CHANGELOG)
    assert all(f"- {c}" in card for c in txt.CHANGELOG)


def test_the_gate_slots_are_filled_by_the_runbook_step():
    from marinedata import card_v12_text as v12

    numbers = {"V120_GATES_RUN": 12, "V120_DEDUP_LINKED": 2074, "V120_SPLIT_TWIN_GROUPS": 2074}
    assert all(f"[[{slot}]]" in v12.GATES_PARAGRAPH for slot in v12.GATE_SLOTS)
    filled = v12.fill_gates(f"a {v12.GATES_PARAGRAPH} b", numbers)
    assert "[[" not in filled and "checks the 2074 V1/V2 stem-linked twin pairs" in filled
    assert "reports 2074 such pairs among the images stored in this release" in filled
    with pytest.raises(ValueError, match="missing"):
        v12.fill_gates(v12.GATES_PARAGRAPH, {"V120_GATES_RUN": 12})
    with pytest.raises(ValueError, match="unknown"):
        v12.fill_gates(v12.GATES_PARAGRAPH, {**numbers, "OTHER": 1})
    with pytest.raises(ValueError, match="no \\[\\[V120_GATES_RUN\\]\\] slot"):
        v12.fill_gates("no slots here", numbers)


def test_a_zero_linked_twin_count_fills_and_reads_honestly():
    numbers = {"V120_GATES_RUN": 12, "V120_DEDUP_LINKED": 0, "V120_SPLIT_TWIN_GROUPS": 2074}
    filled = v12.fill_gates(v12.GATES_PARAGRAPH, numbers)
    assert (
        "[[" not in filled
        and "reports 0 such pairs among the images stored in this release" in filled
    )
    assert "V1 pixels stay in the earlier shards and are not re-scanned" in filled
    assert "checks the 2074 V1/V2 stem-linked twin pairs and finds none split apart" in filled
    assert "`provenance/GATES.json`" in filled


def test_the_v1_2_0_card_keeps_the_v1_1_1_history():
    from marinedata import card_v12_text as v12
    from marinedata.condition_masks_docs import MASK_NOTE

    assert v12.mask_note() == MASK_NOTE
    log = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heads = dict(re.findall(r"^## \[(1\.1\.[01])\] - (\d{4}-\d{2}-\d{2})$", log, re.M))
    assert (heads["1.1.0"], heads["1.1.1"]) == (v12.V110_DATE, v12.V111_DATE)  # publication dates
    assert v12.V111_CHANGELOG and "condition-masks" in " ".join(v12.V111_CHANGELOG)


def _version_rows(card: str) -> list[list[str]]:
    block = card.split("## Versions and changelog", 1)[1].split("\n## ", 1)[0]
    return [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in block.splitlines()
        if line.startswith("| v")
    ]


def test_the_v1_2_0_card_lists_four_versions_the_mask_note_and_the_label_section(
    registry, build, tmp_path, monkeypatch
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "v2")
    rows = _version_rows(card)
    assert [r[0] for r in rows] == ["v1.0", "v1.1.0", "v1.1.1", "v1.2.0"]
    assert [r[1] for r in rows] == [txt.V1_DATE, v12.V110_DATE, v12.V111_DATE, v12.PUBLISHED]
    assert "`condition-masks`" in rows[2][2] and rows[3][2] == v12.VERSION_NOTE
    note = v12.mask_note()
    assert card.count(note) == 1 and note.startswith("**Condition masks (v1.1.1).**")
    order = (
        "\n## Dataset structure", note, "\n## Coralscapes V2",
        "\n### A byte-identical test pair", "\n## Label space",
        "\n## Using labels across sources", "\n## Splits",
    )  # fmt: skip
    positions = [card.index(h) for h in order]
    assert positions == sorted(positions)
    assert D7 in card and "<<" not in card and ">>" not in card
    block = card.split("## Using labels across sources", 1)[1].split("\n## ", 1)[0]
    g = v12.REGISTRY
    for needle in (
        f"taxonomy release {v12.TAXONOMY}", *v12.NEW_NODES, f"{g['units']} units",
        f"{g['with_crosswalk']} have a crosswalk", f"{g['no_crosswalk']} are label-less",
        f"{g['labels']:,} labels", f"{g['edges']:,} edges", f"{g['crosswalks']} crosswalks",
        f"{g['exact']:,} exact", f"{g['coarsened']} coarsened", f"{g['approximate']} approximate",
        f"{g['unmappable']} unmappable", "`coral-binary`", "`benthic-coarse`", "`scene`",
        v12.LABEL_MATCHING_URL, v12.LABEL_MATCHING_JSON_URL,
    ):  # fmt: skip
        assert needle in block, needle
    assert_public_safe(block, "open", where="label section")


def test_the_v1_2_0_summary_counts_the_configs_of_v1_1_0_and_v1_1_1(
    registry, build, tmp_path, monkeypatch
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "v2")
    keeps = re.search(
        r"keeps the (\d+) configurations that v1\.1\.0 "
        r"(?:\((\d+)\) and v1\.1\.1 \(`condition-masks`\) )?added to the frozen v1\.0 release",
        card,
    )
    row = re.search(
        r"\| Configurations \| (\d+) Parquet configs, streamable; image bytes stored once \|", card
    )
    shown = [ln[:300] for ln in card.splitlines() if "Configurations |" in ln or "keeps the" in ln]
    assert keeps and row, shown
    n_added = int(keeps.group(1))  # the configs that v1.1.0 (and v1.1.1) added to the v1.0 ones
    assert keeps.group(2) in (None, str(n_added - 1))  # `condition-masks` is v1.1.1
    assert int(row.group(1)) == len(V1_CONFIGS["open"]) + n_added  # the whole release
    assert keeps.group(2) is None or "v1.1.0 added" not in card
    versions = _section(card, "Versions and changelog")
    assert keeps.start() > card.index("## Versions and changelog")  # the sentence moved there
    assert versions.startswith(f"{PRETTY} v{v12.VERSION} keeps the {n_added} configurations")


PRETTY = "Reef Support Marine Data"
FIELDS = [
    "Images", "Labels", "Sources", "Label space", "Configurations", "Format", "Current version",
    "Code",
]  # fmt: skip


def _section(card: str, heading: str) -> str:
    return card.split(f"## {heading}\n\n", 1)[1].split("\n## ", 1)[0]


def test_the_v1_2_0_summary_describes_the_dataset_and_leaves_the_history_to_the_versions(
    registry, build, tmp_path, monkeypatch
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "v2")
    summary, versions = _section(card, "Dataset summary"), _section(card, "Versions and changelog")
    rows = [ln for ln in summary.splitlines() if ln.startswith("| ")]
    assert summary.startswith(f"{PRETTY} is a collection of ")
    assert [r.split(" | ")[0][2:] for r in rows[1:]] == FIELDS
    assert not re.search(r"keeps the|v1\.1\.0 \(\d+\)|frozen v1\.0 release", summary)
    for gone in ("Images (`metadata` rows)", "Size of the new configs", "from v1.0,"):
        assert gone not in summary
    splits = _section(card, "Splits").split("| `metadata` |", 1)[1].splitlines()[0]
    images = splits.rsplit("|", 2)[1].strip()  # the `metadata` rows: one per image of the release
    assert f"collection of {images} underwater images from " in summary
    assert f"| Images | {images} (" in summary and "(including the 95-class" in summary
    assert "[Versions and changelog](#versions-and-changelog)" in summary
    first = versions.split("\n\n", 1)[0]  # the history moved here, whole and first
    assert first.startswith(f"{PRETTY} v{v12.VERSION} keeps the ") and first.endswith(D7 + ".")
    assert versions.split("\n\n", 2)[1].startswith("| Version | Date | Notes |")
    assert "## Supported tasks" in card.split(summary, 1)[1][:30]


def test_the_licence_families_are_read_from_the_licence_rows():
    from marinedata import card_v12_top as top

    ids = ("US-GOV-PD", "MIT", "CC-BY-3.0", "CC-BY-4.0", "Apache-2.0", "CC-BY-4.0")
    rows = [{"licence": i, "licence_class": "open"} for i in ids]
    names = ["Apache-2.0", "CC BY 4.0", "CC BY 3.0", "MIT", "US Government public domain"]
    assert top.licence_families(rows) == names
    assert top.sources_fact(rows, 16) == f"16, all permitting commercial use ({', '.join(names)})"
    other = [*rows, {"licence": "BSD-3-Clause", "licence_class": "open"}]
    assert top.licence_families(other) == [*names, "BSD-3-Clause"]  # an unknown id is listed last
    nc = [*rows, {"licence": "CC-BY-NC-4.0", "licence_class": "restricted-nc"}]
    assert "permitting" not in top.sources_fact(nc, 17) and "CC BY-NC 4.0" in top.sources_fact(
        nc, 17
    )
    assert top.sources_fact([], 0) == "0"


def test_the_readme_opens_its_published_datasets_with_the_card_summary():
    from marinedata import card_v12_top as top

    readme = (ROOT / "README.md").read_text()
    rows = re.findall(r"^\| `([a-z-]+)` \| ([\d,]+) \| v1\.\d", readme, re.M)
    n = {name: int(count.replace(",", "")) for name, count in rows}
    release = json.loads((ROOT / "releases/marine-data/v1.2.0/open/RELEASE.json").read_text())
    opening = top.summary_paragraphs(
        name=PRETTY, images=n["metadata"], sources=len(release["sources"]), nodes=n["taxonomy"],
        configs=len(n), version=v12.VERSION, published=v12.PUBLISHED, closing=top.CLOSING_README,
    )  # fmt: skip
    section = readme.split("## Published datasets\n\n", 1)[1]
    assert section.startswith(f"{opening}\n\n### Release history\n\nRelease v1.0 (October 2026) is")
    assert f"{n['metadata']:,} underwater images from {len(release['sources'])} sources" in readme
    assert "58,803 rows in the four v1.0 configs" not in readme


def test_the_v1_2_0_row_shows_the_publish_date_without_a_build_date():
    rows = v12.version_rows(txt.V1_DATE, "v1.0 note", None)
    assert [r[0] for r in rows] == ["v1.0", "v1.1.0", "v1.1.1", "v1.2.0"]
    assert rows[-1][1] == v12.PUBLISHED and not any("<<" in c for r in rows for c in r)


def test_label_matching_figures_are_the_verifier_result():
    data = json.loads(MATCHING_JSON.read_text())
    s, fid = data["summary"], data["summary"]["edges_by_fidelity"]
    assert (data["release"], data["taxonomy_version"]) == (v12.VERSION, v12.TAXONOMY)
    expected = {
        "units": s["units"], "with_crosswalk": s["by_status"]["pass"],
        "no_crosswalk": s["by_status"]["no-crosswalk"], "labels": s["labels_verified"],
        "edges": s["crosswalk_edges"], "crosswalks": s["crosswalks"], "exact": fid["exact"],
        "coarsened": fid["coarsened"], "approximate": fid["approximate"],
        "unmappable": fid["ignored"], "lossy_without_note": sum(s["lossy_without_note"].values()),
    }  # fmt: skip
    assert expected == v12.REGISTRY
    assert fid["narrower"] == 0 and sum(fid.values()) == s["crosswalk_edges"]
    assert [s[k] for k in ("unresolved", "ambiguous", "missing", "duplicate")] == [0, 0, 0, 0]
    assert len(data["unmappable"]) == fid["ignored"]
    assert all(u["note"].strip() for u in data["unmappable"])
    assert set(v12.NEW_NODES) <= set(data["new_nodes"]["nodes"])


def test_the_byte_identical_test_pair_is_stated_where_readers_look():
    import pyarrow.parquet as pq

    rows = pq.read_table(MANIFEST).to_pylist()
    by_stem = {r["stem"]: r for r in rows}
    first, second = (by_stem[s] for s in v12.TEST_PAIR)
    assert first["sha256"] == second["sha256"]
    assert first["upstream_split"] == second["upstream_split"] == "test"
    assert len(rows) == v12.BENCHMARK == 643
    assert len({r["sha256"] for r in rows}) == v12.BENCHMARK - 1 == 642
    entry = (ROOT / "registry" / "sources" / "coral-benthic.yaml").read_text()
    entry = entry.split("- id: coralscapes-v2", 1)[1].split("\n  - id: ", 1)[0]
    texts = {
        "card": v12.section(V2),
        "registry note": entry,
        "CHANGELOG": (ROOT / "CHANGELOG.md").read_text(),
    }
    for where, text in texts.items():
        flat = " ".join(text.split())
        assert all(stem in flat for stem in v12.TEST_PAIR), where
        assert f"{v12.BENCHMARK} rows" in flat, where
        assert f"{v12.BENCHMARK - 1} unique image digests" in flat, where


def test_d7_wording_is_verbatim_in_the_card_and_the_changelog():
    assert v12.SUPERSEDES == D7 + "."
    assert D7 in v12.section(V2) and D7 in (ROOT / "CHANGELOG.md").read_text()


GATES = {
    "dedup": {"passed": True, "counts": {"near_pairs_linked_twins": 2074}, "failures": []},
    "splits": {
        "passed": True,
        "counts": {"stem_link_twins": 2074, "stem_link_pairs_apart": 0},
        "failures": [],
    },
    "masks": {"passed": True, "counts": {}, "failures": []},
    "privacy": {"passed": False, "counts": {}, "failures": [], "skipped": "no YuNet model"},
    "summary": {"passed": 3, "failed": 0, "skipped": 1},
}
NUMBERS = {"V120_GATES_RUN": 3, "V120_DEDUP_LINKED": 2074, "V120_SPLIT_TWIN_GROUPS": 2074}


def _stage(registry, build, tmp_path, monkeypatch, *extra: str) -> tuple[int, str]:
    """The card stage CLI over the synthetic export, its one source posing as Coralscapes V2."""
    from marinedata import hf_card

    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    layout, _ = _layout(registry, build, flavour="open", shard_target_bytes=2000)
    out, summary, cff = tmp_path / "export", tmp_path / "summary.json", tmp_path / "CITATION.cff"
    summary.write_text(json.dumps(export_v11(layout, out)))
    cff.write_text(yaml.safe_dump(CFF))
    argv = ["--release-dir", str(build[0]), "--summary", str(summary), "--out", str(out)]
    code = hf_card.main([*argv, "--citation", str(cff), "--flavour", "open", *extra])
    return code, (out / "README.md").read_text()


def test_gate_numbers_accept_a_zero_linked_twin_count():
    zero = {"passed": True, "counts": {"near_pairs_linked_twins": 0}, "failures": []}
    assert v12.gate_numbers({**GATES, "dedup": zero}) == {**NUMBERS, "V120_DEDUP_LINKED": 0}
    for bad in (True, None, "0", 0.0):  # present but not a count
        gates = {**GATES, "dedup": {**zero, "counts": {"near_pairs_linked_twins": bad}}}
        with pytest.raises(ValueError, match="missing or not a count"):
            v12.gate_numbers(gates)


def test_gate_numbers_come_from_gates_json_and_refuse_a_gate_that_did_not_pass():
    assert v12.gate_numbers(GATES) == NUMBERS
    skipped = {**GATES, "splits": {**GATES["splits"], "passed": False, "skipped": "no map"}}
    with pytest.raises(ValueError, match="splits gate is skipped"):
        v12.gate_numbers(skipped)
    apart_counts = {"stem_link_twins": 5, "stem_link_pairs_apart": 2}
    with pytest.raises(ValueError, match="2 twin pair"):
        v12.gate_numbers({**GATES, "splits": {**GATES["splits"], "counts": apart_counts}})
    with pytest.raises(ValueError, match="dedup gate is missing"):
        v12.gate_numbers({k: v for k, v in GATES.items() if k != "dedup"})
    with pytest.raises(ValueError, match="near_pairs_linked_twins"):
        v12.gate_numbers({**GATES, "dedup": {"passed": True, "counts": {}}})


@pytest.mark.parametrize("linked", [2074, 0])  # 0: an additive export scans no linked pair
def test_card_stage_fills_the_gate_slots_from_gates_json(
    registry, build, tmp_path, monkeypatch, capsys, linked
):
    dedup = {"passed": True, "counts": {"near_pairs_linked_twins": linked}, "failures": []}
    gates = tmp_path / "GATES.json"
    gates.write_text(json.dumps({**GATES, "dedup": dedup}))
    code, card = _stage(registry, build, tmp_path, monkeypatch, "--gates-json", str(gates))
    assert code == 0 and "[[" not in card and "<<" not in card
    assert "**v1.2.0 gates.** 3 gates ran" in card
    assert "checks the 2074 V1/V2 stem-linked twin pairs" in card
    assert f"reports {linked} such pairs among the images stored in this release" in card
    assert "WARNING" not in capsys.readouterr().err


def test_card_stage_without_gates_json_keeps_the_slots_and_warns(
    registry, build, tmp_path, monkeypatch, capsys
):
    code, card = _stage(registry, build, tmp_path, monkeypatch)
    err = capsys.readouterr().err
    assert code == 0 and v12.pending_slots(card) == v12.GATE_SLOTS
    assert "WARNING" in err and all(f"[[{slot}]]" in err for slot in v12.GATE_SLOTS)


def test_gates_json_for_a_card_without_slots_is_ignored_with_a_warning(
    registry, build, tmp_path, capsys
):
    layout, _ = _layout(registry, build, flavour="open", shard_target_bytes=2000)
    export_v11(layout, tmp_path)
    write_card_files(
        tmp_path, registry, flavour="open", repo_id="reefsupport/marine-data", cff=CFF,
        released="2026-10-29", gates=GATES,
    )  # fmt: skip
    assert "--gates-json ignored" in capsys.readouterr().err
    assert "[[" not in (tmp_path / "README.md").read_text()


# The frozen v1.2.0 split map: every number of the SPLIT_MAP paragraph is read from the maps.

MAP_V120 = ROOT / "releases" / "marine-data" / "v1.2.0" / "SPLIT_MAP.json"
MAP_V110 = ROOT / "releases" / "marine-data" / "v1.1.0" / "SPLIT_MAP.json"
HUB_SPLIT = {"train": "train", "val": "validation", "test": "test"}  # map name -> Hub and pin name


@pytest.fixture(scope="module")
def maps() -> tuple[dict, dict]:
    """``(v1.2.0 map, v1.1.0 map)`` as committed."""
    return json.loads(MAP_V120.read_text()), json.loads(MAP_V110.read_text())


def _flat(text: str) -> str:
    return " ".join(text.split())


def _moved(new: dict, old: dict) -> dict[str, tuple[str, str]]:
    """Inherited groups in both maps with a different split: ``group -> (v1.1.0, v1.2.0)``."""
    a, b = new["assignments"], old["assignments"]
    return {g: (b[g], a[g]) for g in b if g in a and a[g] != b[g]}


def test_split_map_text_states_the_inheritance_and_the_moved_groups(maps):
    new, old = maps
    inherits, moved = new["inherits"], _moved(new, old)
    by_source = Counter(g.split("/", 1)[0] for g in moved)
    overridden = new["pins"]["v1_continuity_overridden"]
    assert inherits["release"] == old["release"] == "v1.1.0"
    assert inherits["groups"] == len(old["assignments"])
    assert set(old["assignments"]) <= set(new["assignments"])  # every inherited group carries over
    assert set(by_source) == {"roboflow-bleaching", "uiis10k"}  # a third source needs new words
    (uiis,) = (g for g in moved if g.startswith("uiis10k/"))
    assert moved[uiis] == ("val", "train")  # validation in `metadata`; train in every table now
    assert (overridden["images"], overridden["groups"]) == (0, 0)
    flat = _flat(v12.SPLIT_MAP)
    for needle in (
        f"inherits the {inherits['release']} map",
        f"its {inherits['groups']:,} groups carry over, {inherits['groups_kept']:,} of them kept "
        "verbatim",
        f"changed the split of {len(moved)} inherited groups and of no other: "
        f"{by_source['roboflow-bleaching']} are near-duplicate groups from the Roboflow "
        "bleaching exports",
        f"and {by_source['uiis10k']} is a UIIS10K image",
        "v1.2.0 puts it in train in every table",
        f"(`pins.v1_continuity_overridden`: {overridden['images']} images, "
        f"{overridden['groups']} groups)",
    ):
        assert needle in flat


def test_split_map_text_states_the_v2_assignments_the_pins_and_the_stem_links(maps):
    import pyarrow.parquet as pq

    new, old = maps
    v2 = Counter(
        HUB_SPLIT[s] for g, s in new["assignments"].items() if g.startswith("coralscapes-v2/")
    )
    assigned, withheld = sum(v2.values()), len(new["withheld"])
    pins, links = new["pins"]["benchmarks"], new["inherits"]["stem_links"]
    manifest = pq.read_table(MANIFEST).to_pylist()
    digests = {
        s: len({r["sha256"] for r in manifest if r["upstream_split"] == s})
        for s in ("test", "validation")
    }
    assert assigned + withheld == v12.V2_FRAMES
    assert (v2["train"], v2["validation"]) == (v12.V2_SPLITS["train"], v12.V2_SPLITS["validation"])
    assert v2["test"] + withheld == v12.V2_SPLITS["test"]  # the official test split less the one
    assert pins["coralscapes-v2"] == digests  # counted by digest: the byte-identical pair is one
    assert pins["coralscapes"] == old["pins"]["benchmarks"]["coralscapes"]  # V1 pins did not move
    assert (links["twins"], links["target_only"]) == (v12.TWINS, v12.V1_ONLY)
    assert links["declaring_only"] == sum(v12.V2_NEW.values())
    v2_pins, v1_pins = pins["coralscapes-v2"], pins["coralscapes"]
    flat = _flat(v12.SPLIT_MAP)
    for needle in (
        f"{assigned:,} of the {v12.V2_FRAMES:,} V2 frames are assigned: {v2['train']:,} to "
        f"train, {v2['validation']} to validation and {v2['test']} to test",
        f"the official V2 split has {v12.V2_SPLITS['test']} test frames",
        f"hold {v2_pins['test']} test and {v2_pins['validation']} validation images, counted "
        "by digest",
        f"the V1 pins are unchanged at {v1_pins['test']} test and {v1_pins['validation']} "
        "validation images",
        f"{links['twins']:,} twins, {links['declaring_only']} V2-only frames, "
        f"{links['target_only']} V1-only frame, {links['ambiguous_stems']} ambiguous stems and "
        f"{links['conflicts']['count']} conflicts",
    ):
        assert needle in flat


def test_split_map_text_states_the_withheld_group_the_exclusions_and_the_missing_dedup_run(maps):
    new, _ = maps
    band, phash, near = new["band_excluded"], new["phash_excluded"], new["near_dup"]
    confirm = near["union_confirm"]
    high = near["never_eval_exclude_max_hamming"]
    distances = {hit[0] for hit in band.values()}
    assert len(new["withheld"]) == 1 and not set(band) & set(phash)
    # "8 or less", not "5 to 8": a pair at 4 or less that fails the confirm joins the band too
    assert max(distances) <= high and min(distances) <= near["union_max_hamming"]
    assert near["dedup_groups"] == "disabled by --no-near-dup"
    assert "phash_union" in near and "union_confirm" in near  # the unions ran without a dedup run
    flat = _flat(v12.SPLIT_MAP)
    for needle in (
        "One group is withheld",
        f"{len(band)} images that sit in the prior-release near-duplicate band (dHash distance "
        f"{high} or less)",
        f"{len(phash)} images that each collide on the DCT pHash-64 with an image of another split",
        f"these {len(band) + len(phash)} images ship in no split of any config",
        f"A dHash candidate pair (distance {near['union_max_hamming']} or less)",
        f"at most {confirm['phash256_max']} of 256 bits apart",
        f"at least {confirm['min_entropy_bits']:.0f} bits of grey entropy",
        "joins the prior-release band",
        f"(`near_dup.dedup_groups` reads `{near['dedup_groups']}`)",
    ):
        assert needle in flat


def test_split_map_text_names_no_digest_group_or_frame(maps):
    new, old = maps
    flat = _flat(v12.SPLIT_MAP)
    names = [g.split("/", 1)[1] for g in (*new["withheld"], *_moved(new, old))]
    names += [n.removesuffix("_leftImg8bit") for n in names]
    assert not re.search(r"[0-9a-f]{64}", flat) and "sha:" not in flat
    assert names and all(n not in flat for n in names)


def test_the_open_card_renders_the_split_map_after_the_split_text_and_the_nc_card_does_not(
    registry, build, tmp_path, monkeypatch, maps
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "open")
    nc_card, _ = _render(registry, build, tmp_path / "nc", "nc")
    heads = ("### Splits: V1 and V2 twins", v12.SPLIT_MAP, "### A byte-identical test pair")
    first, mid, last = (card.index(h) for h in heads)
    assert first < mid < last and card.count(v12.SPLIT_MAP) == 1
    assert not any(part in nc_card for part in v12.SPLIT_MAP.split("\n\n"))
    assert "disabled by --no-near-dup" not in nc_card
    pinned = maps[0]["pins"]["benchmarks"]["coralscapes-v2"]["test"]  # `_splits` spells it 472
    assert f"{pinned} of the 473 test frames" in card and "test (473)" not in card


def _ints(node) -> set[int]:
    """Every integer anywhere in a parsed JSON document."""
    if isinstance(node, bool):
        return set()
    if isinstance(node, int):
        return {node}
    children = node.values() if isinstance(node, dict) else node if isinstance(node, list) else []
    return {i for child in children for i in _ints(child)}


def test_v2_split_sentence_is_pinned_to_the_frozen_split_map_and_gates():
    release = ROOT / "releases" / "marine-data" / "v1.2.0"
    split_map = json.loads((release / "SPLIT_MAP.json").read_text(encoding="utf-8"))
    gates = json.loads((release / "open" / "GATES.json").read_text(encoding="utf-8"))
    counts = gates["splits"]["counts"]
    assigned = Counter(
        {"val": "validation"}.get(s, s)
        for g, s in split_map["assignments"].items()
        if g.startswith("coralscapes-v2/")
    )
    assert dict(assigned) == v12.V2_ASSIGNED
    assert v12.V2_ASSIGNED["test"] + len(split_map["withheld"]) == v12.V2_SPLITS["test"]
    shipped = counts["ratio_pinned_sources"]["semseg"]["coralscapes-v2"]
    assert shipped["train"] == v12.V2_SHIPPED["train"]
    assert v12.V2_ASSIGNED["train"] - shipped["train"] == v12.NOT_SHIPPED
    assert v12.V2_SHIPPED["validation"] == v12.V2_ASSIGNED["validation"]
    assert v12.V2_SHIPPED["test"] == v12.V2_ASSIGNED["test"]
    assert counts["ratio_left_out_images"]["pinned"] == v12.V2_IMAGES
    for split in (v12.V2_SPLITS, v12.V2_ASSIGNED, v12.V2_SHIPPED):
        triple = f"{split['train']:,} / {split['validation']} / {split['test']}"
        assert triple in v12.V2_SPLIT_SENTENCE
    assert f"{v12.V2_IMAGES:,} images" in v12.V2_SPLIT_SENTENCE


def test_card_states_release_wide_counts_and_the_scope_of_the_gates(
    registry, build, tmp_path, monkeypatch
):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "v2")
    assert v12.V2_SPLIT_SENTENCE in card
    assert card.count(v12.GATES_SCOPE) == 2  # after the dedup and after the privacy sentence
    row = re.search(r"\| Sources \| (\d+)\b", card)  # the sources of the release
    in_config = _section(card, "Splits").split("| `sources` |", 1)[1].splitlines()[0]
    rows = int(in_config.rsplit("|", 2)[1].replace(",", "").strip())  # rows of the `sources` config
    assert row and int(row.group(1)) - rows == len(v12.SOURCES_WITHOUT_ROW)
    release = ROOT / "releases" / "marine-data" / "v1.2.0"
    lock = json.loads((release / "open" / "RELEASE.json").read_text(encoding="utf-8"))
    assert set(v12.SOURCES_WITHOUT_ROW) <= {s["id"] for s in lock["sources"]}
    gates = json.loads((release / "open" / "GATES.json").read_text(encoding="utf-8"))
    assert v12.IMAGES_ADDED in _ints(gates)
    assert f"{v12.IMAGES_EARLIER:,} images" in txt.PERSONAL_OPEN
    gone = (
        "benthic cover and sites",
        "`sites`, `label-map` and `sources`",
        "`benthic-cover` and `sites`",
    )
    assert not [p for p in gone if p in card]  # no mention of a `sites` config in the text


def test_row_shares_come_from_the_cards_own_split_counts():
    def config(train: int, validation: int, test: int) -> dict:
        counts = {"train": train, "validation": validation, "test": test}
        return {"splits": {s: {"num_examples": n} for s, n in counts.items()}}

    text = txt.splits_open_for({"semseg": config(3, 1, 1), "points": config(2, 1, 1)})
    assert (
        "`points` rows are 50.0 / 25.0 / 25.0 % (its 11,356 distinct images 67.1 / 12.8 / 20.2 %) "
        "and `semseg` rows 60.0 / 20.0 / 20.0 %. " in text
    )
    assert txt.SPLITS_ROW_SHARES in txt.SPLITS_OPEN and txt.SPLITS_ROW_SHARES not in text
    assert "74.1" not in text and "65.2" not in text


def test_the_v1_2_0_card_pins_the_corrected_statements(registry, build, tmp_path, monkeypatch):
    monkeypatch.setattr(v12, "SOURCE_ID", "seaview-survey-imagery")
    card, _ = _render(registry, build, tmp_path / "v2")
    pinned = (
        v12.SOURCES_TASK_NOTE,
        v12.SOURCES_NOTE,
        v12.SHIPPED_SENTENCE,
        v12.UNCHANGED_CLAUSE,
        v12.VERSION_NOTE,
        v12.GATES_SCOPE.strip(),
        v12.SPLIT_MAP_PATH,
        "No other non-commercial source is part of v1.2.0",
        'the v1.2.0 figures are under "Split map of v1.2.0" above',
        "is listed in the sources table.",
    )
    problems = [text for text in pinned if text not in card]
    problems += [p for p in ("node(s)", "with its share", "{version}") if p in card]
    if card.count(v12.GATES_SCOPE.strip()) < 2:  # the gates section and the personal-data section
        problems.append("the gates sentence is not in both sections")
    if not re.search(r"\d+ nodes? (?:is|are) reached by a single source\.", card):
        problems.append("single-source note")
    assert not problems, [p[:80] for p in problems]


def test_licence_rows_name_the_citation_pointer_and_the_seaview_licences() -> None:
    base = "Reef Support annotations on XL Catlin Seaview imagery"
    sources = [
        {"source_id": "coralscapes-v2", "attribution": "Coralscapes V2 (Apache-2.0)"},
        {
            "source_id": v12.SEAVIEW_LABELS,
            "attribution": base,
            "licence": "CC-BY-3.0",
            "ann_licence": "CC-BY-3.0",
        },
    ]
    extra = [
        {"source_id": "reef-support-bleaching", "ann_licence": "CC-BY-4.0", "attribution": "x"}
    ]
    out = {r["source_id"]: r for r in v12.licence_rows(sources, extra)}
    assert (
        out["coralscapes-v2"]["attribution"]
        == 'Coralscapes V2 (Apache-2.0); see "Citing Coralscapes V2"'
    )
    assert out[v12.SEAVIEW_LABELS]["ann_licence"] == "CC-BY-4.0"
    assert (
        out[v12.SEAVIEW_LABELS]["attribution"]
        == base + ": CC-BY-3.0 (images); CC-BY-4.0 (masks, labels)"
    )
    assert out["reef-support-bleaching"] == extra[0]
    assert (
        sources[0]["attribution"] == "Coralscapes V2 (Apache-2.0)"
    )  # the input rows are not mutated
