"""v1.2.0 card text: the Coralscapes V2 section, its numbers and its hook in the v1.1 renderer."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml
from test_hf_export_v11 import _layout
from test_hf_export_v11 import build as _build
from test_hf_export_v11 import registry as _registry

from marinedata import card_v11_text as txt
from marinedata import card_v12_text as v12
from marinedata.hf_card_v11 import card_version, write_card_files
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


def _render(registry, build, root: Path) -> tuple[str, dict]:
    layout, _ = _layout(registry, build, flavour="open", shard_target_bytes=2000)
    export_v11(layout, root)
    write_card_files(
        root, registry, flavour="open", repo_id="reefsupport/marine-data", cff=CFF,
        released="2026-10-29",
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
    ):  # fmt: skip
        assert needle in text, needle
    assert "<<" not in text and ">>" not in text
    assert "images in this release" not in v12.section([{"source_id": v12.SOURCE_ID}])
    assert_public_safe(text, "open", where="v1.2.0 section")


def test_limitations_replace_only_the_coralscapes_bullet():
    out = v12.limitations(txt.LIMITATIONS)
    assert out[0] == v12.LIMITATION and out[1:] == txt.LIMITATIONS[1:]
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
    assert "[[" not in filled and "2074 V1/V2 twin pairs" in filled
    with pytest.raises(ValueError, match="missing"):
        v12.fill_gates(v12.GATES_PARAGRAPH, {"V120_GATES_RUN": 12})
    with pytest.raises(ValueError, match="unknown"):
        v12.fill_gates(v12.GATES_PARAGRAPH, {**numbers, "OTHER": 1})
    with pytest.raises(ValueError, match="no \\[\\[V120_GATES_RUN\\]\\] slot"):
        v12.fill_gates("no slots here", numbers)


def test_the_v1_2_0_card_keeps_the_v1_1_1_history():
    from marinedata import card_v12_text as v12
    from marinedata.condition_masks_docs import MASK_NOTE

    assert v12.mask_note() == MASK_NOTE
    assert (v12.V110_DATE, v12.V111_DATE) == ("2026-10-17", "2026-10-24")
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
    assert [r[0] for r in rows] == ["v1.0.0", "v1.1.0", "v1.1.1", "v1.2.0"]
    assert [r[1] for r in rows] == [txt.V1_DATE, v12.V110_DATE, v12.V111_DATE, "2026-10-29"]
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
        r"keeps the (\d+) configurations that (.+?) added to the frozen v1\.0 release", card
    )
    row = re.search(r"\| Configurations \| (\d+): ", card)
    assert keeps and row and keeps.group(1) == row.group(1)  # the count is the export's own
    assert keeps.group(2) == "v1.1.0 and v1.1.1"
    assert "v1.1.0 added" not in card


def test_the_v1_2_0_row_falls_back_to_the_month_without_a_build_date():
    rows = v12.version_rows(txt.V1_DATE, "v1.0 note", None)
    assert [r[0] for r in rows] == ["v1.0.0", "v1.1.0", "v1.1.1", "v1.2.0"]
    assert rows[-1][1] == "2026-10" and not any("<<" in c for r in rows for c in r)


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


def test_card_stage_fills_the_gate_slots_from_gates_json(
    registry, build, tmp_path, monkeypatch, capsys
):
    gates = tmp_path / "GATES.json"
    gates.write_text(json.dumps(GATES))
    code, card = _stage(registry, build, tmp_path, monkeypatch, "--gates-json", str(gates))
    assert code == 0 and "[[" not in card and "<<" not in card
    assert "**v1.2.0 gates.** 3 gates ran" in card and "2074 V1/V2 twin pairs" in card
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
