"""v1.2.0 card text: the Coralscapes V2 section, its numbers and its hook in the v1.1 renderer."""

from __future__ import annotations

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
