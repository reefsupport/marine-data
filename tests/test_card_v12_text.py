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
