"""WP-U2: the crosswalk contract — match_type alias, narrower, resolve, node_facts, native ids."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from marinedata import taxonomy as tx
from marinedata.annotation_schema import MATCH_TYPES
from marinedata.registry import Registry
from marinedata.sample import LabelValue
from marinedata.schema import Axis, Crosswalk, CrosswalkEdge, Fidelity


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


def _edge(label: str, **kw) -> CrosswalkEdge:
    return CrosswalkEdge.model_validate({"source_label": label, **kw})


def test_match_type_is_an_input_alias_for_fidelity():
    assert _edge("a", targets={"taxon": "HC"}, match_type="broader").fidelity is Fidelity.COARSENED
    assert (
        _edge("a", targets={"taxon": "HC"}, match_type="related").fidelity is Fidelity.APPROXIMATE
    )
    assert _edge("a", match_type="unmapped").fidelity is Fidelity.UNMAPPABLE
    assert _edge("a", targets={"taxon": "HC"}, fidelity="coarsened").match_type == "broader"
    with pytest.raises(ValidationError, match="not both"):
        _edge("a", targets={"taxon": "HC"}, match_type="exact", fidelity="exact")


def test_every_fidelity_maps_onto_an_annotation_match_type():
    assert {f.match_type for f in Fidelity} == set(MATCH_TYPES)


def test_narrower_needs_targets_and_is_never_reliable():
    assert _edge("a", targets={"taxon": "HC"}, match_type="narrower").fidelity is Fidelity.NARROWER
    with pytest.raises(ValidationError, match="no targets"):
        _edge("a", match_type="narrower")
    assert not LabelValue("HC", Fidelity.NARROWER).is_reliable()
    assert LabelValue("HC", Fidelity.COARSENED).is_reliable()


def test_resolve_returns_node_match_type_aphia_and_l2_code(reg):
    cw = Crosswalk(
        id="t",
        source_schema="s",
        target_schema="rs-benthic-v1",
        edges=(
            _edge("hard", targets={"taxon": "HC"}),
            _edge("orb", targets={"taxon": "HC_ORBICELLA"}, match_type="exact"),
            _edge("cnid", targets={"taxon": "CNIDARIA"}, match_type="broader"),
            _edge("sand?", targets={"taxon": "HC"}, match_type="narrower"),
            _edge("junk", match_type="unmapped", note="no equivalent"),
        ),
    )
    target = reg.label_schema("rs-benthic-v1")
    l2 = set(reg.task("benthic-l2").classes)
    assert cw.resolve("hard", target=target, l2_codes=l2) == ("HC", "exact", 1363, "HC")
    assert cw.resolve("orb", target=target, l2_codes=l2) == ("HC_ORBICELLA", "exact", 758259, "HC")
    assert cw.resolve("cnid", target=target, l2_codes=l2) == ("CNIDARIA", "broader", 1267, None)
    assert cw.resolve("sand?", target=target, l2_codes=l2)[:2] == ("HC", "narrower")
    assert cw.resolve("junk", target=target) == (None, "unmapped", None, None)
    assert cw.resolve("never seen") == (None, "unmapped", None, None)
    assert cw.resolve("hard") == ("HC", "exact", None, None)  # no target schema: facts stay null


def test_node_facts_follow_the_l2_task_vocabulary(reg):
    assert tx.node_facts(reg, "HC") == ("Order", 1363, "HC")
    assert tx.node_facts(reg, "HC_ORBICELLA").rs_benthic_code == "HC"
    assert tx.node_facts(reg, "CNIDARIA").rs_benthic_code is None
    assert tx.node_facts(reg, None) == (None, None, None)
    assert tx.node_facts(reg, "NOT_A_NODE") == (None, None, None)


def test_edges_match_by_native_id_first_then_name():
    cw = Crosswalk(
        id="t",
        source_schema="s",
        target_schema="rs-benthic-v1",
        edges=(
            _edge("Hard  coral", source_label_id="7", targets={"taxon": "HC"}),
            _edge("sponge", targets={"taxon": "SP"}),
        ),
    )
    assert cw.edge("Hard coral") is None  # the double-space failure mode, by name
    assert cw.edge("Hard coral", "7").targets[Axis.TAXON] == "HC"  # rescued by id
    assert cw.edge("sponge", "99").targets[Axis.TAXON] == "SP"  # unknown id falls back to name
    assert cw.resolve("Hard coral", label_id="7")[:2] == ("HC", "exact")


def test_vocab_label_native_id_column_is_read_and_optional(tmp_path):
    with_ids = tmp_path / "a.tsv"
    with_ids.write_text(
        "# source: s\n# crosswalk: t\n"
        "label\tlabel_native_id\tcount\tdescription\nHard coral\t7\t10\t\n"
    )
    head, rows = tx.read_vocab_rows(with_ids)
    assert head["source"] == "s" and rows[0]["label_native_id"] == "7" and rows[0]["count"] == "10"
    assert tx.read_vocab(with_ids)[1] == {"Hard coral": 10}
    legacy = tmp_path / "b.tsv"
    legacy.write_text("# source: s\nlabel\tcount\tdescription\nsponge\t3\t\n")
    assert tx.read_vocab_rows(legacy)[1][0]["label_native_id"] == ""
    assert tx.read_vocab(legacy)[1] == {"sponge": 3}


def test_audit_counts_narrower_as_mapped_and_matches_by_id():
    cw = Crosswalk(
        id="t",
        source_schema="s",
        target_schema="rs-benthic-v1",
        edges=(
            _edge("Hard  coral", source_label_id="7", targets={"taxon": "HC"}),
            _edge("fish", targets={"taxon": "OTHER_FAUNA"}, match_type="narrower"),
        ),
    )
    audit = tx.audit_crosswalk(
        "s",
        cw,
        {"HC", "OTHER_FAUNA"},
        {"Hard coral": 8, "fish": 2},
        weighted=True,
        label_ids={"Hard coral": "7"},
    )
    assert audit.coverage == 1.0 and not audit.silent_drops
