"""WP-U2: ``marinedata labels check`` — floors, n/a cases, missing crosswalks, specs-only ids."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from marinedata import labels_check as lc
from marinedata.cli import main
from marinedata.registry import Registry
from marinedata.schema import Crosswalk, CrosswalkEdge
from marinedata.taxonomy import VocabAudit, audit_crosswalk


@pytest.fixture(scope="module")
def reg() -> Registry:
    return Registry.load()


@pytest.fixture(scope="module")
def rows(reg) -> dict[str, lc.LabelRow]:
    return {r.source_id: r for r in lc.evaluate(reg, reg.root)}


def _audit(mapped: int, total: int = 100, **kw) -> VocabAudit:
    counts = {f"c{i}": 1 for i in range(total)}
    return VocabAudit(
        source_id="s",
        crosswalk_id="x",
        weighted=True,
        counts=counts,
        mapped=tuple(list(counts)[:mapped]),
        unmappable={k: "reason" for k in list(counts)[mapped:]},
        **kw,
    )


# ── floors ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("mapped", "gated", "expected"),
    [
        (100, False, lc.PASS),
        (95, False, lc.PASS),
        (95, True, lc.PASS),
        (94, False, lc.WARN),  # soft band [0.90, 0.95) on an ungated source
        (90, False, lc.WARN),
        (94, True, lc.FAIL),  # a release includes it: 0.95 is the hard floor
        (89, False, lc.FAIL),  # below the 0.90 soft floor
    ],
)
def test_floors(mapped, gated, expected):
    assert lc.grade(_audit(mapped), gated=gated)[0] == expected


def test_coverage_exception_waives_the_floor_to_a_warn():
    status, why = lc.grade(_audit(90), gated=True, exception=True)
    assert status == lc.WARN and "coverage_exceptions" in why[0]
    assert lc.grade(_audit(80), gated=True, exception=True)[0] == lc.WARN


def test_contract_violations_fail_even_at_full_coverage():
    assert lc.grade(_audit(100, silent_drops=("lost",)))[0] == lc.FAIL
    assert lc.grade(_audit(100, dead_targets=("a->NOPE",)))[0] == lc.FAIL
    bare = _audit(99)
    bare = VocabAudit(**{**bare.__dict__, "unmappable": {"c99": ""}})
    assert lc.grade(bare)[0] == lc.FAIL  # `unmapped` must be explicit AND explained


def test_narrower_counts_as_mapped():
    edge = CrosswalkEdge(source_label="fish", targets={"taxon": "HC"}, match_type="narrower")
    cw = Crosswalk(id="t", source_schema="s", target_schema="rs-benthic-v1", edges=(edge,))
    audit = audit_crosswalk("s", cw, {"HC"}, {"fish": 4}, weighted=True)
    assert audit.coverage == 1.0 and lc.grade(audit)[0] == lc.PASS


# ── n/a, specs-only, missing crosswalk ───────────────────────────────────────────


def test_open_vocabulary_and_label_free_sources_are_na(rows):
    assert rows["coralvqa"].status == lc.NA and "open vocabulary" in rows["coralvqa"].reasons
    assert rows["gebco-2024"].status == lc.NA  # a bathymetry grid: no class labels


def test_specs_only_ids_are_checked(reg, rows):
    from marinedata.licence_class import SPEC_ALIASES

    # a spec whose id is a registry source's staged-id alias is that source, not a second one (R4)
    only_specs = lc.spec_ids(reg.root / "ingest-specs") - {s.id for s in reg.sources}
    assert only_specs >= set(SPEC_ALIASES.values()) and not set(SPEC_ALIASES.values()) & set(rows)
    only_specs -= set(SPEC_ALIASES.values())
    assert len(only_specs) >= 100
    assert only_specs <= set(rows)
    assert {rows[i].origin for i in only_specs} == {"spec"}


def test_specs_only_id_with_class_labels_and_no_crosswalk_is_missing(reg, tmp_path):
    (tmp_path / "zz-new-fish.yaml").write_text("id: zz-new-fish\nadapter: hf\n")
    cache = {"zz-new-fish": {"id": "zz-new-fish", "label_kinds": "boxes", "crosswalk_file": "none"}}
    kw = {"spec_dir": tmp_path, "audit_cache": cache, "only": ["zz-new-fish"]}
    (row,) = lc.evaluate(reg, reg.root, **kw)
    assert (row.origin, row.status, row.crosswalk_missing) == ("spec", lc.MISSING, True)
    assert not row.blocks  # report only until a release or --strict gates it
    (row,) = lc.evaluate(reg, reg.root, strict=True, **kw)
    assert row.blocks
    (row,) = lc.evaluate(reg, reg.root, release=["zz-new-fish"], **kw)
    assert row.blocks and row.origin == "spec"


def test_label_free_source_needs_no_crosswalk_but_labelled_one_still_fails(reg):
    """WP-R9: atlantis-synthetic-depth (depth maps, no class labels) is skipped with the reason
    ``label-free``; the same source with staged label rows, or a labelled source with no
    crosswalk, still blocks."""
    sid = "atlantis-synthetic-depth"
    cache = {"atlantis": {"id": "atlantis", "label_kinds": "none", "label_kinds_declared": "masks"}}
    kw = {"audit_cache": cache, "release": [sid], "only": [sid]}
    (row,) = lc.evaluate(reg, reg.root, **kw)
    assert row.status == lc.NA and not row.blocks and row.gated
    assert any(r.startswith("label-free") for r in row.reasons)
    (row,) = lc.evaluate(reg, reg.root, bucket_labels={"atlantis": 5}, **kw)
    assert row.status == lc.MISSING and row.blocks
    assert not lc._label_free(reg, "zz-not-in-registry", 0)
    labelled = next(s.id for s in reg.sources if s.id in set(lc.labelled_sources(reg)))
    assert not lc._label_free(reg, labelled, 0)


def test_declared_unknown_labels_are_listed_not_failed(reg, tmp_path):
    cache = {"zz-x": {"id": "zz-x", "label_kinds": "none", "label_kinds_declared": "unknown"}}
    (row,) = lc.evaluate(
        reg, reg.root, spec_dir=tmp_path, audit_cache=cache, strict=True, only=["zz-x"]
    )
    assert row.source_id == "zz-x" and row.status == lc.NA and not row.blocks


def test_a_crosswalk_without_a_vocab_is_measured_on_its_edges(rows):
    # mouss-detection used to be the example; WP-R3 gave it a vocab TSV (label-types basis)
    assert rows["mouss-detection"].basis == "label-types"
    row = rows["benthicnet-1m"]
    assert row.basis == "crosswalk-edges" and row.status == lc.PASS and row.mapped_pct == 100.0


def test_measured_sources_report_unmapped_labels(rows):
    assert rows["plc-beijbom2015"].status == lc.WARN and rows["plc-beijbom2015"].unmapped
    assert rows["coralscapes"].status == lc.PASS and rows["coralscapes"].mapped_pct == 100.0


def test_unknown_release_id_fails(reg):
    (row,) = lc.evaluate(reg, reg.root, release=["no-such-source"], only=["no-such-source"])
    assert row.status == lc.FAIL and row.blocks


def test_bucket_listing_marks_staged_ids_and_aborts_on_error(reg):
    class Pager:
        def __init__(self, pages):
            self.pages = pages

        def paginate(self, **_):
            return iter(self.pages)

    class Client:
        def __init__(self, pages):
            self.pages = pages

        def get_paginator(self, _):
            return Pager(self.pages)

    keys = ["sources/a/1/labels/x.png", "sources/a/1/labels/y.png", "sources/a/1/images/z.jpg"]
    counts = lc.bucket_label_counts(Client([{"Contents": [{"Key": k} for k in keys]}]))
    assert counts == {"a": 2}
    boom = Client([{"Contents": []}])
    boom.get_paginator = lambda _: (_ for _ in ()).throw(RuntimeError("403"))
    with pytest.raises(lc.LabelsCheckError, match="unverified"):
        lc.bucket_label_counts(boom)
    (row,) = lc.evaluate(reg, reg.root, bucket_labels={"zz-staged": 3}, only=["zz-staged"])
    assert row.source_id == "zz-staged" and row.status == lc.MISSING and row.ingested


# ── CLI ──────────────────────────────────────────────────────────────────────────


def test_cli_writes_json_and_tsv_and_exits_by_gate(tmp_path, capsys):
    j, t = tmp_path / "r.json", tmp_path / "r.tsv"
    assert main(["labels", "check", "--json", str(j), "--tsv", str(t)]) == 0
    assert "labels check:" in capsys.readouterr().out
    doc = json.loads(j.read_text())
    assert {"pass", "warn", "fail", "n/a", "missing-crosswalk", "spec_only"} <= set(doc["summary"])
    tsv = list(csv.DictReader(t.open(), delimiter="\t"))
    assert list(tsv[0]) == list(lc.TSV_COLUMNS) and len(tsv) == len(doc["rows"])

    ok = tmp_path / "ok.txt"
    ok.write_text("# release 1\ncoralscapes\n")
    assert main(["labels", "check", "--release-sources", str(ok)]) == 0
    bad = tmp_path / "bad.txt"
    bad.write_text("coralscapes\nno-such-source\n")
    assert main(["labels", "check", "--release-sources", str(bad)]) == 1
    assert main(["labels", "check", "--strict"]) == 1  # class-bearing sources without crosswalk
    assert main(["labels", "check", "coralscapes", "--strict"]) == 0  # narrowed to one id


def test_cli_exit_2_on_unreadable_input(tmp_path):
    assert main(["labels", "check", "--release-sources", str(tmp_path / "missing.txt")]) == 2


def test_cli_report_json_is_the_same_rows_the_api_returns(tmp_path, reg):
    j = Path(tmp_path / "r.json")
    main(["labels", "check", "--json", str(j)])
    assert [r["source_id"] for r in json.loads(j.read_text())["rows"]] == [
        r.source_id for r in lc.evaluate(reg, reg.root)
    ]
