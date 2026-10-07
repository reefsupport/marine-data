"""WP-L1b: release flavours (open | nc), the NC gate profile and the per-flavour cards."""

# ruff: noqa: E501, E731
from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_cli_release import _image_bytes, _registry, _source

from marinedata.checksums import file_digest
from marinedata.cli import main
from marinedata.enums import AccessClass, Tier
from marinedata.gate import evaluate
from marinedata.hf_card import render_card
from marinedata.models import Licence
from marinedata.registry import Registry
from marinedata.tables import StagedImage, write_metadata_table

IDS = {  # source id -> (tier, access class, image-seed offset)
    "open1": (Tier.PERMISSIVE, AccessClass.OPEN, 0),
    "nc1": (Tier.NONCOMMERCIAL, AccessClass.RESTRICTED_NC, 100),
    "nd1": (Tier.NONCOMMERCIAL, AccessClass.RESTRICTED_ND, 200),
    "int1": (Tier.NONCOMMERCIAL, AccessClass.INTERNAL_ONLY, 300),
    "unk1": (Tier.PERMISSIVE, AccessClass.UNKNOWN, 400),
}


def test_gate_admission_agrees_with_access_class() -> None:
    reg = Registry.load()
    allowed = {
        "ship-commercial": {"open"},
        "ship-open": {"open"},
        "ship-noncommercial": {"open", "restricted-nc"},
    }
    for pid, classes in allowed.items():
        for s in reg:
            if evaluate(s, reg.profile(pid)).allowed:
                assert s.access_class.value in classes, (pid, s.id, s.access_class)
    nc = [s for s in reg if evaluate(s, reg.profile("ship-noncommercial")).allowed]
    assert any(s.access_class is AccessClass.RESTRICTED_NC for s in nc)
    assert not any(
        s.access_class.value in ("restricted-nd", "internal-only", "unknown") for s in nc
    )


def _stage(root: Path, sid: str, offset: int) -> Path:
    rows = []
    for i in range(10):
        p = root / "images" / "p" / f"a{i}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(_image_bytes(offset + i))
        rows.append(
            StagedImage(stem=f"a{i}", partition="p", upstream_path=f"o/a{i}.jpg",
                        upstream_split=None, width=4, height=4, split_group=f"{sid}/g{i}")
        )  # fmt: skip
    write_metadata_table(root / "metadata.parquet", rows)
    return root


@pytest.fixture
def built(monkeypatch, tmp_path):
    base = _registry()
    sources = {}
    for sid, (tier, cls, _) in IDS.items():
        lic = Licence(id=f"L-{sid}", name=sid, tier=tier)
        sources[sid] = _source().model_copy(update={"id": sid, "name": sid, "licence": lic,
                                                    "access_class": cls})  # fmt: skip
    reg = type(base)(sources=sources, licences={}, profiles={p.id: p for p in base.profiles},
                     schemas={s.id: s for s in base.schemas}, tasks={t.id: t for t in base.tasks})  # fmt: skip
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: reg))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "cache"))
    roots = {sid: _stage(tmp_path / "src" / sid, sid, IDS[sid][2]) for sid in IDS}
    local = [f"--local={sid}={root}" for sid, root in roots.items()]

    def run(*extra: str) -> int:
        return main(["release", "build", "--release", "r1", "--split-map", str(tmp_path / "sm.json"),
                     "--out", str(tmp_path / "out"), *local, *extra])  # fmt: skip

    return run, roots, tmp_path / "out" / "releases" / "r1"


def _shas(root: Path) -> set[str]:
    return {file_digest(p) for p in (root / "images").rglob("*.jpg")}


def test_both_flavours_are_disjoint_and_ship_no_nd_internal_unknown(built) -> None:
    run, roots, rel = built
    assert run("--flavour", "open", "--generate-split-map", "--no-near-dup") == 0
    assert run("--flavour", "nc") == 0
    tsv = lambda f: {
        ln.split("\t")[0]
        for ln in (rel / f / "tasks" / "pretrain-set.tsv").read_text().splitlines()[1:]
    }
    open_shas, nc_shas = tsv("open"), tsv("nc")
    assert open_shas and nc_shas and not (open_shas & nc_shas)
    assert open_shas <= _shas(roots["open1"]) and nc_shas <= _shas(
        roots["nc1"]
    )  # a near-dup drop is fine
    for sid in ("nd1", "int1", "unk1"):
        assert not ((open_shas | nc_shas) & _shas(roots[sid]))
    meta = json.loads((rel / "nc" / "RELEASE.json").read_text())
    assert meta["flavour"] == "nc" and meta["repo_id"] == "reefsupport/marine-data-nc"
    assert [s["id"] for s in meta["sources"]] == ["nc1"]
    assert meta["excluded_sources"]["class-restricted-nd"] == ["nd1"]
    assert meta["excluded_sources"]["class-internal-only"] == ["int1"]
    assert meta["class_counts"]["release_sources"] == {"restricted-nc": 1}


def test_cli_refuses_missing_flavour(built, capsys) -> None:
    run, _, rel = built
    assert run("--generate-split-map", "--no-near-dup") == 1
    assert "--flavour" in capsys.readouterr().err and not rel.exists()


def test_cli_refuses_research_profile_with_a_flavour(built, capsys) -> None:
    run, _, _ = built
    assert (
        run("--flavour", "open", "--profile", "research", "--generate-split-map", "--no-near-dup")
        == 1
    )
    assert "must be built under" in capsys.readouterr().err


class _D(dict):
    def __missing__(self, key):
        return 0


def _card(flavour: str) -> str:
    sources = [{"id": "s1", "version": "v1", "licence": "CC-BY-4.0", "licence_hf": "cc-by-4.0",
                "tier": "T1_PERMISSIVE", "images": 3, "citation": "C", "attribution": "A. Author"}]  # fmt: skip
    summary = {"configs": {"images": {"splits": {"train": {"rows": 3}}}}}
    release = {"release": "r1", "split_map_sha256": "0" * 64, "near_dup": _D(algorithm="dHash", pil_version="x"),
               "never_eval_near_dup_excluded": _D()}  # fmt: skip
    return render_card(summary, release, sources, pretty_name="X", flavour=flavour,
                       repo_id="reefsupport/marine-data" + ("-nc" if flavour == "nc" else ""),
                       takedown_url="https://github.com/reefsupport/marine-data/issues")  # fmt: skip


def _header(card: str) -> str:
    return card.split("---\n")[1]


def test_open_card_snapshot() -> None:
    card = _card("open")
    head = _header(card)
    assert "license: other\nlicense_name: mixed-open\nlicense_link: LICENSE\n" in head
    assert "extra_gated" not in head
    assert "| `s1` | CC-BY-4.0 | A. Author |" in card
    # The public card never names or links the gated repo.
    assert "marine-data-nc" not in card and "-nc repo" not in card and "gated" not in card
    assert "akedown" not in card and "## Contact" in card
    assert "## Excluded sources" in card and "restricted-nd" in card
    assert "https://github.com/reefsupport/marine-data/issues" in card


def test_nc_card_snapshot() -> None:
    card = _card("nc")
    head = _header(card)
    assert "license_name: mixed-non-commercial" in head
    assert "extra_gated_prompt: " in head and "non-commercial" in head.lower()
    assert ("extra_gated_fields:\n  Name: text\n  Affiliation: text\n  Intended use: text\n"
            "  I will use this dataset for non-commercial purposes only: checkbox\n") in head  # fmt: skip
    assert "combined set" in head and "## Excluded from both repos" in card
    assert "NON-COMMERCIAL, gated" in card


def test_nc_card_says_the_nc_flavour_is_open_plus_restricted_together() -> None:
    """WP-R11 (Yohan): the restricted flavour is composed of both the open and restricted
    options together; the card names both repos and shows a snippet that loads both."""
    card = _card("nc")
    assert "composed of both the open and the restricted options together" in card
    assert "`reefsupport/marine-data` (open) + this repo (restricted-nc)" in card
    assert "## The NC flavour: open + restricted-nc together" in card
    assert 'load_dataset("reefsupport/marine-data", "images", split="train")' in card
    assert 'load_dataset("reefsupport/marine-data-nc", "images", split="train")  # gated' in card
    assert "concatenate_datasets([open_, restricted])" in card
    assert "extra_gated_prompt: " in _header(card)  # gating kept


def test_local_only_never_fetches_and_builds_from_local_trees(built, monkeypatch) -> None:
    """WP-R1: --local-only builds (incl. the split-map generation) from the --local trees alone."""
    import marinedata.cli_release as cr

    def _no_fetch(*a, **k):
        raise AssertionError("--local-only must never resolve/fetch a root")

    monkeypatch.setattr(cr, "_resolve_roots", _no_fetch)
    run, _roots, rel = built
    assert run("--flavour", "open", "--generate-split-map", "--no-near-dup", "--local-only") == 0
    assert (rel / "open" / "RELEASE.json").is_file()
