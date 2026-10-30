"""``marinedata release publish --part``: one part per call, the tag on the sha they made (PR-P).

``HfApi`` is the recorder of test_release_publish: nothing here touches the network or a token.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from test_release_publish import DEFAULT_GATES, TOKEN, FakeApi, _gate, _gates, hub, make_release
from test_release_publish import publish as run_publish

from marinedata import release_publish as rp
from marinedata import release_publish_parts as parts
from marinedata.gates import V11_REQUIRED, GateResult, merge_gates_json, publish_verdict
from marinedata.gates.v11_runner import V11_ALL

__all__ = ["hub"]  # a fixture imported from test_release_publish

ROOT = Path(__file__).parent.parent
RUNBOOK = ROOT / "docs" / "release-runbook.md"
V10_GATES = ("licence", "dedup", "splits", "masks", "privacy", "load", "card", "croissant")
IMAGES = ("train-00000-of-00002.parquet", "train-00001-of-00002.parquet")
EXTRAS = {
    "LICENSE": "licence\n", "NOTICE": "notice\n", "CITATION.cff": "cff-version: 1.2.0\n",
    "croissant.json": "{}", "assets/logo.png": "png", "docs/GATES.md": "# gates\n",
}  # fmt: skip
PARTS = ["data:coral-masks", "data:images", "data:scene-masks", "provenance", "card", "tag"]
FORBIDDEN = {"create_repo", "update_repo_visibility", "update_repo_settings", "delete_repo"}


def make_v11(root: Path, *, override=None, **kwargs) -> Path:
    """The v1.0-shaped fixture plus a ``data/images`` config, the export's extras and G1..G9."""
    d = make_release(
        root, gates=_gates((*DEFAULT_GATES, *V11_REQUIRED), **(override or {})), **kwargs
    )
    (d / "data" / "images").mkdir()
    for name in IMAGES:
        (d / "data" / "images" / name).write_bytes(b"i" * 50)
    for rel, text in EXTRAS.items():
        (d / rel).parent.mkdir(exist_ok=True)
        (d / rel).write_text(text)
    return d


def publish(d, *extra, **kwargs):
    return run_publish("--dir", str(d), *extra, **kwargs)


def hub_sends(d: Path, patterns) -> list[str]:
    """What the Hub client would send for ``allow_patterns`` (its own filter, not ours)."""
    utils = pytest.importorskip("huggingface_hub.utils")
    everything = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    return list(utils.filter_repo_objects(everything, allow_patterns=list(patterns)))


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)


def test_part_data_uploads_only_that_configs_paths(tmp_path, hub, token):
    d = make_v11(tmp_path)
    assert publish(d, "--go", "--part", "data:images") == 0
    assert hub.api.names() == ["list_repo_refs", "upload_large_folder"]
    patterns = hub.api.calls[1][2]["allow_patterns"]
    assert patterns == ["data/images/*.parquet"]
    assert hub_sends(d, patterns) == [f"data/images/{name}" for name in IMAGES]
    state = json.loads((d / parts.STATE_FILE).read_text())
    assert state["uploads"] == [{"part": "data:images", "sha": None}]


def test_vm_loop_one_part_per_call_card_alone_and_the_tag_on_the_card_commit(
    tmp_path, capsys, hub, token
):
    d = make_v11(tmp_path)
    assert publish(d, "--list-parts") == 0
    listed = capsys.readouterr().out.split()
    assert listed == PARTS
    for name in listed:
        assert publish(d, "--go", "--part", name) == 0
    uploads = [(n, kw["allow_patterns"]) for n, _, kw in hub.api.calls if n.startswith("upload")]
    assert uploads == [
        ("upload_large_folder", ["data/coral-masks/*.parquet"]),
        ("upload_large_folder", ["data/images/*.parquet"]),
        ("upload_large_folder", ["data/scene-masks/*.parquet"]),
        ("upload_folder", list(rp.PROVENANCE_PATTERNS)),
        ("upload_folder", ["README.md"]),
    ]
    assert hub_sends(d, ["README.md"]) == ["README.md"]
    tags = [(a, kw) for n, a, kw in hub.api.calls if n == "create_tag"]
    assert len(tags) == 1 and len(hub.api.commits) == 2
    assert tags[0][0] == ("reefsupport/marine-data",)
    assert tags[0][1]["tag"] == "v1.1" and tags[0][1]["revision"] == hub.api.commits[1]
    assert re.fullmatch(r"[0-9a-f]{40}", tags[0][1]["revision"])
    assert hub.api.names()[-2:] == ["list_repo_refs", "create_tag"]  # the tag call uploads nothing


@pytest.mark.parametrize("part", ["data:images", "provenance", "card", "tag"])
def test_without_go_nothing_is_uploaded(tmp_path, capsys, hub, part):
    d = make_v11(tmp_path)
    assert publish(d, "--part", part) == 0
    out = capsys.readouterr().out
    assert "DRY RUN" in out and f"this run    : {part}" in out
    assert hub.made == [] and hub.api.calls == []
    assert not (d / "provenance").exists() and not (d / parts.STATE_FILE).exists()


def test_a_part_that_is_not_in_the_build_is_refused(tmp_path, capsys, hub, token):
    d = make_v11(tmp_path)
    assert publish(d, "--go", "--part", "data:nope") == 1
    assert "is not a part of this build" in capsys.readouterr().err
    assert publish(d, "--go", "--part", "data:../x") == 1
    assert hub.made == []


def test_card_waits_for_every_other_part(tmp_path, capsys, hub, token):
    d = make_v11(tmp_path)
    assert publish(d, "--go", "--part", "card") == 1
    err = capsys.readouterr().err
    assert "card is uploaded last: not uploaded yet: data:coral-masks, data:images" in err
    assert "provenance" in err and "upload_folder" not in hub.api.names()


def test_tag_needs_the_card_as_the_last_upload(tmp_path, capsys, hub, token):
    d = make_v11(tmp_path)
    assert publish(d, "--go", "--part", "tag") == 1
    assert "tag needs the card as the last upload (that is nothing)" in capsys.readouterr().err
    for name in (*PARTS[:-1], "provenance"):  # something is uploaded after the card
        assert publish(d, "--go", "--part", name) == 0
    assert publish(d, "--go", "--part", "tag") == 1
    assert "(that is provenance)" in capsys.readouterr().err
    assert "create_tag" not in hub.api.names()


def test_a_state_file_of_another_publish_is_refused(tmp_path, capsys, hub, token):
    d = make_v11(tmp_path)
    (d / parts.STATE_FILE).write_text(json.dumps({"repo_id": "x/y", "uploads": []}))
    assert publish(d, "--go", "--part", "provenance") == 1
    assert "records another publish" in capsys.readouterr().err
    assert "upload_folder" not in hub.api.names()


@pytest.mark.parametrize("private", [False, None])
def test_nc_refuses_to_proceed_unless_the_repo_is_private(tmp_path, capsys, hub, token, private):
    hub.api = FakeApi(private=private)
    d = make_v11(tmp_path, flavour="nc")
    assert publish(d, "--go", "--part", "data:images", flavour="nc") == 1
    assert "reefsupport/marine-data-nc is not private" in capsys.readouterr().err
    assert hub.api.names() == ["repo_info"]


def test_nc_goes_ahead_on_a_private_repo_and_checks_it_first(tmp_path, hub, token):
    d = make_v11(tmp_path, flavour="nc")
    assert publish(d, "--go", "--part", "data:images", flavour="nc") == 0
    assert hub.api.names() == ["repo_info", "list_repo_refs", "upload_large_folder"]
    assert hub.api.calls[0][1:] == (("reefsupport/marine-data-nc",), {"repo_type": "dataset"})


@pytest.mark.parametrize("module", [rp, parts])
def test_no_visibility_or_repo_creation_call_anywhere(module):
    tree = ast.parse(Path(module.__file__).read_text())
    called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    called |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    keywords = {k.arg for n in ast.walk(tree) if isinstance(n, ast.Call) for k in n.keywords}
    assert not called & FORBIDDEN and "private" not in keywords


def test_plan_lists_added_paths_the_v1_delta_and_g2_g3(tmp_path, capsys, hub):
    d = make_v11(tmp_path)
    v1 = tmp_path / "v1"
    (v1 / "data" / "coral-masks").mkdir(parents=True)
    (v1 / "data" / "coral-masks" / "train-00000-of-00001.parquet").write_bytes(b"t" * 1000)
    (v1 / "data" / "coral-masks" / "validation-00000-of-00001.parquet").write_bytes(b"X" * 200)
    (v1 / "README.md").write_text("# the v1.0 card, which v1.1 replaces\n")
    (v1 / ".gitattributes").write_text("*.parquet\n")
    assert publish(d, "--v1-dir", str(v1)) == 0
    out = capsys.readouterr().out
    assert "adds 12, replaces 2, identical 1" in out
    for added in (f"data/images/{IMAGES[0]}", "provenance/GATES.json", "LICENSE", "docs/GATES.md"):
        assert f"    + {added}\n" in out
    assert "    ~ README.md\n" in out
    assert (
        "    ~ data/coral-masks/validation-00000-of-00001.parquet  (replaces a prior-release shard"
        in out
    )
    assert ".gitattributes" not in out
    assert "G2 frozen v1.0: PASS" in out and "G3 leak       : PASS" in out
    assert hub.made == []


def test_a_failing_g2_is_reported_and_blocks(tmp_path, capsys, hub):
    d = make_v11(tmp_path, override={"frozen": _gate(False)})
    assert publish(d) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and "gate frozen: failed (a bad row)" in captured.err
    assert "G2 frozen v1.0: FAIL (a bad row)" in captured.err
    assert "G3 leak       : PASS" in captured.err


def test_provenance_part_carries_everything_the_card_links_to(tmp_path, capsys, hub, token):
    d = make_v11(tmp_path)
    assert publish(d, "--part", "provenance") == 0
    line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("    provenance:"))
    wanted = ("provenance/GATES.json", "provenance/release.lock.yaml", "provenance/RELEASE.json",
              "LICENSE", "NOTICE", "CITATION.cff", "croissant.json", "assets/logo.png",
              "docs/GATES.md")  # fmt: skip
    assert all(rel in line for rel in wanted) and "README.md" not in line
    assert publish(d, "--go", "--part", "provenance") == 0
    assert set(wanted) <= set(hub_sends(d, hub.api.calls[1][2]["allow_patterns"]))


def test_the_parts_cover_the_upload_set_exactly_once(tmp_path):
    d = make_v11(tmp_path)
    assert {"LICENSE", "NOTICE", "CITATION.cff", "assets/*", "docs/**"} <= set(rp.UPLOAD_PATTERNS)
    (d / "provenance").mkdir()
    for name in rp.PROVENANCE_FILES:
        (d / "provenance" / name).write_text("{}")
    per_part = [hub_sends(d, parts.part_patterns(p)) for p in PARTS[:-1]]
    flat = [rel for files in per_part for rel in files]
    assert sorted(flat) == sorted(hub_sends(d, rp.UPLOAD_PATTERNS)) and len(flat) == len(set(flat))
    assert parts.part_patterns("tag") == ()


def test_gates_json_keeps_both_gate_sets_and_the_verdict_reads_the_merged_file(tmp_path):
    d = make_v11(tmp_path)
    path = d / "GATES.json"
    path.unlink()
    merge_gates_json(path, [GateResult(name, True) for name in V10_GATES])  # release gate --all
    blocked = publish_verdict(d)[1]
    assert "gate frozen: missing" in blocked and "gate licence: missing" not in blocked
    assert set(V11_REQUIRED) <= set(V11_ALL)
    merge_gates_json(path, [GateResult(name, True) for name in V11_ALL])  # release gate --v11
    kept = set(json.loads(path.read_text()))
    assert set(V10_GATES) | set(V11_REQUIRED) | {"summary"} <= kept
    assert publish_verdict(d)[1] == []
    path.unlink()  # --v11 on its own: the v1.0 gates were never run
    merge_gates_json(path, [GateResult(name, True) for name in V11_ALL])
    assert "gate licence: missing" in publish_verdict(d)[1]


def test_runbook_upload_list_is_upload_patterns_and_documents_the_loop():
    text = RUNBOOK.read_text()
    block = text.split("<!-- upload-patterns:start -->")[1].split("<!-- upload-patterns:end -->")[0]
    assert tuple(re.findall(r"^- `([^`]+)`", block, flags=re.M)) == rp.UPLOAD_PATTERNS
    assert "are never uploaded" not in text
    for needle in ("--list-parts", "--part", "--citation", "--all", "--v11", "delete_tag"):
        assert needle in text
