"""``marinedata release publish`` (Phase A, P1.7): the plan, every refusal, the mocked upload.

Nothing here touches the network or a real token: ``HfApi`` is replaced by a recorder.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from marinedata import release_publish as rp
from marinedata.cli import main
from marinedata.flavours import flavour_spec
from marinedata.release_lock import lock_sha256

GOLDEN = Path(__file__).parent / "fixtures" / "release_publish" / "plan_open.golden"
REPOS = {"open": "reefsupport/marine-data", "nc": "reefsupport/marine-data-nc"}
DEFAULT_GATES = ("licence", "dedup", "splits", "masks", "load", "card")
LOCK_TEXT = "release_id: v1.1\nsources: []\n"
TOKEN = "hf_tok-secret-value"


def _gate(passed=True, **extra):
    return {"passed": passed, "counts": {}, "failures": [] if passed else ["a bad row"], **extra}


def _gates(names=DEFAULT_GATES, **override):
    out = {name: _gate() for name in names}
    out.update(override)
    out["summary"] = {"required_for_publish": list(names)}
    return out


def make_release(root: Path, *, flavour="open", release_id="v1.1", gates=..., release=None) -> Path:
    """A 4-file upload (1,512 bytes) next to bookkeeping files that must stay off the Hub."""
    d = root / "releases" / release_id / flavour
    (d / "data" / "coral-masks").mkdir(parents=True)
    (d / "data" / "scene-masks").mkdir()
    (d / "tasks").mkdir()
    (d / "README.md").write_text("# a card\n")  # 9 bytes
    (d / "data" / "coral-masks" / "train-00000-of-00001.parquet").write_bytes(b"t" * 1000)
    (d / "data" / "coral-masks" / "validation-00000-of-00001.parquet").write_bytes(b"v" * 200)
    (d / "data" / "scene-masks" / "train-00000-of-00001.parquet").write_bytes(b"s" * 303)
    for name in ("SPLIT_MAP.json", "DEDUP_GATE.json", "tasks/open.tsv", "tasks/part.tmp"):
        (d / name).write_text("bookkeeping\n" * 50)
    (d / "release.lock.yaml").write_text(LOCK_TEXT)
    record = {
        "release": release_id,
        "flavour": flavour,
        "repo_id": REPOS[flavour],
        "lock_sha256": lock_sha256(d / "release.lock.yaml"),
        "configs": {
            "coral-masks": {"train": 120, "val": 15, "test": 15},
            "scene-masks": {"splits": {"train": {"rows": 8}, "val": {"rows": 1}, "test": 1}},
        },
    }
    (d / "RELEASE.json").write_text(json.dumps({**record, **(release or {})}))
    if gates is not None:
        (d / "GATES.json").write_text(json.dumps(_gates() if gates is ... else gates))
    return d


class FakeApi:
    """Records the calls ``upload`` makes; stands in for ``huggingface_hub.HfApi``."""

    def __init__(self, tags=(), fail_on=None):
        self.calls, self.tags, self.fail_on = [], list(tags), fail_on

    def _call(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.fail_on == name:
            raise RuntimeError("boom")

    def list_repo_refs(self, *args, **kwargs):
        self._call("list_repo_refs", *args, **kwargs)
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def upload_large_folder(self, *args, **kwargs):
        self._call("upload_large_folder", *args, **kwargs)
        return SimpleNamespace(oid="c0ffee")

    def create_tag(self, *args, **kwargs):
        self._call("create_tag", *args, **kwargs)

    def names(self):
        return [name for name, _, _ in self.calls]


@pytest.fixture
def hub(monkeypatch):
    """Patch the Hub client constructor; a test that never reaches it sees ``made == []``."""
    state = SimpleNamespace(api=FakeApi(), made=[])

    def make(token):
        state.made.append(token)
        return state.api

    monkeypatch.setattr(rp, "_hf_api", make)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return state


def publish(*extra, release_id="v1.1", flavour="open"):
    return main(["release", "publish", "--release-id", release_id, "--flavour", flavour, *extra])


def test_dry_run_plan_matches_golden(tmp_path, capsys, hub):
    d = make_release(tmp_path)
    assert publish("--dir", str(d)) == 0
    sha = lock_sha256(d / "release.lock.yaml")
    shown = capsys.readouterr().out.replace(str(d), "<DIR>").replace(sha, "<LOCK>")
    assert shown == GOLDEN.read_text()
    assert hub.made == []


def test_default_dir_is_where_release_build_writes(tmp_path, capsys, hub):
    make_release(tmp_path)
    assert publish("--out", str(tmp_path)) == 0
    assert "files       : 4" in capsys.readouterr().out


def test_nc_flavour_targets_the_nc_repo(tmp_path, capsys, hub):
    d = make_release(tmp_path, flavour="nc")
    assert publish("--dir", str(d), flavour="nc") == 0
    assert "repo        : reefsupport/marine-data-nc (dataset)" in capsys.readouterr().out


def test_flavour_specs_match_the_plan_mapping():
    assert {f: flavour_spec(f).repo_id for f in ("open", "nc")} == REPOS


def test_plan_counts_exactly_what_the_hub_filter_would_send(tmp_path):
    utils = pytest.importorskip("huggingface_hub.utils")
    d = make_release(tmp_path)
    everything = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    sent = list(utils.filter_repo_objects(everything, allow_patterns=list(rp.PLAN_PATTERNS)))
    assert [name for name, _ in rp.upload_files(d)] == sent
    assert "tasks/open.tsv" in everything and "tasks/open.tsv" not in sent


@pytest.mark.parametrize(
    ("gates", "expect"),
    [
        (None, "GATES.json missing"),
        (_gates(dedup=_gate(False)), "gate dedup: failed (a bad row)"),
        (
            _gates(load={"passed": False, "skipped": "no datasets"}),
            "gate load: skipped (no datasets)",
        ),
        ({n: _gate() for n in DEFAULT_GATES[1:]}, "gate licence: missing"),
        ({**_gates(), "privacy": _gate(False, required=True)}, "gate privacy: failed"),
        (
            {**_gates(DEFAULT_GATES), "extra": _gate(False)}
            | {"summary": {"required_for_publish": ["extra"]}},
            "gate extra: failed",
        ),
    ],
    ids=["no-file", "failed", "skipped", "missing", "flagged-required", "listed-in-summary"],
)
def test_refuses_on_gates(tmp_path, capsys, hub, gates, expect):
    d = make_release(tmp_path, gates=gates)
    assert publish("--dir", str(d)) == 1
    captured = capsys.readouterr()
    assert expect in captured.err
    assert captured.out == ""  # a refusal prints no plan


def test_unrequired_failing_gate_does_not_block(tmp_path, capsys, hub):
    d = make_release(tmp_path, gates={**_gates(), "croissant": _gate(False)})
    assert publish("--dir", str(d)) == 0


def test_gates_without_summary_need_the_default_set(tmp_path, capsys, hub):
    d = make_release(tmp_path, gates={"masks": _gate()})  # what 'release gate --masks' alone writes
    assert publish("--dir", str(d)) == 1
    err = capsys.readouterr().err
    assert "gate licence: missing" in err and "gate masks" not in err


def test_malformed_gates_file_is_a_refusal(tmp_path, capsys, hub):
    d = make_release(tmp_path)
    (d / "GATES.json").write_text("[1, 2")
    assert publish("--dir", str(d)) == 1
    assert "GATES.json is not valid JSON" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("release", "expect"),
    [
        ({"flavour": "nc"}, "built as flavour 'nc', not 'open'"),
        ({"flavour": None}, "built as flavour None"),
        ({"repo_id": "reefsupport/marine-data-nc"}, "targets 'reefsupport/marine-data-nc'"),
        ({"release": "v1.0"}, "is release 'v1.0', not 'v1.1'"),
        ({"lock_sha256": ""}, "no lock_sha256"),
        ({"lock_sha256": "0" * 64}, "does not hash to the lock_sha256"),
        ({"configs": {}}, "lists no configs"),
    ],
    ids=["flavour", "no-flavour", "repo", "release-id", "no-lock", "lock-drift", "no-configs"],
)
def test_refuses_on_release_json(tmp_path, capsys, hub, release, expect):
    d = make_release(tmp_path, release=release)
    assert publish("--dir", str(d)) == 1
    assert expect in capsys.readouterr().err


def test_refuses_without_release_json_card_or_folder(tmp_path, capsys, hub):
    d = make_release(tmp_path)
    (d / "RELEASE.json").unlink()
    (d / "README.md").unlink()
    assert publish("--dir", str(d)) == 1
    err = capsys.readouterr().err
    assert "RELEASE.json missing" in err and "card missing" in err
    assert publish("--dir", str(tmp_path / "nope")) == 1
    assert "not a directory" in capsys.readouterr().err


def test_every_reason_is_listed_at_once(tmp_path, capsys, hub):
    d = make_release(tmp_path, gates=None, release={"flavour": "nc", "lock_sha256": ""})
    assert publish("--dir", str(d)) == 1
    assert capsys.readouterr().err.count("release publish: refused:") == 3


@pytest.mark.parametrize("release_id", ["v1.0", "1.0", "v1.0.0", "1"])
def test_go_refuses_frozen_v1_0_without_allow_frozen(
    tmp_path, capsys, hub, monkeypatch, release_id
):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    d = make_release(tmp_path, release_id=release_id)
    assert publish("--dir", str(d), "--go", release_id=release_id) == 1
    assert "frozen v1.0" in capsys.readouterr().err
    assert hub.made == []


def test_frozen_v1_0_dry_run_is_allowed_and_flagged(tmp_path, capsys, hub):
    d = make_release(tmp_path, release_id="v1.0")
    assert publish("--dir", str(d), release_id="v1.0") == 0
    assert "[frozen v1.0]" in capsys.readouterr().out


def test_go_needs_a_token(tmp_path, capsys, hub):
    d = make_release(tmp_path)
    assert publish("--dir", str(d), "--go") == 1
    assert "HF_TOKEN is not set" in capsys.readouterr().err
    assert hub.made == []


def test_go_never_uploads_a_failing_release(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    d = make_release(tmp_path, gates=_gates(splits=_gate(False)))
    assert publish("--dir", str(d), "--go") == 1
    assert "gate splits: failed" in capsys.readouterr().err
    assert hub.made == []


def test_go_uploads_one_commit_then_tags_it(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    hub.api = FakeApi(tags=["v1.0"])
    d = make_release(tmp_path)
    assert publish("--dir", str(d), "--go") == 0
    assert hub.made == [TOKEN]
    assert hub.api.names() == ["list_repo_refs", "upload_large_folder", "create_tag"]
    _, refs_args, refs_kwargs = hub.api.calls[0]
    assert (refs_args, refs_kwargs) == (("reefsupport/marine-data",), {"repo_type": "dataset"})
    _, up_args, up_kwargs = hub.api.calls[1]
    assert up_args == ()
    assert up_kwargs == {
        "repo_id": "reefsupport/marine-data",
        "repo_type": "dataset",
        "folder_path": str(d),
        "allow_patterns": list(rp.UPLOAD_PATTERNS),
    }
    _, tag_args, tag_kwargs = hub.api.calls[2]
    assert tag_args == ("reefsupport/marine-data",)
    assert tag_kwargs["tag"] == "v1.1" and tag_kwargs["revision"] is None
    assert tag_kwargs["repo_type"] == "dataset" and tag_kwargs["exist_ok"] is False
    captured = capsys.readouterr()
    assert "UPLOAD (--go)" in captured.out and "tagged v1.1" in captured.out
    assert TOKEN not in captured.out + captured.err


def test_go_allow_frozen_reaches_the_upload(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    d = make_release(tmp_path, release_id="v1.0")
    assert publish("--dir", str(d), "--go", "--allow-frozen", release_id="v1.0") == 0
    assert hub.api.calls[2][2]["tag"] == "v1.0"


def test_go_stops_before_uploading_when_the_tag_exists(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    hub.api = FakeApi(tags=["v1.1"])
    d = make_release(tmp_path)
    assert publish("--dir", str(d), "--go") == 1
    assert "tag v1.1 already exists" in capsys.readouterr().err
    assert hub.api.names() == ["list_repo_refs"]


def test_go_tag_failure_names_the_commit(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    hub.api = FakeApi(fail_on="create_tag")
    d = make_release(tmp_path)
    assert publish("--dir", str(d), "--go") == 1
    err = capsys.readouterr().err
    assert "uploaded reefsupport/marine-data" in err and "tag it by hand" in err


def test_go_upload_error_is_reported_without_the_token(tmp_path, capsys, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    hub.api = FakeApi(fail_on="upload_large_folder")
    d = make_release(tmp_path)
    assert publish("--dir", str(d), "--go") == 1
    captured = capsys.readouterr()
    assert "upload failed: RuntimeError: boom" in captured.err
    assert hub.api.names() == ["list_repo_refs", "upload_large_folder"]
    assert TOKEN not in captured.out + captured.err


@pytest.mark.parametrize(
    ("release_id", "frozen"),
    [("1.0", True), ("v1.0", True), ("v1.0.0", True), ("1", True),
     ("v1.1", False), ("v1.0.1", False), ("v2.0", False), ("t1", False), ("", False)],
)  # fmt: skip
def test_is_frozen(release_id, frozen):
    assert rp.is_frozen(release_id) is frozen


@pytest.mark.parametrize(
    ("release_id", "tag"), [("1.1", "v1.1"), ("v1.1", "v1.1"), ("t1", "vt1"), ("v1.0", "v1.0")]
)
def test_tag_for(release_id, tag):
    assert rp.tag_for(release_id) == tag


def test_publish_is_listed_under_release(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["release", "publish", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--go" in out and "--allow-frozen" in out
