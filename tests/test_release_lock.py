"""release.lock.yaml (Phase A, P1.3): generator, loader, and 'no lock, no build'."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from marinedata import release_lock as rl
from marinedata.cli import main
from marinedata.registry import Registry
from marinedata.release_lock import (
    ReleaseLockError,
    build_lock,
    check_build_against_lock,
    load_lock,
    write_lock,
)

REQUIRES_RELEASE_LOCK = True  # conftest: do not apply the pre-lock test escape here
REG = Registry.load()
PINNED = ["coralscapes", "reef-support-benthic-own", "reef-support-seaview-labels", "suim"]
FIXED = {"created": "2026-10-08T00:00:00+00:00", "code_sha": "0" * 40}
# v1.0 sources and the rows they contributed (INV-HF section 1; RELEASE.json is not in the repo
# nor under releases/marine-data/v1.0/ in rs-storage-open).
RELEASED = {
    "coralscapes": 2075, "reef-support-benthic-own": 1226, "reef-support-seaview-labels": 2696,
    "suim": 1598, "uiis": 4622, "uiis10k": 10042, "usis10k": 10632, "roboflow-aquarium": 637,
}  # fmt: skip


def _lock(ids=PINNED):
    return build_lock(
        REG, release_id="t1", flavour="open", configs=["coral-masks"], source_ids=ids, **FIXED
    )


def test_roundtrip_matches_registry_pins(tmp_path):
    lock = _lock()
    path = write_lock(lock, tmp_path / "release.lock.yaml")
    assert load_lock(path, REG) == lock
    for e in lock.sources:
        assert e.root_digest == REG.source(e.id).checksums.root_digest
        assert e.access_class == "open" and e.version == str(REG.source(e.id).version)


def test_missing_pin_fails_listing_every_id(monkeypatch):
    orig = rl._pin
    monkeypatch.setattr(
        rl, "_pin", lambda reg, i: "" if i in ("suim", "coralscapes") else orig(reg, i)
    )
    with pytest.raises(ReleaseLockError, match="coralscapes, suim"):
        _lock()


def test_loader_rejects_tampered_digest_version_and_class(tmp_path):
    path = write_lock(_lock(), tmp_path / "l.yaml")
    for key, val in (
        ("root_digest", "0" * 64),
        ("version", "9.9"),
        ("access_class", "restricted-nc"),
    ):
        bad = yaml.safe_load(path.read_text())
        bad["sources"][0][key] = val
        other = tmp_path / f"{key}.yaml"
        other.write_text(yaml.safe_dump(bad))
        with pytest.raises(ReleaseLockError, match=key if key != "root_digest" else "root_digest"):
            load_lock(other, REG)


def test_denied_or_wrong_flavour_source_cannot_be_locked():
    with pytest.raises(ReleaseLockError, match="duo"):
        _lock([*PINNED, "duo"])


def test_cli_lock_writes_file_and_fails_on_missing_pin(tmp_path, capsys, monkeypatch):
    out = tmp_path / "release.lock.yaml"
    args = ["release", "lock", "--release-id", "t1", "--flavour", "open", "--config", "coral-masks",
            "--source", "suim", "--out", str(out)]  # fmt: skip
    assert main(args) == 0 and load_lock(out, REG).source_ids() == {"suim"}
    monkeypatch.setattr(rl, "_pin", lambda reg, i: "")
    assert main(args) == 1 and "suim" in capsys.readouterr().err


def test_build_refuses_without_a_lock(tmp_path, capsys):
    rc = main(["release", "build", "--release", "t1", "--split-map", str(tmp_path / "sm.json"),
               "--flavour", "open", "--out", str(tmp_path), "--generate-split-map"])  # fmt: skip
    assert rc == 1 and "--lock" in capsys.readouterr().err


def test_build_refuses_a_source_the_lock_does_not_list():
    lock = _lock(["suim"])
    with pytest.raises(ReleaseLockError, match=r"^source not in the release lock: .*coralscapes"):
        check_build_against_lock(lock, REG, release_id="t1", flavour="open", profile="ship-open")
    with pytest.raises(ReleaseLockError, match="not 't2'"):
        check_build_against_lock(lock, REG, release_id="t2", flavour="open", profile="ship-open")


def test_v10_public_lock_is_complete_or_names_every_missing_pin():
    missing = sorted(i for i in RELEASED if not rl._pin(REG, i))
    if missing:  # registry gap (no silent lock): the error must name each one
        with pytest.raises(ReleaseLockError) as exc:
            _lock(sorted(RELEASED))
        assert all(i in str(exc.value) for i in missing)
    lock = _lock([i for i in sorted(RELEASED) if i not in missing])
    for e in lock.sources:  # released rows are a subset of the pinned tree
        assert e.n_images is not None and RELEASED[e.id] <= e.n_images


def test_committed_v10_lock_matches_registry_pins():
    # committed v1.0 open lock: the 8 public sources, no drift from the registry pins
    path = (
        Path(__file__).resolve().parents[1]
        / "releases"
        / "marine-data"
        / "v1.0"
        / "release.lock.yaml"
    )
    lock = load_lock(path, REG)
    assert sorted(e.id for e in lock.sources) == sorted(RELEASED)
    # v1.0 ships four configs; the lock is flat: all four config ids + the union of their 8 sources
    assert sorted(lock.configs) == ["coral-masks", "fish-boxes", "instance-masks", "scene-masks"]
    assert len(lock.sources) == 8
    for sha in (lock.code_sha, lock.reconstructed_at_sha):  # real commits, not the placeholder
        assert sha and len(sha) == 40 and set(sha) <= set("0123456789abcdef")
        assert set(sha) != {"0"}
    # v1.0 was built by hand on 2026-10-07 from a commit that cannot be identified: say so
    assert lock.reconstructed is True
    comments = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    assert len(comments) == 1 and "2026-10-08" in comments[0] and "build time" in comments[0]
    for e in lock.sources:
        assert e.access_class == "open"
        assert e.root_digest == REG.source(e.id).checksums.root_digest
        assert e.n_images is not None and RELEASED[e.id] <= e.n_images


def test_reconstructed_keys_roundtrip_and_stay_out_of_ordinary_locks(tmp_path):
    plain = rl.write_lock(_lock(), tmp_path / "plain.yaml")
    assert "reconstructed" not in plain.read_text(encoding="utf-8")
    assert rl.load_lock(plain, REG).reconstructed is False
    sha = "b" * 40
    rebuilt = build_lock(
        REG,
        release_id="t1",
        flavour="open",
        configs=["coral-masks"],
        source_ids=PINNED,
        reconstructed_at_sha=sha,
        **FIXED,
    )
    back = rl.load_lock(rl.write_lock(rebuilt, tmp_path / "rebuilt.yaml"), REG)
    assert back.reconstructed is True and back.reconstructed_at_sha == sha
    assert back.code_sha == FIXED["code_sha"]  # code_sha is kept, not overwritten
