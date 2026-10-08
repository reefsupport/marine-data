"""deploy/release-vm scripts: they parse, print the exact hcloud commands and refuse early.

Nothing here runs docker or the real hcloud. A recorder named ``hcloud`` is put first on PATH,
so a refusal that reached the CLI would leave a trace in ``calls``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

DEPLOY = Path(__file__).resolve().parent.parent / "deploy" / "release-vm"
BASH = shutil.which("bash")
SHA = "a" * 40
GOOD = {"MARINEDATA_REF": SHA, "RELEASE_ID": "v1.1", "FLAVOUR": "open"}
KEYS = "MARINEDATA_S3_ACCESS_KEY=ak\nMARINEDATA_S3_SECRET_KEY=s3cr3t-value\n"
pytestmark = pytest.mark.skipif(BASH is None, reason="bash is not installed")


@pytest.fixture
def sandbox(tmp_path):
    """(env, calls): a PATH with a recording ``hcloud`` first; ``calls`` exists once it ran."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "hcloud.calls"
    stub = bin_dir / "hcloud"
    stub.write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\n')
    stub.chmod(0o755)
    env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "LC_ALL": "C", "TMPDIR": str(tmp_path)}
    return env, calls


def create(env, *args):
    cmd = [BASH, str(DEPLOY / "create.sh"), *args]
    return subprocess.run(
        cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False
    )


def args_with(over=None):
    base = {"--release-id": "v1.1", "--flavour": "open", "--ref": SHA, **(over or {})}
    return [word for pair in base.items() for word in pair]


def bootstrap(snippet, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "BOOTSTRAP_SOURCE_ONLY": "1"}
    script = f'. "{DEPLOY}/bootstrap.sh"; {snippet}'
    return subprocess.run(
        [BASH, "-c", script],
        env={**env, **(env_extra or {})},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("name", ["bootstrap.sh", "create.sh"])
def test_script_parses_and_is_executable(name):
    path = DEPLOY / name
    assert os.access(path, os.X_OK)
    done = subprocess.run([BASH, "-n", str(path)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


def test_help_exits_zero(sandbox):
    env, _ = sandbox
    done = create(env, "--help")
    assert done.returncode == 0
    assert "usage: create.sh" in done.stdout


def test_print_only_is_the_default_and_prints_the_exact_commands(sandbox, tmp_path):
    env, calls = sandbox
    ud = tmp_path / "ud"
    done = create(env, *args_with(), "--out-dir", str(ud))
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "PRINT ONLY" in out
    assert "hcloud volume create --name marinedata-v1-1-open-data --size 200" in out
    assert "--location hel1 --format ext4" in out
    assert "hcloud server create --name marinedata-v1-1-open --type cx32" in out
    assert "--image ubuntu-24.04 --location hel1 --ssh-key YOUR_SSH_KEY" in out
    assert f"--user-data-from-file {ud}/user-data.sh --volume marinedata-v1-1-open-data" in out
    assert "hcloud server delete marinedata-v1-1-open" in out
    assert "hcloud volume delete marinedata-v1-1-open-data" in out
    assert not calls.exists()


def test_user_data_carries_settings_only(sandbox, tmp_path):
    env, _ = sandbox
    ud = tmp_path / "ud"
    assert create(env, *args_with(), "--out-dir", str(ud)).returncode == 0
    text = (ud / "user-data.sh").read_text()
    lines = text.splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert [x for x in lines if x.startswith("export ")] == [
        f"export MARINEDATA_REF={SHA}",
        "export RELEASE_ID=v1.1",
        "export FLAVOUR=open",
    ]
    assert len(text.encode()) < 32 * 1024  # Hetzner's user-data limit
    parse = subprocess.run([BASH, "-n", str(ud / "user-data.sh")], check=False)
    assert parse.returncode == 0


def test_extra_settings_are_exported_quoted(sandbox, tmp_path):
    env, _ = sandbox
    ud = tmp_path / "ud"
    done = create(
        env, *args_with(), "--build-args", "--decon --profile research", "--out-dir", str(ud)
    )
    assert done.returncode == 0, done.stderr
    text = (ud / "user-data.sh").read_text()
    assert "export RELEASE_BUILD_ARGS=--decon\\ --profile\\ research" in text


def test_run_needs_a_token(sandbox, tmp_path):
    env, calls = sandbox
    done = create(env, *args_with(), "--out-dir", str(tmp_path / "ud"), "--run", "--ssh-key", "k")
    assert done.returncode != 0
    assert "HCLOUD_TOKEN" in done.stderr
    assert not calls.exists()


def test_run_needs_an_ssh_key(sandbox, tmp_path):
    env, calls = sandbox
    env = {**env, "HCLOUD_TOKEN": "not-a-real-token"}
    done = create(env, *args_with(), "--out-dir", str(tmp_path / "ud"), "--run")
    assert done.returncode != 0
    assert "--ssh-key" in done.stderr
    assert not calls.exists()


def test_run_needs_a_confirmation(sandbox, tmp_path):
    env, calls = sandbox
    env = {**env, "HCLOUD_TOKEN": "not-a-real-token"}
    done = create(env, *args_with(), "--out-dir", str(tmp_path / "ud"), "--run", "--ssh-key", "k")
    assert done.returncode != 0
    assert "terminal" in done.stderr
    assert not calls.exists()


@pytest.mark.parametrize(
    "over",
    [
        {"--flavour": "both"},
        {"--release-id": "v1/../x"},
        {"--ref": "--upload-pack=evil"},
        {"--lock-file": "/etc/passwd"},
    ],
)
def test_create_refuses_bad_settings(sandbox, tmp_path, over):
    env, calls = sandbox
    ud = tmp_path / "ud"
    done = create(env, *args_with(over), "--out-dir", str(ud))
    assert done.returncode != 0
    assert not ud.exists()
    assert not calls.exists()


def test_validate_settings_accepts_a_good_set():
    assert bootstrap("validate_settings", GOOD).returncode == 0


@pytest.mark.parametrize(
    ("over", "message"),
    [
        ({"FLAVOUR": "both"}, "FLAVOUR must be open or nc"),
        ({"FLAVOUR": ""}, "FLAVOUR must be open or nc"),
        ({"RELEASE_ID": "v1/../x"}, "RELEASE_ID"),
        ({"RELEASE_ID": ""}, "RELEASE_ID is required"),
        ({"MARINEDATA_REF": "-x"}, "MARINEDATA_REF"),
        ({"MARINEDATA_REF": ""}, "MARINEDATA_REF is required"),
        ({"LOCK_FILE": "/etc/passwd"}, "LOCK_FILE"),
        ({"SPLIT_MAP_FILE": "../x.json"}, "SPLIT_MAP_FILE"),
        ({"STAGES": "build upload"}, "unknown stage 'upload'"),
    ],
)
def test_validate_settings_refuses(over, message):
    done = bootstrap("validate_settings", {**GOOD, **over})
    assert done.returncode == 1
    assert message in done.stderr


def secrets_env(tmp_path, body, mode=0o600):
    path = tmp_path / "marinedata.env"
    path.write_text(body)
    path.chmod(mode)
    return {**GOOD, "SECRETS_FILE": str(path), "SECRETS_WAIT_SECONDS": "0"}


def test_secrets_file_is_accepted_and_never_echoed(tmp_path):
    done = bootstrap("wait_for_secrets", secrets_env(tmp_path, KEYS))
    assert done.returncode == 0, done.stderr
    assert "s3cr3t-value" not in done.stdout + done.stderr


@pytest.mark.parametrize(
    ("body", "mode", "message"),
    [
        (KEYS, 0o644, "chmod 600"),
        (KEYS + "HF_TOKEN=hf_x\n", 0o600, "HF_TOKEN must not be in"),
        ("MARINEDATA_S3_ACCESS_KEY=ak\n", 0o600, "no complete"),
    ],
)
def test_secrets_file_is_refused(tmp_path, body, mode, message):
    done = bootstrap("wait_for_secrets", secrets_env(tmp_path, body, mode))
    assert done.returncode == 1
    assert message in done.stderr


def test_missing_secrets_file_is_a_bounded_wait(tmp_path):
    env = {**GOOD, "SECRETS_FILE": str(tmp_path / "nope.env"), "SECRETS_WAIT_SECONDS": "0"}
    done = bootstrap("wait_for_secrets", env)
    assert done.returncode == 1
    assert "no complete" in done.stderr
