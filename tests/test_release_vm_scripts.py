"""deploy/release-vm scripts: they parse, print the exact hcloud commands and refuse early.

Nothing here runs the real docker or hcloud. A recorder named ``hcloud`` (or ``docker``) is put
first on PATH, so a refusal that reached the CLI would leave a trace in ``calls``.
"""

from __future__ import annotations

import os
import re
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


# ---- Proxmox target (--target pve) and the data-disk guard ---------------------------------------

DEFAULT_STAGES = "lock build export croissant gate plan"
HOST_CMDS = ("qm ", "pct ", "ping ", "install ")
# The build plan's section 3.2 with pve.env.example's values: the example holds the plan's numbers
# and addresses, and placeholders where the plan has the operator's host, user, key VM and keys.
EXAMPLE_ENV = DEPLOY / "pve.env.example"
REQUIRED = (
    "PVE_HOST", "PVE_CI_USER", "PVE_SSHKEYS_FROM_VMID", "PVE_KEY_COMMENTS",
    "PVE_STATIC_IP", "PVE_GATEWAY", "PVE_BRIDGE", "PVE_VMID",
)  # fmt: skip
VM_900_SCRIPT = [
    "set -euo pipefail",
    'qm status 900 >/dev/null 2>&1 && { echo "vmid 900 exists"; exit 1; }',
    "qm list; pct list",
    'ping -c1 -W1 10.0.0.90 >/dev/null && { echo "10.0.0.90 answers"; exit 1; }',
    "ip neigh show 10.0.0.90                                   # expect nothing or FAILED",
    "pvesm status | grep -E '^(local|local-lvm) '              # both active",
    "stat -c %s /var/lib/vz/template/iso/debian-12-genericcloud-amd64.qcow2   # 349634560",
    "printf '%b\\n' \"$(qm config 0 | sed -n 's/^sshkeys: //p' | sed 's/%/\\\\x/g')\" | "
    "grep -v '^$' > /root/vm900-keys.pub",
    "ssh-keygen -lf /root/vm900-keys.pub                       # exactly 2: "
    "root@pve.example.internal, deploy-key-example",
    "install -m 0644 marinedata-build.yaml /var/lib/vz/snippets/marinedata-build.yaml",
    "qm create 900 --name marinedata-build \\",
    "  --description 'Disposable marine-data v1.1.0 build. Raw qm, not in OpenTofu by design. "
    "Destroy after publish.' \\",
    "  --ostype l26 --cpu host --sockets 1 --cores 6 --memory 16384 --scsihw virtio-scsi-single \\",
    "  --agent enabled=1 --serial0 socket --vga serial0 --onboot 0 --net0 "
    "virtio,bridge=vmbr1,firewall=0,mtu=1",
    "qm importdisk 900 /var/lib/vz/template/iso/debian-12-genericcloud-amd64.qcow2 local-lvm",
    "qm config 900 | grep '^unused0: local-lvm:vm-900-disk-0'  # the name the next line uses",
    "qm set 900 --scsi0 local-lvm:vm-900-disk-0,discard=on,iothread=1,ssd=1,backup=0",
    "qm resize 900 scsi0 32G",
    "qm set 900 --scsi1 local-lvm:100,discard=on,iothread=1,ssd=1,backup=0",
    "qm set 900 --ide2 local-lvm:cloudinit --boot order=scsi0",
    "qm set 900 --ciuser admin --sshkeys /root/vm900-keys.pub \\",
    "  --ipconfig0 ip=10.0.0.90/24,gw=10.0.0.2 --nameserver 1.1.1.1 --searchdomain localdomain",
    "qm set 900 --cicustom vendor=local:snippets/marinedata-build.yaml",
    "qm cloudinit dump 900 user | grep -c 'ssh-'               # 2",
    "qm start 900",
    "timeout 900 sh -c 'until qm guest cmd 900 ping 2>/dev/null; do sleep 15; done'",
    "qm guest exec 900 --timeout 600 -- cloud-init status --wait",
    "qm guest exec 900 -- sh -c 'ip -br link; ls -l /dev/disk/by-id/; lsblk -o "
    "NAME,SIZE,TYPE,PTTYPE'",
]
FAKE_LSBLK = r"""#!/bin/sh
case "$2" in
  PTTYPE) printf '%s\n' "${FAKE_PTTYPE-}" ;;
  TYPE) printf 'disk\n'; for kid in ${FAKE_KIDS-}; do printf '%s\n' "$kid"; done ;;
  MOUNTPOINTS) printf '%s\n' "${FAKE_MOUNTS-}" ;;
esac
"""
FAKES = {
    "lsblk": FAKE_LSBLK,
    "blkid": "#!/bin/sh\nprintf '%s\\n' \"${FAKE_FS-}\"\n",
    "mountpoint": "#!/bin/sh\nexit 1\n",
}


def pve(env, out, over=None, *extra, env_file=EXAMPLE_ENV):
    """create.sh --target pve with the example values; ``over`` replaces or adds flag/value pairs"""
    flags = args_with({"--release-id": "v1.1.0", **(over or {})})
    file_args = ["--env-file", str(env_file)] if env_file else []
    return create(env, "--target", "pve", *flags, "--out-dir", str(out), *file_args, *extra)


@pytest.fixture
def disk(tmp_path):
    """run(extra, unset): bootstrap.sh in dry-run mode on fake lsblk, blkid and mountpoint."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in FAKES.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)

    def run(extra=None, unset=()):
        env = {
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "LC_ALL": "C",
            "BOOTSTRAP_DRY_RUN": "1",
            "DATA_DEV": "/dev/null",
            "SETTINGS_ENV": str(tmp_path / "settings.env"),
            "REPO_DIR": str(tmp_path / "no-checkout"),
            **GOOD,
            **(extra or {}),
        }
        for key in unset:
            env.pop(key)
        return subprocess.run(
            [BASH, str(DEPLOY / "bootstrap.sh")],
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            check=False,
        )

    return run


def test_pve_renders_the_plan_script_and_a_settings_only_snippet(sandbox, tmp_path):
    env, calls = sandbox
    out = tmp_path / "out"
    done = pve(env, out)
    assert done.returncode == 0, done.stderr
    assert "PROXMOX TARGET" in done.stdout
    script = out / "vm-900.sh"
    lines = script.read_text().splitlines()
    assert lines[0] == "#!/usr/bin/env bash"
    assert lines[-len(VM_900_SCRIPT) :] == VM_900_SCRIPT
    assert [x for x in lines if x.startswith(HOST_CMDS)] == [
        x for x in VM_900_SCRIPT if x.startswith(HOST_CMDS)
    ]
    snippet = (out / "marinedata-build.yaml").read_text()
    assert f"      MARINEDATA_REF={SHA}\n" in snippet
    for text in (script.read_text(), snippet, done.stdout):
        assert not any(banned in text for banned in ("HF_TOKEN=", "SECRET", "AKIA"))
    assert os.access(script, os.X_OK)
    assert subprocess.run([BASH, "-n", str(script)], check=False).returncode == 0
    assert not calls.exists()


def test_pve_snippet_is_cloud_config_with_the_settings(sandbox, tmp_path):
    yaml = pytest.importorskip("yaml")
    env, _ = sandbox
    out = tmp_path / "out"
    over = {"--lock-file": "releases/a/b.yaml", "--split-map-file": "releases/a/map.json"}
    assert pve(env, out, over, "--build-args", "--decon").returncode == 0
    doc = yaml.safe_load((out / "marinedata-build.yaml").read_text())
    assert doc["write_files"][0]["path"] == "/etc/marinedata/settings.env"
    assert doc["write_files"][0]["content"].splitlines() == [
        f"MARINEDATA_REF={SHA}",
        "RELEASE_ID=v1.1.0",
        "FLAVOUR=open",
        "LOCK_FILE=releases/a/b.yaml",
        "SPLIT_MAP_FILE=releases/a/map.json",
        "RELEASE_BUILD_ARGS=--decon",
        "DATA_DEV=/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_drive-scsi1",
    ]
    assert doc["runcmd"][1][:3] == ["git", "clone", "--no-checkout"]
    assert doc["runcmd"][2][-1].endswith('checkout --detach "$MARINEDATA_REF"')


def test_pve_flags_reach_the_script(sandbox, tmp_path):
    env, _ = sandbox
    out = tmp_path / "out"
    over = {
        "--vmid": "901", "--name": "md-nc", "--ip": "10.0.0.91/24", "--gw": "10.0.0.1",
        "--bridge": "vmbr2", "--storage": "local-zfs", "--cores": "4", "--memory": "8192",
        "--data-gb": "60", "--keys-file": "/root/k.pub", "--host": "host.test", "--ci-user": "ops",
        "--sshkeys-from-vmid": "7", "--key-comments": "a@h, b, c", "--nameserver": "9.9.9.9",
    }  # fmt: skip
    done = pve(env, out, over)
    assert done.returncode == 0, done.stderr
    text = (out / "vm-901.sh").read_text()
    for needle in (
        "qm create 901 --name md-nc",
        "--cores 4 --memory 8192",
        "bridge=vmbr2,firewall=0",
        "--scsi1 local-zfs:60,",
        "ip=10.0.0.91/24,gw=10.0.0.1",
        "ping -c1 -W1 10.0.0.91 ",
        "> /root/k.pub",
        "install -m 0644 md-nc.yaml /var/lib/vz/snippets/md-nc.yaml",
        "qm config 7 | sed",
        "--ciuser ops --sshkeys /root/k.pub",
        "--nameserver 9.9.9.9 --searchdomain",
        "# exactly 3: a@h, b, c",
        "grep -c 'ssh-'               # 3",
    ):
        assert needle in text
    assert "ssh -J root@host.test ops@10.0.0.91 " in done.stdout
    assert "900" not in text.replace("timeout 900", "")
    assert (out / "md-nc.yaml").exists()


def test_pve_description_follows_the_release_id_and_the_wait_loop_probes_with_guest_cmd(
    sandbox, tmp_path
):
    env, _ = sandbox
    out = tmp_path / "out"
    done = pve(env, out, {"--release-id": "v1.2.0"})
    assert done.returncode == 0, done.stderr
    text = (out / "vm-900.sh").read_text()
    assert "--description 'Disposable marine-data v1.2.0 build. " in text
    assert "v1.1.0" not in text
    # PVE 9.2 has no `qm guest ping`; the guest agent is probed with `qm guest cmd <vmid> ping`.
    assert "qm guest ping" not in text
    assert "until qm guest cmd 900 ping 2>/dev/null;" in text


@pytest.mark.parametrize("ref", ["deadbeef", "main", SHA[:-1], SHA + "0", SHA.upper()])
def test_pve_refuses_a_ref_that_is_not_40_hex(sandbox, tmp_path, ref):
    env, calls = sandbox
    out = tmp_path / "out"
    done = pve(env, out, {"--ref": ref})
    assert done.returncode != 0
    assert "40-character" in done.stderr
    assert not out.exists()
    assert not calls.exists()


def test_pve_refuses_run(sandbox, tmp_path):
    env, calls = sandbox
    out = tmp_path / "out"
    done = pve(env, out, None, "--run", "--yes")
    assert done.returncode != 0
    assert "--run is not supported" in done.stderr
    assert not out.exists()
    assert not calls.exists()


def test_targets_refuse_each_others_flags(sandbox, tmp_path):
    env, _ = sandbox
    mixed = pve(env, tmp_path / "a", None, "--ssh-key", "k")
    assert mixed.returncode != 0
    assert "--target hetzner" in mixed.stderr
    other = create(env, *args_with(), "--vmid", "901", "--out-dir", str(tmp_path / "b"))
    assert other.returncode != 0
    assert "--target pve only" in other.stderr
    bogus = create(env, "--target", "azure", *args_with(), "--out-dir", str(tmp_path / "c"))
    assert bogus.returncode != 0
    assert not (tmp_path / "a").exists()
    assert not (tmp_path / "b").exists()


def test_explicit_hetzner_target_changes_nothing(sandbox, tmp_path):
    env, _ = sandbox
    ud = str(tmp_path / "ud")
    plain = create(env, *args_with(), "--out-dir", ud)
    explicit = create(env, "--target", "hetzner", *args_with(), "--out-dir", ud)
    assert plain.returncode == explicit.returncode == 0
    assert plain.stdout == explicit.stdout


@pytest.mark.parametrize(
    ("fake", "reason"),
    [
        ({"FAKE_PTTYPE": "gpt"}, "partition table (gpt)"),
        ({"FAKE_KIDS": "part part"}, "partitions"),
        ({"FAKE_MOUNTS": "/mnt/other"}, "mounted"),
        ({"FAKE_MOUNTS": "/"}, "backs /"),
    ],
)
def test_dry_run_refuses_a_disk_that_is_not_blank(disk, fake, reason):
    done = disk(fake)
    assert done.returncode == 1
    assert "refusing to format /dev/null" in done.stderr
    assert reason in done.stderr


def test_dry_run_formats_a_blank_disk_and_keeps_an_existing_filesystem(disk):
    blank = disk()
    assert blank.returncode == 0, blank.stderr
    assert "would be formatted ext4 (-m 0)" in blank.stdout
    used = disk({"FAKE_FS": "ext4"})
    assert used.returncode == 0, used.stderr
    assert "never reformatted" in used.stdout


def test_dry_run_stage_lists(disk):
    assert f"stages=[{DEFAULT_STAGES}]" in disk().stdout
    empty = disk({"STAGES": ""})
    assert empty.returncode == 0
    assert "stages=[]" in empty.stdout
    assert "nothing waits for" in empty.stdout
    assert "the run waits for" in disk({"STAGES": "croissant build"}).stdout
    assert "does not wait" in disk({"STAGES": "croissant gate"}).stdout
    assert "unknown stage 'upload'" in disk({"STAGES": "upload"}).stderr


def test_data_disk_resolution(disk, tmp_path):
    (tmp_path / "settings.env").write_text("RELEASE_ID=v1\nDATA_DEV=/dev/null\n")
    assert "data disk /dev/null" in disk(unset=("DATA_DEV",)).stdout
    (tmp_path / "settings.env").unlink()
    none = disk(unset=("DATA_DEV",))
    assert none.returncode == 1
    for name in ("DATA_DEV", "scsi-0HC_Volume_", "scsi-0QEMU_QEMU_HARDDISK_drive-scsi1"):
        assert name in none.stderr
    assert "DATA_DEV" in disk({"DATA_DEV": "/etc/passwd"}).stderr


@pytest.mark.parametrize("flavour", ["open", "nc"])
def test_croissant_stage_uses_the_flavours_repo_id(tmp_path, flavour):
    yaml = pytest.importorskip("yaml")
    root = DEPLOY.parents[1]
    repo = yaml.safe_load((root / "registry" / "flavours.yaml").read_text())
    repo_id = repo["flavours"][flavour]["repo_id"]
    wrapper = tmp_path / "wrapper"
    wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    wrapper.chmod(0o755)
    done = bootstrap(
        f"RUN_WRAPPER={wrapper}; derive_paths; stage_croissant",
        {**GOOD, "FLAVOUR": flavour, "REPO_DIR": str(root)},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (
        f"release croissant /data/export/v1.1/{flavour} --release-dir /data/releases/v1.1/{flavour}"
        f" --citation /app/CITATION.cff --repo-id {repo_id}"
    )


def test_wrapper_keeps_scratch_on_the_data_disk_and_caps_memory(tmp_path):
    wrapper = tmp_path / "marinedata-run"
    done = bootstrap(
        f"RUN_WRAPPER={wrapper}; IMAGE=marinedata:t; write_wrapper; cat {wrapper}", GOOD
    )
    assert done.returncode == 0, done.stderr
    assert "-e TMPDIR=/data/tmp -e TORCH_HOME=/data/torch --memory 14g" in done.stdout


def env_values(path):
    """The KEY=VALUE lines of an env file as a dict (quotes stripped, comments skipped)."""
    pairs = (x.split("=", 1) for x in path.read_text().splitlines() if x.startswith("PVE_"))
    return {key: value.strip("\"'") for key, value in pairs}


def example_env(tmp_path, drop=(), extra=""):
    """A copy of pve.env.example without the keys in ``drop``, plus the lines in ``extra``."""
    lines = [x for x in EXAMPLE_ENV.read_text().splitlines() if x.split("=", 1)[0] not in drop]
    path = tmp_path / "pve.env"
    path.write_text("\n".join(lines) + "\n" + extra)
    return path


def test_pve_example_env_is_complete_and_its_optional_lines_are_valid(sandbox, tmp_path):
    env, _ = sandbox
    text = EXAMPLE_ENV.read_text()
    assert set(env_values(EXAMPLE_ENV)) == set(REQUIRED)
    lines = [x[2:] if x.startswith("# PVE_") else x for x in text.splitlines()]
    everything = tmp_path / "all.env"
    everything.write_text("\n".join(lines) + "\n")
    assert pve(env, tmp_path / "out", env_file=everything).returncode == 0


@pytest.mark.parametrize("key", REQUIRED)
def test_pve_refuses_to_render_while_a_required_value_is_unset(sandbox, tmp_path, key):
    env, calls = sandbox
    out = tmp_path / "out"
    done = pve(env, out, env_file=example_env(tmp_path, drop=(key,)))
    assert done.returncode != 0
    assert f"{key} (--" in done.stderr
    assert "Nothing is guessed" in done.stderr
    assert not out.exists()
    assert not calls.exists()


def test_pve_with_no_values_at_all_names_every_required_one(sandbox, tmp_path):
    env, _ = sandbox
    done = pve(env, tmp_path / "out", env_file=None)
    assert done.returncode != 0
    assert all(key in done.stderr for key in REQUIRED)
    assert not (tmp_path / "out").exists()


def test_pve_values_come_from_the_environment_alone(sandbox, tmp_path):
    env, _ = sandbox
    out = tmp_path / "out"
    done = pve(env | env_values(EXAMPLE_ENV), out, env_file=None)
    assert done.returncode == 0, done.stderr
    assert (out / "vm-900.sh").read_text().splitlines()[4:32] == VM_900_SCRIPT


def test_pve_value_precedence_is_flag_then_environment_then_file(sandbox, tmp_path):
    env, _ = sandbox

    def ciuser(tag, extra_env, *flags):
        out = tmp_path / tag
        done = pve({**env, **extra_env}, out, None, *flags)
        assert done.returncode == 0, done.stderr
        return re.search(r"--ciuser (\S+)", (out / "vm-900.sh").read_text()).group(1)

    assert ciuser("file", {}) == "admin"
    assert ciuser("env", {"PVE_CI_USER": "fromenv"}) == "fromenv"
    assert ciuser("flag", {"PVE_CI_USER": "fromenv"}, "--ci-user", "fromflag") == "fromflag"


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("FOO=1", "unknown key FOO"),
        ("PVE_HOST host.example", "expected KEY=VALUE"),
        ("PVE_CI_USER=$(touch {marker})", "PVE_CI_USER"),
        ("PVE_CI_USER=`touch {marker}`", "PVE_CI_USER"),
    ],
)
def test_pve_env_file_is_read_and_never_executed(sandbox, tmp_path, line, message):
    env, calls = sandbox
    marker = tmp_path / "executed"
    path = example_env(tmp_path, drop=("PVE_CI_USER",), extra=line.format(marker=marker) + "\n")
    done = pve(env, tmp_path / "out", env_file=path)
    assert done.returncode != 0
    assert message in done.stderr
    assert not marker.exists()
    assert not (tmp_path / "out").exists()
    assert not calls.exists()


def test_pve_env_file_that_is_missing_is_refused(sandbox, tmp_path):
    env, _ = sandbox
    done = pve(env, tmp_path / "out", env_file=tmp_path / "nope.env")
    assert done.returncode != 0
    assert "not a readable file" in done.stderr


def test_shipped_files_name_no_operator_host_user_or_key():
    # Spelled in pieces so this file stays clean when the repo is grepped for the same words.
    words = [
        "rs" + "-server",
        "rs" + "admin",
        "gitlab" + "-user",
        r"\brs" + r"-[12]\b",
        "reef" + r"\.support",
    ]
    pattern = re.compile("|".join(words))
    for name in ("README.md", "bootstrap.sh", "create.sh", "pve.env.example"):
        hit = pattern.search((DEPLOY / name).read_text())
        assert hit is None, f"{name}: {hit.group(0)}"


GIT = shutil.which("git", path="/usr/bin:/bin")
needs_git = pytest.mark.skipif(GIT is None, reason="git is not installed")


def git(cwd, *args):
    config = [
        "-c",
        "user.name=t",
        "-c",
        "user.email=t@example.invalid",
        "-c",
        "commit.gpgsign=false",
    ]
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd), "LC_ALL": "C"}
    done = subprocess.run(
        [GIT, *config, *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


@pytest.fixture
def origin(tmp_path):
    """(url, first, second): two commits in a local repo; a.txt differs between them, c.txt not."""
    repo = tmp_path / "origin"
    repo.mkdir()
    git(repo, "init", "-q")
    shas = []
    for body in ("one", "two"):
        (repo / "a.txt").write_text(body)
        (repo / "c.txt").write_text("same")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", body)
        shas.append(git(repo, "rev-parse", "HEAD"))
    return f"file://{repo}", *shas


def clone_at(url, repo, sha):
    """bootstrap.sh's clone_repo for ``sha``: a first run clones, a re-run moves the checkout"""
    env = {**GOOD, "REPO_URL": url, "REPO_DIR": str(repo), "MARINEDATA_REF": sha}
    return bootstrap("clone_repo", env)


def fake_docker(tmp_path):
    """(env, calls): a PATH with a ``docker`` that only records its arguments."""
    bin_dir = tmp_path / "dockerbin"
    bin_dir.mkdir()
    calls = tmp_path / "docker.calls"
    stub = bin_dir / "docker"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{calls}"\n')
    stub.chmod(0o755)
    return {"PATH": f"{bin_dir}:/usr/bin:/bin"}, calls


@needs_git
def test_rerun_moves_the_checkout_to_the_new_ref_and_leaves_untracked_files(origin, tmp_path):
    url, first, second = origin
    repo = tmp_path / "opt" / "marine-data"
    one = clone_at(url, repo, first)
    assert one.returncode == 0, one.stderr
    assert f"checked out {first} = {first}" in one.stdout
    (repo / "scratch.txt").write_text("keep me\n")
    two = clone_at(url, repo, second)
    assert two.returncode == 0, two.stderr
    assert f"moved {repo} from {first[:12]} to {second[:12]}" in two.stdout
    assert git(repo, "rev-parse", "HEAD") == second
    assert (repo / "a.txt").read_text() == "two"
    assert (repo / "scratch.txt").read_text() == "keep me\n"


@needs_git
@pytest.mark.parametrize("name", ["a.txt", "c.txt"])
def test_rerun_refuses_local_changes_and_discards_nothing(origin, tmp_path, name):
    url, first, second = origin
    repo = tmp_path / "opt" / "marine-data"
    assert clone_at(url, repo, first).returncode == 0
    (repo / name).write_text("local edit\n")  # a.txt would conflict; c.txt git would carry along
    done = clone_at(url, repo, second)
    assert done.returncode != 0
    assert "local changes" in done.stderr
    assert "never resets or cleans" in done.stderr
    assert git(repo, "rev-parse", "HEAD") == first
    assert (repo / name).read_text() == "local edit\n"


@needs_git
def test_rerun_with_the_same_ref_keeps_a_local_edit(origin, tmp_path):
    url, first, _ = origin
    repo = tmp_path / "opt" / "marine-data"
    assert clone_at(url, repo, first).returncode == 0
    (repo / "a.txt").write_text("fixed on the VM\n")
    again = clone_at(url, repo, first)
    assert again.returncode == 0, again.stderr
    assert (repo / "a.txt").read_text() == "fixed on the VM\n"


@needs_git
def test_dry_run_says_where_the_checkout_is_and_where_it_would_go(disk, origin, tmp_path):
    url, first, second = origin
    repo = tmp_path / "opt" / "marine-data"
    base = {"REPO_URL": url, "REPO_DIR": str(repo)}
    absent = disk({**base, "MARINEDATA_REF": first})
    assert absent.returncode == 0, absent.stderr
    assert "no checkout at" in absent.stdout
    assert not repo.exists()
    assert clone_at(url, repo, first).returncode == 0
    same = disk({**base, "MARINEDATA_REF": first})
    assert f"{repo} is already at {first}" in same.stdout
    moving = disk({**base, "MARINEDATA_REF": second, "STAGES": "gate plan"})
    assert moving.returncode == 0, moving.stderr
    assert f"is at {first}: it would be fetched and checked out at {second}" in moving.stdout
    assert "stages=[gate plan]" in moving.stdout
    assert "does not wait" in moving.stdout
    (repo / "a.txt").write_text("local edit\n")
    dirty = disk({**base, "MARINEDATA_REF": second})
    assert dirty.returncode == 1
    assert "refusing to move" in dirty.stderr
    assert (repo / "a.txt").read_text() == "local edit\n"
    assert git(repo, "rev-parse", "HEAD") == first


def test_rerun_rebuilds_the_image_under_a_tag_with_the_new_commit(tmp_path):
    env, calls = fake_docker(tmp_path)
    one, two = "1" * 40, "2" * 40
    snippet = f"COMMIT={one}; build_image; COMMIT={two}; build_image"
    done = bootstrap(snippet, {**GOOD, **env, "REPO_DIR": "/opt/x"})
    assert done.returncode == 0, done.stderr
    assert calls.read_text().splitlines() == [
        f"build --tag marinedata:v1.1-{one[:12]} /opt/x",
        f"build --tag marinedata:v1.1-{two[:12]} /opt/x",
    ]


def test_build_passes_the_lock_and_split_map_as_app_paths(tmp_path):
    wrapper = tmp_path / "wrapper"
    wrapper.write_text('#!/bin/sh\nprintf "%s\\n" "$*"\n')
    wrapper.chmod(0o755)
    lock = "releases/marine-data/v1.1.0/open/release.lock.yaml"
    split_map = "releases/marine-data/v1.1.0/SPLIT_MAP.json"
    done = bootstrap(
        f"RUN_WRAPPER={wrapper}; derive_paths; stage_build",
        {**GOOD, "LOCK_FILE": lock, "SPLIT_MAP_FILE": split_map},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (
        f"release build --release v1.1 --flavour open --lock /app/{lock}"
        f" --split-map /app/{split_map} --out /data"
    )


def test_wrapper_mounts_the_checkouts_releases_read_only_at_app_releases(tmp_path):
    env, calls = fake_docker(tmp_path)
    repo = tmp_path / "checkout"
    (repo / "releases").mkdir(parents=True)
    wrapper = tmp_path / "marinedata-run"
    write = f"RUN_WRAPPER={wrapper}; IMAGE=marinedata:t; write_wrapper"
    assert bootstrap(write, {**GOOD, "REPO_DIR": str(repo)}).returncode == 0
    mount = f"-v {repo}/releases:/app/releases:ro"
    assert mount in wrapper.read_text()
    run = subprocess.run([str(wrapper), "release", "gate", "--help"], env=env, check=False)
    assert run.returncode == 0
    assert mount in calls.read_text()
    # A ref without releases/ gets no mount, so docker never creates the folder in the checkout.
    (repo / "releases").rmdir()
    calls.unlink()
    assert subprocess.run([str(wrapper), "release", "gate"], env=env, check=False).returncode == 0
    assert "/app/releases" not in calls.read_text()
    assert not (repo / "releases").exists()


EXPORT, REL = "/data/export/v1.1/open", "/data/releases/v1.1/open"
YUNET = "/opt/models/face_detection_yunet_2023mar.onnx"


def fake_wrapper(tmp_path, *, fail_all=False):
    """Prints its arguments; answers ``release gate --help``; can fail the ``--all`` gate."""
    wrapper = tmp_path / "wrapper"
    fail = "  *--all*) printf '%s\\n' \"$*\"; exit 1 ;;\n" if fail_all else ""
    wrapper.write_text(
        '#!/bin/sh\ncase "$*" in\n'
        '  "release gate --help") echo "usage: release gate --all --require-all --v11" ;;\n'
        f"{fail}  *) printf '%s\\n' \"$*\" ;;\nesac\n"
    )
    wrapper.chmod(0o755)
    return wrapper


def run_stage(tmp_path, stage, extra=None, **kwargs):
    wrapper = fake_wrapper(tmp_path, **kwargs)
    return bootstrap(f"RUN_WRAPPER={wrapper}; derive_paths; {stage}", {**GOOD, **(extra or {})})


def test_card_stage_cites_the_repo_citation_file(tmp_path):
    done = run_stage(tmp_path, "stage_card")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == (
        f"--py -m marinedata.hf_card --release-dir {REL}"
        " --summary /data/export/v1.1/open.summary.json"
        f" --out {EXPORT} --flavour open --citation /app/CITATION.cff"
    )
    # hf_export writes no CITATION.cff (hf_card renders it): the card cannot cite one in the export
    assert f"{EXPORT}/CITATION.cff" not in done.stdout
    assert "stage_card\n}" in (DEPLOY / "bootstrap.sh").read_text()  # stage_export ends with it


def test_gate_stage_runs_all_then_v11_on_the_same_release_dir(tmp_path):
    done = run_stage(tmp_path, "stage_gate", {"GATE_V11_ARGS": "--v1-dir /data/v1.0"})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        f"release gate {EXPORT} --all --require-all --flavour open --release-dir {REL}"
        " --v1-dir /data/v1.0",
        f"release gate {EXPORT} --v11 --flavour open --release-dir {REL} --v1-dir /data/v1.0",
    ]


def test_gate_stage_still_runs_v11_when_the_v1_0_gates_fail(tmp_path):
    done = run_stage(tmp_path, "stage_gate", fail_all=True)
    assert done.returncode != 0
    assert [line.split()[3] for line in done.stdout.splitlines()] == ["--all", "--v11"]


P7_GATE_ARGS = (
    "--v1-dir /data/v1.0-repo --privacy-review /app/registry/privacy/review.yaml"
    " --hidden-test /data/checks/hidden-test-manifest.parquet"
    " --benchmark /app/registry/benchmarks/manifests"
)
ALL_CALL = f"release gate {EXPORT} --all --require-all --flavour open --release-dir {REL}"


def test_gate_stage_gives_the_all_call_the_v1_dir_and_no_other_v11_flag(tmp_path):
    """The rehearsal's --all call lacked --v1-dir (card: 4 extra_config_section, masks: no mask
    rows). Any other v1.1.0 flag would route that call to the --v11 gate set, so the v1.0 gates
    would not run."""
    done = run_stage(tmp_path, "stage_gate", {"GATE_V11_ARGS": P7_GATE_ARGS})
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        f"{ALL_CALL} --v1-dir /data/v1.0-repo",
        f"release gate {EXPORT} --v11 --flavour open --release-dir {REL} {P7_GATE_ARGS}",
    ]


@pytest.mark.parametrize(
    "args",
    [
        "--v1-dir=/data/v1.0",
        "--rehearsal --v1-dir /data/v1.0 --nc-id x",
        "--nc-id x --v1-dir=/old --v1-dir /data/v1.0",
    ],
)
def test_gate_stage_finds_the_v1_dir_in_any_position_and_spelling(tmp_path, args):
    done = run_stage(tmp_path, "stage_gate", {"GATE_V11_ARGS": args})
    assert done.stdout.splitlines()[0] == f"{ALL_CALL} --v1-dir /data/v1.0"


@pytest.mark.parametrize("args", ["", "--hidden-test /m.parquet --benchmark /b", "--v1-dir"])
def test_gate_stage_without_a_v1_dir_keeps_the_all_call_as_it_was(tmp_path, args):
    done = run_stage(tmp_path, "stage_gate", {"GATE_V11_ARGS": args})
    assert done.stdout.splitlines()[0] == ALL_CALL


def test_both_gate_calls_in_bootstrap_carry_the_v1_dir():
    text = (DEPLOY / "bootstrap.sh").read_text()
    body = text.split("\nstage_gate() {", 1)[1].split("\nstage_plan() {", 1)[0]
    calls = [c for c in body.split("|| rc=1") if '"$RUN_WRAPPER" release gate "$EXPORT_DIR"' in c]
    assert len(calls) == 2
    assert "--all --require-all" in calls[0] and '"${v1dir[@]}"' in calls[0]
    assert "--v11" in calls[1] and '"${v11[@]}"' in calls[1]  # GATE_V11_ARGS, --v1-dir DIR included


def test_plan_stage_forwards_the_plan_args(tmp_path):
    plain = run_stage(tmp_path, "stage_plan")
    assert plain.stdout.strip() == (
        f"release publish --release-id v1.1 --flavour open --dir {EXPORT} --release-dir {REL}"
        " --lock /app/release.lock.yaml"
    )
    delta = run_stage(tmp_path, "stage_plan", {"PLAN_ARGS": "--v1-dir /data/v1.0-repo"})
    assert delta.stdout.strip().endswith("--lock /app/release.lock.yaml --v1-dir /data/v1.0-repo")


def test_wrapper_passes_the_verified_yunet_model_path_and_the_image_has_it(tmp_path):
    wrapper = tmp_path / "marinedata-run"
    done = bootstrap(
        f"RUN_WRAPPER={wrapper}; IMAGE=marinedata:t; write_wrapper; cat {wrapper}", GOOD
    )
    assert done.returncode == 0, done.stderr
    assert f"-e MARINEDATA_YUNET_MODEL={YUNET}" in done.stdout
    assert YUNET in (DEPLOY.parents[1] / "Dockerfile").read_text()


def test_next_steps_print_the_per_part_publish_loop():
    done = bootstrap("COMMIT=abc; LOG_DIR=/data/logs; derive_paths; print_next_steps", GOOD)
    assert done.returncode == 0, done.stderr
    out = done.stdout
    assert "for part in $(marinedata-run release publish --release-id v1.1" in out
    assert "--list-parts); do" in out and '--go --part "$part" || break' in out
    assert out.count("--go") == 1 and "done\n" in out


def test_stage_export_tells_hf_export_the_release(tmp_path):
    """An additive export (--v1-dir in EXPORT_ARGS) names its images shards after the release."""
    calls = tmp_path / "calls"
    wrapper = tmp_path / "wrapper"
    wrapper.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" >> "{calls}"\n')
    wrapper.chmod(0o755)
    snippet = (
        f"RUN_WRAPPER={wrapper}; REL_DIR={tmp_path}/rel; EXPORT_DIR={tmp_path}/export; "
        f"SUMMARY={tmp_path}/s.json; FLAVOUR=open; RELEASE_ID=v1.2.0; "
        'EXPORT_ARGS="--v1-dir /v1"; install() { :; }; stage_card() { :; }; stage_export'
    )
    done = bootstrap(snippet)
    assert done.returncode == 0, done.stderr
    words = calls.read_text().split()
    assert words[words.index("--release-id") + 1] == "v1.2.0"
    assert words[words.index("--flavour") + 1] == "open"
    assert words[words.index("--v1-dir") + 1] == "/v1"  # the extra flags still follow
