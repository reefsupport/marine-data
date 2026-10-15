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


# ---- Proxmox target (--target pve) and the data-disk guard ---------------------------------------

DEFAULT_STAGES = "lock build export croissant gate plan"
HOST_CMDS = ("qm ", "pct ", "ping ", "install ")
# The build plan's section 3.2: what ``--target pve`` renders into out/vm-900.sh with the defaults.
VM_900_SCRIPT = [
    "set -euo pipefail",
    'qm status 900 >/dev/null 2>&1 && { echo "vmid 900 exists"; exit 1; }',
    "qm list; pct list",
    'ping -c1 -W1 10.0.0.90 >/dev/null && { echo "10.0.0.90 answers"; exit 1; }',
    "ip neigh show 10.0.0.90                                   # expect nothing or FAILED",
    "pvesm status | grep -E '^(local|local-lvm) '              # both active",
    "stat -c %s /var/lib/vz/template/iso/debian-12-genericcloud-amd64.qcow2   # 349634560",
    "printf '%b\\n' \"$(qm config 103 | sed -n 's/^sshkeys: //p' | sed 's/%/\\\\x/g')\" | "
    "grep -v '^$' > /root/vm900-keys.pub",
    "ssh-keygen -lf /root/vm900-keys.pub                       # exactly 2: "
    "root@rs-server-2, gitlab-user-20260213",
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
    "qm set 900 --ciuser rsadmin --sshkeys /root/vm900-keys.pub \\",
    "  --ipconfig0 ip=10.0.0.90/24,gw=10.0.0.2 --nameserver 1.1.1.1 --searchdomain localdomain",
    "qm set 900 --cicustom vendor=local:snippets/marinedata-build.yaml",
    "qm cloudinit dump 900 user | grep -c 'ssh-'               # 2",
    "qm start 900",
    "timeout 900 sh -c 'until qm guest ping 900 2>/dev/null; do sleep 15; done'",
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


def pve(env, out, over=None, *extra):
    """create.sh --target pve with the default flags; ``over`` replaces or adds flag/value pairs."""
    flags = args_with({"--release-id": "v1.1.0", **(over or {})})
    return create(env, "--target", "pve", *flags, "--out-dir", str(out), *extra)


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
        "--data-gb": "60", "--keys-file": "/root/k.pub",
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
    ):
        assert needle in text
    assert "900" not in text.replace("timeout 900", "")
    assert (out / "md-nc.yaml").exists()


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
