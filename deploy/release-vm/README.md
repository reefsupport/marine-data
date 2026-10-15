# Release build VM (Hetzner or Proxmox)

A temporary VM builds one release flavour next to the bucket, runs the gates and prints the
`marinedata release publish` plan as a dry run. Everything here is for **you** to run. Nothing in
the repo creates a VM, reads a secret or uploads to the Hub by itself, and `bootstrap.sh` never
passes `--go`.

| File | What it does |
|---|---|
| `create.sh` | `--target hetzner` (the default): writes the cloud-init user-data and prints the `hcloud` commands (`--print-only`, the default: no token, no network); `--run` executes the same commands after a typed confirmation. `--target pve`: writes `out/marinedata-build.yaml` and `out/vm-900.sh` for the Proxmox host and never runs anything. |
| `bootstrap.sh` | Run once as root inside the VM: docker, the data disk at `/data`, a clone of `MARINEDATA_REF`, the image, then the stages below. On Hetzner it is the user-data body; on Proxmox you start it by hand. |

| Stage | What runs | Lands in |
|---|---|---|
| lock | host check: the lock and the split map are in the checkout; prints their sha256 | `/data/logs/lock.log` |
| build | `marinedata release build --release ID --flavour F --lock ... --split-map ... --out /data` | `/data/releases/ID/F/` (`RELEASE.json`, `tasks/`) |
| export | `python -m marinedata.hf_export`, then `python -m marinedata.hf_card` | `/data/export/ID/F/` (`data/`, `README.md`, `LICENSE`) and `/data/export/ID/F.summary.json` |
| croissant | `marinedata release croissant DIR --release-dir ... --citation /app/CITATION.cff --repo-id R` (R is read from `registry/flavours.yaml`) | `croissant.json` in the export folder |
| gate | `marinedata release gate DIR --all --require-all` (only `--masks` until the gates branch is merged) | `GATES.json` in the export folder |
| plan | `marinedata release publish ...` without `--go` | `/data/logs/plan.log` |

`STAGES` defaults to `lock build export croissant gate plan`. Without `croissant` there is no
`croissant.json`, the croissant gate skips and `--require-all` cannot pass. An **empty** `STAGES=""` is
valid and means setup only (packages, data disk, data directory, clone, image, wrapper): no stage
runs and bootstrap does not wait for secrets. `BOOTSTRAP_DRY_RUN=1` prints the resolved stage list and
the data-disk decision and exits before it touches the system.

Every stage logs to `/data/logs/<stage>.log`; `/data/logs/STATUS` has one line per stage outcome
and `/data/logs/bootstrap.log` the whole run. Extra `release build` flags (`--profile`, `--decon`,
...) go through `create.sh --build-args "..."`. Containers run through `marinedata-run` with
`--memory 14g` (an out-of-memory kill takes the container, not sshd), `TMPDIR=/data/tmp` and
`TORCH_HOME=/data/torch`, so scratch space and model weights stay off the small root disk.

## Hetzner or Proxmox

| | Hetzner (`--target hetzner`, the default) | Proxmox (`--target pve`) |
|---|---|---|
| VM | cx32 in hel1, Ubuntu 24.04, 200 GB volume | VM 900 `marinedata-build`: Debian 12 genericcloud, 6 cores, 16 GiB RAM, 32 GB root disk, 100 GB data disk (`scsi1`) |
| `create.sh` writes | user-data with `bootstrap.sh` inlined (32768-byte limit) | `out/marinedata-build.yaml` (a cloud-init vendor snippet: settings and the clone, no secret, no `bootstrap.sh`) and `out/vm-900.sh` (the host script) |
| Who creates it | you, with the printed `hcloud` lines, or `--run` | you, as root on the host: `bash vm-900.sh`. `--run` is refused: there is no API path like hcloud's |
| First boot | cloud-init runs `bootstrap.sh` | cloud-init installs the packages and checks out the ref. Nothing else runs; every phase is started by you |
| Data disk | `/dev/disk/by-id/scsi-0HC_Volume_*` | `/dev/disk/by-id/scsi-0QEMU_QEMU_HARDDISK_drive-scsi1` |
| Teardown | `hcloud server delete`, `hcloud volume delete` | `qm destroy`, then remove the snippet and the keys file |

The Hetzner path is unchanged. `bootstrap.sh` picks the data disk in this order: `DATA_DEV` (the
environment, else the `DATA_DEV=` line of `/etc/marinedata/settings.env`), the one Hetzner volume, the
Proxmox `scsi1` disk. It dies naming all of them when none is there, and when two Hetzner volumes match.
A disk that already holds a filesystem is mounted as it is. A blank one is formatted
`mkfs.ext4 -q -F -m 0` (the container runs as uid 10001, so the 5 % root reserve would be unusable)
and mounted through `UUID=` in `/etc/fstab` (`discard,nofail,defaults`). Before the format it refuses a
disk that has a partition table (`lsblk -no PTTYPE`), has partitions, is mounted (itself or a
partition) or backs `/`. Set `DATA_DEV` only to a disk you have checked.

## Prerequisites

- A release ref: the commit that holds the reviewed `release.lock.yaml` and the split map. Use the
  full sha; a tag can move. `--target pve` refuses anything but 40 hex characters.
- The bucket credentials, operator-provided, in a local file of plain `KEY=value` lines (no quotes,
  no `export`), mode 600, kept out of the repo and out of user-data:
  `MARINEDATA_S3_ACCESS_KEY`, `MARINEDATA_S3_SECRET_KEY`, `MARINEDATA_S3_ENDPOINT`, `MARINEDATA_S3_REGION`.
- `HF_TOKEN` (fine-grained, write access to the two dataset repos only) is needed for step 6 and
  nowhere else. It is never written to a file on the VM.
- Hetzner, for `--run` only: the `hcloud` CLI, an SSH key registered in the Hetzner project
  (`hcloud ssh-key list`) and a **fresh** project API token in `HCLOUD_TOKEN`. The earlier token is
  compromised; revoke the new one when the build is done. Check the server type is still offered
  (`hcloud server-type list`); `--server-type` picks another 4 vCPU / 8 GB type.
- Proxmox: root on the host, the image at `/var/lib/vz/template/iso/debian-12-genericcloud-amd64.qcow2`,
  `local` and `local-lvm` active, the bridge `vmbr1`, the address `10.0.0.90` unused and VM 103's
  `sshkeys` holding the keys that may log in. `vm-900.sh` checks the vmid, the address, the storage and the image
  before it creates anything; it does not check the bridge.

## Cost

Hetzner bills by the hour: `hours x (server + IPv4 EUR/h) + volume GB x (EUR/GB-month) x hours / 730`.
Read the three prices off https://www.hetzner.com/cloud (cx32 in hel1, primary IPv4, volumes) before
running; these figures are an order of magnitude only. With 0.012 EUR/h for server plus IPv4 and
0.045 EUR/GB-month, a 10 h run with the 200 GB volume is 10 x 0.012 + 200 x 0.045 x 10 / 730 = 0.12 + 0.12,
about 0.25 EUR. The 10-07 plan budgets about 1 EUR per build. The real cost is a VM left running, so
teardown (step 8) is part of the build.

Proxmox has no hourly bill. The VM can write up to 132 GiB into the host's shared thin pool (32 GB root
plus 100 GB data); `discard` returns the space as files are deleted and `qm destroy` returns the rest.
Read `pvesm status` before and after.

## Steps on Hetzner

1. **Pick the ref** (full sha) and the release id (for example `v1.1`).
2. **Render and read** what would be created. No token, no network:
   ```bash
   deploy/release-vm/create.sh --release-id v1.1 --flavour open --ref <sha> --ssh-key <name>
   ```
   Open the `user-data.sh` it names: three `export` lines of settings, then `bootstrap.sh`.
3. **Create** the volume and the server (200 GB volume, cx32, hel1, Ubuntu 24.04):
   ```bash
   export HCLOUD_TOKEN=...   # fresh project token
   deploy/release-vm/create.sh --release-id v1.1 --flavour open --ref <sha> --ssh-key <name> --run
   ```
   or paste the printed `hcloud` lines yourself. Run it once per flavour (`open`, then `nc`), each on its own VM.
4. **Copy the secrets file.** `bootstrap.sh` waits up to an hour for `/root/marinedata.env` and then
   checks it (mode 600, owner root, no `HF_TOKEN`). Move it into place in one step so a half-copied file is never seen:
   ```bash
   IP=$(hcloud server ip marinedata-v1-1-open)
   scp ~/secure/marinedata.env root@$IP:/root/.marinedata.env.tmp
   ssh root@$IP 'chmod 600 /root/.marinedata.env.tmp && mv /root/.marinedata.env.tmp /root/marinedata.env'
   ```
5. **Watch.** `ssh root@$IP tail -f /data/logs/bootstrap.log`; `cat /data/logs/STATUS` for the stage outcomes.
   If a stage fails, fix the cause on the VM and re-run only what is left, for example
   `STAGES="gate plan" bash /var/lib/cloud/instance/scripts/part-001` (the rendered user-data; the set-up steps are idempotent).
   Remove `/data/export/ID/F` first to export again.

## Steps on Proxmox

1. **Pick the ref** (the full 40-character sha) and the release id (for example `v1.1.0`).
2. **Render and read.** No token, no network, nothing is executed:
   ```bash
   deploy/release-vm/create.sh --target pve --release-id v1.1.0 --flavour open --ref <40-hex sha>
   ```
   It writes `deploy/release-vm/out/marinedata-build.yaml` and `deploy/release-vm/out/vm-900.sh` (`out/` is
   git-ignored, `--out-dir` moves it) and prints the next commands. Read both files. Flags and defaults:
   `--vmid 900 --name marinedata-build --ip 10.0.0.90/24 --gw 10.0.0.2 --bridge vmbr1 --storage local-lvm
   --image local:iso/debian-12-genericcloud-amd64.qcow2 --cores 6 --memory 16384 --data-gb 100
   --keys-file /root/vm900-keys.pub` (a path on the host). `--lock-file`, `--split-map-file`, `--repo-url`,
   `--build-args` and `--export-args` work as on Hetzner; `--ssh-key`, `--location`, `--server-type`,
   `--volume-size`, `--os-image` and `--yes` belong to Hetzner and are refused here.
3. **Run the host script** as root, in a folder that holds both files:
   ```bash
   ssh root@<host> mkdir -p /root/vm900
   scp deploy/release-vm/out/marinedata-build.yaml deploy/release-vm/out/vm-900.sh root@<host>:/root/vm900/
   ssh root@<host> 'cd /root/vm900 && bash vm-900.sh'
   ```
   It stops at the first failing command (`set -euo pipefail`): the vmid is free, the address does not answer,
   `local` and `local-lvm` are active, the image exists. It writes the keys file from VM 103's `sshkeys` and prints
   their fingerprints: read them. Then it installs the snippet at
   `/var/lib/vz/snippets/marinedata-build.yaml`, creates the VM (`qm create`, `importdisk`, `scsi0` resized to 32G,
   `scsi1` as the data disk, a cloud-init drive, `cicustom vendor=local:snippets/marinedata-build.yaml`), starts it,
   waits for the guest agent and `cloud-init status --wait`, and prints the guest's links and
   `/dev/disk/by-id/`. Read that listing: the data disk must be the `scsi-0QEMU_QEMU_HARDDISK_drive-scsi1` name.
4. **Set up the VM.** Log in through the host and run bootstrap with an empty `STAGES`:
   ```bash
   ssh -J root@<host> rsadmin@10.0.0.90        # then: sudo -i
   set -a; . /etc/marinedata/settings.env; set +a; cd /opt/marine-data
   git rev-parse HEAD                          # must equal MARINEDATA_REF
   STAGES="" bash deploy/release-vm/bootstrap.sh
   ```
5. **Copy the secrets file, then run the stages.** From your machine; the file is written mode 600 and no copy
   stays in the home directory:
   ```bash
   ssh -J root@<host> rsadmin@10.0.0.90 "sudo sh -c 'umask 077; cat > /root/marinedata.env'" < ~/secure/marinedata.env
   ```
   In the VM, as root and in the shell of step 4, run the stages under `nohup` so a dropped ssh does not stop them,
   and watch with bounded reads:
   ```bash
   nohup env STAGES="lock build export croissant gate plan" bash deploy/release-vm/bootstrap.sh > /data/logs/run.out 2>&1 &
   tail -n 5 /data/logs/bootstrap.log; cat /data/logs/STATUS
   ```
   If a stage fails, fix the cause and re-run only what is left (`STAGES="gate plan"`). Remove `/data/export/ID/F` first to
   export again. `qm guest exec 900 -- <cmd>` reaches the VM if ssh does not.

## Steps on both: publish, copy back, tear down

6. **Publish by hand**, after reading `/data/logs/plan.log` and `GATES.json`. The plan is a dry run; `--go` is yours to add:
   ```bash
   rm -f /root/marinedata.env             # the upload needs no S3 secrets
   read -rs HF_TOKEN && export HF_TOKEN
   marinedata-run release publish --release-id v1.1 --flavour open \
     --dir /data/export/v1.1/open --release-dir /data/releases/v1.1/open --lock /app/release.lock.yaml --go
   unset HF_TOKEN
   ```
   It refuses unless every required gate passed, refuses an existing tag, and refuses `v1.0` without
   `--allow-frozen` (v1.0 is frozen while the hackathon runs). The upload is `upload_large_folder`: resumable,
   in batched commits (not one commit), over an allow-list only: `data/**/*.parquet`, the card (`README.md`),
   `croissant.json` and `provenance/*`. `provenance/` is written at upload time: `RELEASE.json` as its public-safe
   projection, `GATES.json` and the lock as they are. The tag (`v1.1`) is created on the head of the branch after the
   last batch and is never moved. `SPLIT_MAP.json`, `DEDUP_GATE.json` and `tasks/` are not uploaded.
7. **Copy back** before deleting anything (`scp -r root@$IP:<path> .` on Hetzner; on Proxmox
   `ssh -J root@<host> rsadmin@10.0.0.90 'sudo tar -C / -cf - <paths without the leading />' > copy-back.tar`):
   `/data/releases/ID/F/RELEASE.json`, `/data/export/ID/F/GATES.json`, `/data/export/ID/F/README.md` (the card),
   `/data/export/ID/F/croissant.json`, `/data/export/ID/F.summary.json`, the lock and the split map
   (`/opt/marine-data/<LOCK_FILE>` and `/opt/marine-data/<SPLIT_MAP_FILE>`, the two paths in the settings), and
   `/data/logs/` (`plan.log`, `STATUS`, the stage logs). Mirroring them and the parquet to the bucket (plan
   section 4.4) is a separate step this README does not automate.
8. **Tear down.** Hetzner: delete the server and the volume, and revoke the project token:
   ```bash
   hcloud server delete marinedata-v1-1-open
   hcloud volume delete marinedata-v1-1-open-data   # destroys /data
   ```
   Proxmox: as root on the host (`create.sh --target pve` prints the same lines with your flag values):
   ```bash
   qm shutdown 900 --timeout 120 || qm stop 900
   qm destroy 900 --purge 1 --destroy-unreferenced-disks 1
   rm -f /var/lib/vz/snippets/marinedata-build.yaml /root/vm900-keys.pub
   ```
   Then `qm list` no longer shows 900 and `pvesm status` shows the thin pool back at its earlier level.

## Verification status

Nothing here has run on a VM, on Hetzner or Proxmox, against the bucket or the Hub. So far only
`tests/test_release_vm_scripts.py` (no root, no docker, no network), `bash -n` and `shellcheck` have run:

- `create.sh --target pve` renders a `vm-900.sh` with the `qm`, `pct`, `ping` and `install` lines of the plan in order, a
  snippet with the 40-hex ref and without `HF_TOKEN=`, `SECRET` or an access key; `--run` and a short ref are refused.
- `bootstrap.sh` with `BOOTSTRAP_DRY_RUN=1` and a fake `lsblk`: an empty `STAGES` resolves to no stage, an unset one to
  the six stages in order, and a disk with a partition table, partitions, a mount or the root is refused.

What only the first run on the Proxmox VM can show (the rehearsal checks these before a real build):

- the image builds with Debian 12's `docker.io` (classic builder) and stays at or under 1.5 GB (`docker image ls marinedata`);
- the data disk appears under the by-id name above, and `tune2fs -l` reports 0 reserved blocks after `mkfs.ext4 -m 0`;
- the guest MTU is 1400 like the bridge (`ip -br link`; `ping -c1 -M do -s 1372 1.1.1.1` from the VM);
- `df -h / /data`: docker's layers fit the 32 GB root disk while `TMPDIR` and `TORCH_HOME` sit on `/data`;
- build time and peak memory (`docker stats`) against the 14 GiB container limit;
- `qm guest ping 900` and `cloud-init status --wait` return on first boot.

Also open:

- `release gate --all --require-all` and the `configs` block in `RELEASE.json` come with the gates branch. Until it
  is merged, the gate stage runs only `--masks` and `release publish` refuses.
- Runbook coverage (`docs/release-runbook.md`): the VM runs step 3 (build), step 6 (export), the card part of
  step 8, the gates and the step 9 dry run. Steps 1-2 are replaced by the frozen split map at the
  release ref; steps 4-5 (dedup gate, decon check) are not separate stages: pass `--decon` and the other
  `release build` flags with `create.sh --build-args`, and expect the `dedup` and `splits` gates from
  `gate --all`. Step 7 is skipped. Export and card have not run on a real corpus.
- The image is built on the VM (x86_64). A local arm64 `make image` is only a smoke test.
