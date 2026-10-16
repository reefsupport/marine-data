# Release build VM (Hetzner or Proxmox)

A temporary VM builds one release flavour next to the bucket, runs the gates and prints the
`marinedata release publish` plan as a dry run. Everything here is for **you** to run. Nothing in
the repo creates a VM, reads a secret or uploads to the Hub by itself, and `bootstrap.sh` never
passes `--go`.

| File | What it does |
|---|---|
| `create.sh` | `--target hetzner` (the default): writes the cloud-init user-data and prints the `hcloud` commands (`--print-only`, the default: no token, no network); `--run` executes the same commands after a typed confirmation. `--target pve`: writes `out/<name>.yaml` and `out/vm-<VMID>.sh` for the Proxmox host and never runs anything. The host, the cloud-init user, the VM id, the address and the key names are yours to give (flags, `PVE_*` variables or `--env-file`); none is built in. |
| `bootstrap.sh` | Run as root inside the VM, once or again for another ref (see Re-run with a new REF): docker, the data disk at `/data`, a clone of `MARINEDATA_REF`, the image, then the stages below. On Hetzner it is the user-data body; on Proxmox you start it by hand. |
| `fetch-pinned.py` | Run by the `Dockerfile`, not by you: downloads the pinned YuNet model, checks its sha256 and fails the image build on a mismatch. It replaces `ADD --checksum`, which only BuildKit has; the VM builds with the classic builder, and `tests/test_dockerfile_classic_builder.py` keeps both Dockerfiles free of BuildKit-only syntax. |

| Stage | What runs | Lands in |
|---|---|---|
| lock | host check: the lock and the split map are in the checkout; prints their sha256 | `/data/logs/lock.log` |
| build | `marinedata release build --release ID --flavour F --lock ... --split-map ... --out /data` | `/data/releases/ID/F/` (`RELEASE.json`, `tasks/`) |
| export | `python -m marinedata.hf_export` writes the shards and the summary only; `python -m marinedata.hf_card ... --citation /app/CITATION.cff` then renders the card and writes `LICENSE`, `NOTICE` and `CITATION.cff` (the repo's file, the same input as the croissant stage) | `/data/export/ID/F/` (`data/`, `README.md`, `LICENSE`, `NOTICE`, `CITATION.cff`) and `/data/export/ID/F.summary.json` |
| croissant | `marinedata release croissant DIR --release-dir ... --citation /app/CITATION.cff --repo-id R` (R is read from `registry/flavours.yaml`) | `croissant.json` in the export folder |
| gate | `marinedata release gate DIR --all --require-all [--v1-dir V]`, then `marinedata release gate DIR --v11` (G1..G9; `GATE_V11_ARGS` carries `--v1-dir V` and the manifests, and the `--v1-dir V` in it also goes to the `--all` call so its card, load and masks gates count the frozen v1.0 configs; the wrapper passes the pinned YuNet model path) | `GATES.json` in the export folder |
| plan | `marinedata release publish ...` without `--go` (`PLAN_ARGS`, e.g. `--v1-dir`, adds the v1.0 delta) | `/data/logs/plan.log` |

`STAGES` defaults to `lock build export croissant gate plan`. Without `croissant` there is no
`croissant.json`, the croissant gate skips and `--require-all` cannot pass. An **empty** `STAGES=""` is
valid and means setup only (packages, data disk, data directory, clone, image, wrapper): no stage
runs and bootstrap does not wait for secrets. `BOOTSTRAP_DRY_RUN=1` prints the resolved stage list and
the data-disk decision and exits before it touches the system.

Every stage logs to `/data/logs/<stage>.log`; `/data/logs/STATUS` has one line per stage outcome
and `/data/logs/bootstrap.log` the whole run. Extra `release build` flags (`--profile`, `--decon`,
...) go through `create.sh --build-args "..."`. Containers run through `marinedata-run` with
`--memory 14g` (an out-of-memory kill takes the container, not sshd), `TMPDIR=/data/tmp` and
`TORCH_HOME=/data/torch`, so scratch space and model weights stay off the small root disk. The image has no
`releases/` folder (`.dockerignore` excludes it), so the wrapper also mounts the checkout's `releases/` read-only at
`/app/releases`: the lock and the split map a stage reads are the ones committed at `MARINEDATA_REF`, and the
repo-relative `LOCK_FILE` and `SPLIT_MAP_FILE` become `/app/<path>`. A ref without a `releases/` folder gets no mount.

## Hetzner or Proxmox

| | Hetzner (`--target hetzner`, the default) | Proxmox (`--target pve`) |
|---|---|---|
| VM | cx32 in hel1, Ubuntu 24.04, 200 GB volume | VM `<VMID>` `marinedata-build`: Debian 12 genericcloud, 6 cores, 16 GiB RAM, 32 GB root disk, 100 GB data disk (`scsi1`) |
| `create.sh` writes | user-data with `bootstrap.sh` inlined (32768-byte limit) | `out/<name>.yaml` (a cloud-init vendor snippet: settings and the clone, no secret, no `bootstrap.sh`) and `out/vm-<VMID>.sh` (the host script) |
| Who creates it | you, with the printed `hcloud` lines, or `--run` | you, as root on the host: `bash vm-<VMID>.sh`. `--run` is refused: there is no API path like hcloud's |
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
  `local` and `local-lvm` active, a bridge with a free address on it (and its gateway), a free VM id and an existing
  VM whose `sshkeys` hold the keys that may log in. These are your values: `create.sh` has no default for them and
  refuses to render until you give them (step 2, `pve.env.example`). `vm-<VMID>.sh` checks the vmid, the address,
  the storage and the image before it creates anything; it does not check the bridge.

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
   Remove `/data/export/ID/F` first to export again. For another ref, see Re-run with a new REF below.

## Steps on Proxmox

1. **Pick the ref** (the full 40-character sha) and the release id (for example `v1.1.0`).
2. **Give your values, then render and read.** The repo holds none of your names. Copy the template, fill it in,
   and keep the copy out of git (`deploy/release-vm/pve.env` is git-ignored). Nothing is executed and no
   token or network is used:
   ```bash
   cp deploy/release-vm/pve.env.example deploy/release-vm/pve.env      # then edit it
   deploy/release-vm/create.sh --target pve --release-id v1.1.0 --flavour open --ref <40-hex sha> \
     --env-file deploy/release-vm/pve.env
   ```
   Eight values are required and have no default: `PVE_HOST` (`--host`: the Proxmox host, used only in the printed
   `ssh -J` lines), `PVE_CI_USER` (`--ci-user`: the cloud-init login), `PVE_SSHKEYS_FROM_VMID` (`--sshkeys-from-vmid`:
   the VM whose `sshkeys` are copied), `PVE_KEY_COMMENTS` (`--key-comments`: the comments of the keys you expect,
   comma-separated), `PVE_STATIC_IP` (`--ip`, with the prefix), `PVE_GATEWAY` (`--gw`), `PVE_BRIDGE` (`--bridge`) and
   `PVE_VMID` (`--vmid`). While one is unset `create.sh` names it, exits non-zero and writes no file. A value is taken
   from the flag, else the environment variable of the same name, else the `--env-file` line. The file is read as
   `KEY=VALUE` lines (`#` comments, one pair of quotes allowed) and never sourced: an unknown key, a line that is
   not `KEY=VALUE` or a value that fails its check (such as `$(...)`) is refused. Optional, with these defaults:
   `PVE_NAME=marinedata-build` (`--name`), `PVE_NAMESERVER=1.1.1.1` (`--nameserver`), `PVE_STORAGE=local-lvm`
   (`--storage`), `PVE_IMAGE=local:iso/debian-12-genericcloud-amd64.qcow2` (`--image`), `PVE_CORES=6` (`--cores`),
   `PVE_MEMORY=16384` (`--memory`), `PVE_DATA_GB=100` (`--data-gb`) and `PVE_KEYS_FILE=/root/vm<VMID>-keys.pub`
   (`--keys-file`, a path on the host). The 32G root disk, `onboot 0`, `discard=on` and `backup=0` are fixed in the
   template. `--lock-file`, `--split-map-file`, `--repo-url`, `--build-args` and `--export-args` work as on Hetzner;
   `--ssh-key`, `--location`, `--server-type`, `--volume-size`, `--os-image` and `--yes` belong to Hetzner and are
   refused here. It writes `deploy/release-vm/out/<name>.yaml` and `deploy/release-vm/out/vm-<VMID>.sh` (`out/` is
   git-ignored, `--out-dir` moves it) and prints the next commands. Read both files.
3. **Run the host script** as root, in a folder that holds both files:
   ```bash
   ssh root@<host> mkdir -p /root/vm-build
   scp deploy/release-vm/out/<name>.yaml deploy/release-vm/out/vm-<VMID>.sh root@<host>:/root/vm-build/
   ssh root@<host> 'cd /root/vm-build && bash vm-<VMID>.sh'
   ```
   It stops at the first failing command (`set -euo pipefail`): the vmid is free, the address does not answer,
   `local` and the storage are active, the image exists. It writes the keys file from the `sshkeys` of VM
   `PVE_SSHKEYS_FROM_VMID` and prints their fingerprints: read them against the count and the comments
   (`PVE_KEY_COMMENTS`) noted beside `ssh-keygen -lf` in the script. Then it installs the snippet at
   `/var/lib/vz/snippets/<name>.yaml`, creates the VM (`qm create`, `importdisk`, `scsi0` resized to 32G,
   `scsi1` as the data disk, a cloud-init drive, `cicustom vendor=local:snippets/<name>.yaml`), starts it,
   waits for the guest agent and `cloud-init status --wait`, and prints the guest's links and
   `/dev/disk/by-id/`. Read that listing: the data disk must be the `scsi-0QEMU_QEMU_HARDDISK_drive-scsi1` name.
4. **Set up the VM.** Log in through the host and run bootstrap with an empty `STAGES`:
   ```bash
   ssh -J root@<host> <ci-user>@<vm-ip>        # then: sudo -i
   set -a; . /etc/marinedata/settings.env; set +a; cd /opt/marine-data
   git rev-parse HEAD                          # must equal MARINEDATA_REF
   STAGES="" bash deploy/release-vm/bootstrap.sh
   ```
5. **Copy the secrets file, then run the stages.** From your machine; the file is written mode 600 and no copy
   stays in the home directory:
   ```bash
   ssh -J root@<host> <ci-user>@<vm-ip> "sudo sh -c 'umask 077; cat > /root/marinedata.env'" < ~/secure/marinedata.env
   ```
   In the VM, as root and in the shell of step 4, run the stages under `nohup` so a dropped ssh does not stop them,
   and watch with bounded reads:
   ```bash
   nohup env STAGES="lock build export croissant gate plan" bash deploy/release-vm/bootstrap.sh > /data/logs/run.out 2>&1 &
   tail -n 5 /data/logs/bootstrap.log; cat /data/logs/STATUS
   ```
   If a stage fails, fix the cause and re-run only what is left (`STAGES="gate plan"`). Remove `/data/export/ID/F` first to
   export again. `qm guest exec <VMID> -- <cmd>` reaches the VM if ssh does not. For another ref, see Re-run with a
   new REF below.

## Steps on both: publish, copy back, tear down

6. **Publish by hand**, after reading `/data/logs/plan.log` and `GATES.json`. The plan is a dry run; `--go` is yours to add,
   one part per call so a failure stops at a known part (`--list-parts` prints them: one `data:<config>` each, `provenance`,
   `card`, `tag`):
   ```bash
   rm -f /root/marinedata.env             # the upload needs no S3 secrets
   read -rs HF_TOKEN && export HF_TOKEN
   for part in $(marinedata-run release publish --release-id v1.1 --flavour open \
       --dir /data/export/v1.1/open --release-dir /data/releases/v1.1/open --lock /app/release.lock.yaml --list-parts); do
     marinedata-run release publish --release-id v1.1 --flavour open \
       --dir /data/export/v1.1/open --release-dir /data/releases/v1.1/open --lock /app/release.lock.yaml \
       --go --part "$part" || break
   done
   unset HF_TOKEN
   ```
   It refuses unless every required gate passed (`--all` and `--v11` both ran), refuses an existing tag, and refuses `v1.0`
   without `--allow-frozen` (v1.0 is frozen while the hackathon runs). Data parts are `upload_large_folder` (resumable, batched
   commits) over `data/<config>/*.parquet`; `provenance` and `card` are one commit each. The allow-list is `UPLOAD_PATTERNS`
   (listed in `docs/release-runbook.md`): `data/**/*.parquet`, `README.md`, `croissant.json`, `provenance/*`, `LICENSE`, `NOTICE`,
   `CITATION.cff`, `assets/*` and `docs/**`. `provenance/` is written at upload time: `RELEASE.json` as its public-safe
   projection, `GATES.json` and the lock as they are. `card` sends `README.md` alone and only after every other part; `tag`
   creates `v1.1` on the sha of the card commit (kept in `PUBLISH_STATE.json` in the release dir), never on the head of the
   branch, and is never moved. `nc` refuses unless its repo is already private; nothing here changes visibility.
   `SPLIT_MAP.json`, `DEDUP_GATE.json`, `tasks/` and `PUBLISH_STATE.json` are not uploaded. Rollback: `docs/release-runbook.md`.
7. **Copy back** before deleting anything (`scp -r root@$IP:<path> .` on Hetzner; on Proxmox
   `ssh -J root@<host> <ci-user>@<vm-ip> 'sudo tar -C / -cf - <paths without the leading />' > copy-back.tar`):
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
   qm shutdown <VMID> --timeout 120 || qm stop <VMID>
   qm destroy <VMID> --purge 1 --destroy-unreferenced-disks 1
   rm -f /var/lib/vz/snippets/<name>.yaml <keys file>      # /root/vm<VMID>-keys.pub unless you set --keys-file
   ```
   Then `qm list` no longer shows the VM and `pvesm status` shows the thin pool back at its earlier level.

## Re-run with a new REF

For a VM set up earlier that must now build from another commit (a re-frozen lock or split map, a fix to the
pipeline). The VM, `/data` and the docker cache stay. Use the full 40-hex sha.

1. In the VM, as root, load the settings and export the new ref **after** them: `settings.env` still holds the
   first-boot ref and would win otherwise. Export `RELEASE_ID`, `FLAVOUR`, `LOCK_FILE` or `SPLIT_MAP_FILE` the same
   way when the new ref changes them. (Hetzner has no `settings.env`: export `RELEASE_ID`, `FLAVOUR` and
   `MARINEDATA_REF` in the shell instead.)
   ```bash
   set -a; . /etc/marinedata/settings.env; set +a
   export MARINEDATA_REF=<new 40-hex sha>
   cd /opt/marine-data
   ```
2. **Dry run first.** It installs, mounts, formats, fetches, clones and builds nothing, and prints the stage list,
   the data-disk decision and where the checkout is and where it would go:
   ```bash
   BOOTSTRAP_DRY_RUN=1 STAGES="lock gate plan" bash deploy/release-vm/bootstrap.sh
   ```
   Read `stages=[lock gate plan]`, the data-disk line (`/data is already a mount point: the data disk is left
   alone`) and `/opt/marine-data is at <old>: it would be fetched and checked out at <new>`. It exits 1 and says
   so when the checkout has local changes to tracked files.
3. **Run it** with only the stages you want, under `nohup` as in step 5:
   ```bash
   nohup env STAGES="lock build export croissant gate plan" bash deploy/release-vm/bootstrap.sh > /data/logs/run.out 2>&1 &
   ```

What it does and what it never does:

- It fetches `origin` and moves the checkout with `git checkout --detach <commit>`, logging
  `moved /opt/marine-data from <old> to <new>`. There is no `reset --hard` and no `clean`: untracked files stay.
  Local changes to tracked files stop the move before it starts (`git -C /opt/marine-data status`; commit, stash
  or revert them, then run again). The same ref again leaves the tree as it is.
- It rebuilds the image under a tag that carries the new commit (`marinedata:<release id>-<12 hex>`) and rewrites
  `marinedata-run`, which mounts the new checkout's `releases/` at `/app/releases`. Older tags stay until you
  remove them (`docker image ls marinedata`).
- It keeps `/data`: a disk with a filesystem is mounted as it is, never formatted, and nothing under `/data`
  is removed. The `export` stage refuses a non-empty `/data/export/ID/F`; remove that folder first to export again.
- It runs only the stages in `STAGES` (`STAGES=""` stops after the set-up). A `build` or `export` stage waits for
  `/root/marinedata.env` again: copy it as in step 5 if step 6 removed it.
- The copy of `bootstrap.sh` that runs is the one you started, from the old checkout. When the new ref changes
  `deploy/release-vm/` itself, run it first with `STAGES=""` (moves the checkout, rebuilds the image), then again with
  your stages: the second run starts the new copy and the image build reuses docker's cache.

## Verification status

Nothing here has run on a VM, on Hetzner or Proxmox, against the bucket or the Hub. So far only
`tests/test_release_vm_scripts.py` (no root, no network; a recorder stands in for `hcloud` and for `docker`, and the
re-run tests use a local git repository as origin), `bash -n` and `shellcheck` have run:

- `create.sh --target pve` renders a `vm-<VMID>.sh` with the `qm`, `pct`, `ping` and `install` lines of the plan in
  order (`pve.env.example` reproduces the plan's numbers and addresses), a snippet with the 40-hex ref and without
  `HF_TOKEN=`, `SECRET` or an access key; `--run` and a short ref are refused. Each of the eight required values,
  left unset, stops the render with a message that names it and writes nothing; flag, variable and env file apply in
  that order; the env file is read and never executed; no shipped file names an operator's host, user or key.
- `bootstrap.sh` with `BOOTSTRAP_DRY_RUN=1` and a fake `lsblk`: an empty `STAGES` resolves to no stage, an unset one to
  the six stages in order, and a disk with a partition table, partitions, a mount or the root is refused. The dry run
  also says whether the checkout is absent, already at the ref or would move, and refuses local changes.
- The wrapper that `bootstrap.sh` writes mounts `releases/` read-only at `/app/releases` when the folder exists and
  otherwise adds no mount; `stage_build` passes `/app/<LOCK_FILE>` and `/app/<SPLIT_MAP_FILE>`.
- A re-run with a new ref (local git origin, fake `docker`): the checkout moves and keeps its untracked files, local
  changes to tracked files stop the move and stay as they were, and the image tag differs per commit.

What only the first run on the Proxmox VM can show (the rehearsal checks these before a real build):

- the image builds with Debian 12's `docker.io` (classic builder) and stays at or under 1.5 GB (`docker image ls marinedata`);
- the data disk appears under the by-id name above, and `tune2fs -l` reports 0 reserved blocks after `mkfs.ext4 -m 0`;
- the guest MTU is 1400 like the bridge (`ip -br link`; `ping -c1 -M do -s 1372 1.1.1.1` from the VM);
- `df -h / /data`: docker's layers fit the 32 GB root disk while `TMPDIR` and `TORCH_HOME` sit on `/data`;
- build time and peak memory (`docker stats`) against the 14 GiB container limit;
- `qm guest ping <VMID>` and `cloud-init status --wait` return on first boot;
- inside the container `/app/<LOCK_FILE>` and `/app/<SPLIT_MAP_FILE>` exist, and a re-run with a new ref on the VM
  moves the checkout and rebuilds the image (the tests cover the logic with a local origin and a fake `docker`).

Also open:

- `release gate --all --require-all` and the `configs` block in `RELEASE.json` come with the gates branch. Until it
  is merged, the gate stage runs only `--masks` and `release publish` refuses.
- Runbook coverage (`docs/release-runbook.md`): the VM runs step 3 (build), step 6 (export), the card part of
  step 8, the gates and the step 9 dry run. Steps 1-2 are replaced by the frozen split map at the
  release ref; steps 4-5 (dedup gate, decon check) are not separate stages: pass `--decon` and the other
  `release build` flags with `create.sh --build-args`, and expect the `dedup` and `splits` gates from
  `gate --all`. Step 7 is skipped. Export and card have not run on a real corpus.
- The image is built on the VM (x86_64). A local arm64 `make image` is only a smoke test.
