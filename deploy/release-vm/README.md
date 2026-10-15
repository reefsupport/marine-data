# Release build VM (Hetzner, hel1)

A temporary VM builds one release flavour next to the bucket, runs the gates and prints the
`marinedata release publish` plan as a dry run. Everything here is for **you** to run. Nothing in
the repo creates a VM, reads a secret or uploads to the Hub by itself, and `bootstrap.sh` never
passes `--go`.

| File | What it does |
|---|---|
| `create.sh` | Writes the cloud-init user-data and prints the `hcloud` commands (`--print-only`, the default: no token, no network). `--run` executes the same commands after a typed confirmation. |
| `bootstrap.sh` | The user-data body, run once as root: docker, the volume at `/data`, a clone of `MARINEDATA_REF`, the image, then the stages below. |

| Stage | What runs | Lands in |
|---|---|---|
| lock | host check: the lock and `registry/SPLIT_MAP.json` are in the checkout; prints their sha256 | `/data/logs/lock.log` |
| build | `marinedata release build --release ID --flavour F --lock ... --split-map ... --out /data` | `/data/releases/ID/F/` (`RELEASE.json`, `tasks/`) |
| export | `python -m marinedata.hf_export`, then `python -m marinedata.hf_card` | `/data/export/ID/F/` (`data/`, `README.md`, `LICENSE`) |
| gate | `marinedata release gate DIR --all --require-all` (only `--masks` until the gates branch is merged) | `GATES.json` in the export folder |
| plan | `marinedata release publish ...` without `--go` | `/data/logs/plan.log` |

Every stage logs to `/data/logs/<stage>.log`; `/data/logs/STATUS` has one line per stage outcome
and `/data/logs/bootstrap.log` the whole run. Extra `release build` flags (`--profile`, `--decon`,
...) go through `create.sh --build-args "..."`.

## Prerequisites

- A release ref: the commit that holds the reviewed `release.lock.yaml` and `registry/SPLIT_MAP.json`.
  Use the full sha; a tag can move.
- For `--run` only: the `hcloud` CLI, an SSH key registered in the Hetzner project
  (`hcloud ssh-key list`) and a **fresh** project API token in `HCLOUD_TOKEN`. The earlier token is
  compromised; revoke the new one when the build is done. Check the server type is still offered
  (`hcloud server-type list`); `--server-type` picks another 4 vCPU / 8 GB type.
- The bucket credentials, operator-provided, in a local file of plain `KEY=value` lines (no quotes,
  no `export`), mode 600, kept out of the repo and out of user-data:
  `MARINEDATA_S3_ACCESS_KEY`, `MARINEDATA_S3_SECRET_KEY`, `MARINEDATA_S3_ENDPOINT`, `MARINEDATA_S3_REGION`.
- `HF_TOKEN` (fine-grained, write access to the two dataset repos only) is needed for step 6 and
  nowhere else. It is never written to a file on the VM.

## Cost

Hetzner bills by the hour: `hours x (server + IPv4 EUR/h) + volume GB x (EUR/GB-month) x hours / 730`.
Read the three prices off https://www.hetzner.com/cloud (cx32 in hel1, primary IPv4, volumes) before
running; these figures are an order of magnitude only. With 0.012 EUR/h for server plus IPv4 and
0.045 EUR/GB-month, a 10 h run with the 200 GB volume is 10 x 0.012 + 200 x 0.045 x 10 / 730 = 0.12 + 0.12,
about 0.25 EUR. The 10-07 plan budgets about 1 EUR per build. The real cost is a VM left running, so
teardown (step 8) is part of the build.

## Steps

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
6. **Publish by hand**, after reading `/data/logs/plan.log` and `GATES.json`. The plan is a dry run; `--go` is yours to add:
   ```bash
   rm -f /root/marinedata.env             # the upload needs no S3 secrets
   read -rs HF_TOKEN && export HF_TOKEN
   marinedata-run release publish --release-id v1.1 --flavour open \
     --dir /data/export/v1.1/open --release-dir /data/releases/v1.1/open --lock /app/release.lock.yaml --go
   unset HF_TOKEN
   ```
   It refuses unless every required gate passed, refuses an existing tag, and refuses `v1.0` without
   `--allow-frozen` (v1.0 is frozen while the hackathon runs). One commit is uploaded and tagged `v1.1`;
   `RELEASE.json`, `GATES.json`, the lock, `SPLIT_MAP.json`, `DEDUP_GATE.json` and `tasks/` are not.
7. **Copy back** before deleting anything (`scp -r root@$IP:<path> .`):
   `/data/releases/ID/F/RELEASE.json`, `/data/export/ID/F/GATES.json`, `/data/export/ID/F/README.md`
   (the card), the lock (`/opt/marine-data/release.lock.yaml`), `/data/logs/`. Mirroring them and the
   parquet to the bucket (plan section 4.4) is a separate step this README does not automate.
8. **Tear down.** Delete the server and the volume, and revoke the project token:
   ```bash
   hcloud server delete marinedata-v1-1-open
   hcloud volume delete marinedata-v1-1-open-data   # destroys /data
   ```

## Not verified yet

- Nothing here has been run on a VM or against Hetzner, the bucket or the Hub. `create.sh --print-only`,
  `bash -n` and the unit tests in `tests/test_release_vm_scripts.py` are the checks so far.
- `release gate --all --require-all` and the `configs` block in `RELEASE.json` come with the gates
  branch. Until it is merged, the gate stage runs only `--masks` and `release publish` refuses.
- Runbook coverage (`docs/release-runbook.md`): the VM runs step 3 (build), step 6 (export), the card part of
  step 8, the gates and the step 9 dry run. Steps 1-2 are replaced by the frozen `registry/SPLIT_MAP.json` at the
  release ref; steps 4-5 (dedup gate, decon check) are not separate stages: pass `--decon` and the other
  `release build` flags with `create.sh --build-args`, and expect the `dedup` and `splits` gates from
  `gate --all`. Step 7 is skipped. Export and card have not run on a real corpus.
- The image is built on the VM (x86_64). A local arm64 `make image` is only a smoke test.
