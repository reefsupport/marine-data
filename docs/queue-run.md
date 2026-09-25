# QUEUE-local: draining the ok ingest queue (2026-09-25)

One background runner drains the 54 remaining `ok` rows of
`registry/ingest-specs/_queue.tsv` (the 55th, `usis10k`, runs separately —
see below). It streams per item (`fetched.path.unlink()` after each item's
decode, `images/`/`labels/` files removed only after their S3 upload
verifies), so local temp never holds more than one flush window
(`temp_cap_gb/2`, default 3 GB) regardless of source size.

## Relaunch command

If the runner dies or needs restarting, it is safe to just re-run — every
source is resumable via its `CHECKSUMS.sha256` marker (skip-done) and every
partial upload resumes via its multipart checkpoint:

```sh
SP=/private/tmp/claude-501/-Users-yohanrunhaar-dev-reefsupport/0ca12ad3-aada-4ede-ab99-14fec1fa7cc2/scratchpad
nohup sh "$SP/queue/run.sh" > "$SP/queue/run.log" 2>&1 &
echo $! > "$SP/queue/pid"
```

`run.sh` runs two sequential `ingest-batch` calls (wave-w1a first, per
manager priority, then the rest), each `--jobs 3` (the Do-NOT-exceed
concurrency cap). Regenerate the id lists if the queue changes:

```sh
cd ~/dev/.wt/marine-data/ingest-int
awk -F'\t' 'NR==1{next} !/^#/ && $8=="ok" && $1!="usis10k"{print $10"\t"$1}' \
  registry/ingest-specs/_queue.tsv | sort -t$'\t' -k1,1nr | cut -f2
```

## Disk floor

Every spec this run touches got `disk_floor_gib: 36.0` added (charter D-F's
40 GiB default paused every source at the ~38 GiB free this Mac had at
launch). `DiskGuard.check()` raises `DiskFloorError` when free space drops
below the floor; `ingest-batch` catches that as a per-source `error` (not a
batch abort — the JSONL line is written and the next source still runs).
**No staged data is ever deleted by this**: `_Uploader.push` only unlinks a
local file after `upload_file`'s own verify (etag/size match), so an
interrupted source just resumes into its own checkpoint on the next
`run.sh` pass.

## Runner 2 (INT-ingest2, 2026-09-25)

A second, independent background runner drains the 17 rows the WP-6d-A/B
merge newly flipped to `ok` (`aqualoc, caddy, duo, eilat-rsmas, fathomnet,
hicrd, lsui, ntnu-arl-uw, oceaninstruct, pangaea-ccz-gsr, pangaea-ofos-msm77,
salmon-cage, seamapd21, seathru-nerf, underwater-images-2542305, usod10k,
viame-public`) plus the 4 rows that hit an HF 429 in runner 1's `run.log`
(`ruod, marineeval, uiis10k, uiis`) — 21 ids total, `--jobs 2`.

```sh
SP=/private/tmp/claude-501/-Users-yohanrunhaar-dev-reefsupport/0ca12ad3-aada-4ede-ab99-14fec1fa7cc2/scratchpad
nohup sh "$SP/queue2/run.sh" > "$SP/queue2/run.log" 2>"$SP/queue2/run.err" &
echo $! > "$SP/queue2/pid"
```

- **PID:** `55020` (recorded at `$SP/queue2/pid`)
- **Log:** `$SP/queue2/run.log` (JSONL, one line per finished source; stderr
  separately at `$SP/queue2/run.err`)
- **Ids:** `$SP/queue2/ids.txt` (21 ids, comma-separated, same file as passed
  to `--only`)
- **Disk floor:** `disk_floor_gib: 34.0` set on all 21 specs (down from the
  default 40 GiB / runner 1's 36 GiB — free disk was 33 GiB at launch, so
  this runner **starts paused**: every source raises `DiskFloorError`,
  caught as a per-source `error` line, until free space rises above 34 GiB.
  That is expected; do not lower the floor further. It shares this Mac's
  disk with runner 1 (36 GiB floor) and `usis10k`, so it will start doing
  real work once either of those frees enough space or completes.
- **D-AA (HF 429 backoff):** `ingest-batch` now cools an HF source down
  (>= 15 min) and retries it in place on a 429 rather than failing the
  batch, keeps HF concurrency at 1 regardless of `--jobs`, and gives up as
  `needs-yohan: hf-rate-limit` after 3 cool-downs — see
  `_run_one_hf_aware` in `src/marinedata/cli_ingest_batch.py`. This runner
  is the first to exercise it live (`ruod`, `marineeval`, `uiis10k`, `uiis`
  are all HF sources that 429'd on runner 1).

## Checking progress

```sh
SP=/private/tmp/claude-501/-Users-yohanrunhaar-dev-reefsupport/0ca12ad3-aada-4ede-ab99-14fec1fa7cc2/scratchpad
tail -5 "$SP/queue/run.log"                      # latest JSONL results
.venv/bin/python "$SP/queue/ledger_from_log.py" "$SP/queue/run.log" "$SP/queue/ledger.tsv" \
  && column -t -s$'\t' "$SP/queue/ledger.tsv"    # rebuild + view the ledger
ps -p "$(cat "$SP/queue/pid")"                   # confirm the runner is alive
```

## Stopping

```sh
SP=/private/tmp/claude-501/-Users-yohanrunhaar-dev-reefsupport/0ca12ad3-aada-4ede-ab99-14fec1fa7cc2/scratchpad
kill "$(cat "$SP/queue/pid")"   # kills run.sh; its current ingest-batch child exits with it
```

Killing mid-source is safe (resumable, see above) — never `kill -9` a source
mid multipart-complete, since that could leave an orphaned incomplete
upload (harmless, but `S3_ENDPOINT` storage costs accrue until Yohan runs a
multipart-abort sweep).

## `usis10k` (separate process, do not touch)

`usis10k` runs as its own `ingest-source` process (PID noted in the QUEUE-local
report), started before this batch, with `--work $SP/w1a/work`. It is
**not** part of `run.sh`. Once `ps -p <that pid>` shows it has exited, verify
its S3 prefix (size+ETag against `$SP/w1a/work/uploaded.json`) the same way
`verify_source.py` does below, then add its row to `ledger.tsv` by hand (it
never appears in `run.log`, since it never ran through `ingest-batch`).

## Verification helpers used to prove `floating-marine-debris`

`$SP/queue/list_sources.py` — flat `list_objects_v2` + `Delimiter=/` listing
of `sources/` (top-level staged ids). `$SP/queue/verify_source.py <id>
<version>` — lists one source's full prefix, confirms object count, and
confirms `CHECKSUMS.sha256` has the latest `LastModified` (uploaded last, the
version-complete marker). Anonymous GETs are
`https://rs-storage-open.hel1.your-objectstorage.com/<key>` (virtual-hosted
style — the path-style `hel1.../rs-storage-open/<key>` 403s).
