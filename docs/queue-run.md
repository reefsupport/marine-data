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
