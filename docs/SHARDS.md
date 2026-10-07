# Shards, storage, and running anywhere

How the data is packaged, where it lives, and why the same package works on RunPod, on
Snellius, and on whatever GPU we rent in two years.

---

## 1. What a shard actually is

A plain POSIX tar. Nothing more. Here is a real one, listed:

```
   403 B  reef-support-benthic-own__SEAFLOWER_BOLIVAR__f0.jpg
   208 B  reef-support-benthic-own__SEAFLOWER_BOLIVAR__f0.mask.png
   244 B  reef-support-benthic-own__SEAFLOWER_BOLIVAR__f0.json
   403 B  reef-support-benthic-own__SEAFLOWER_BOLIVAR__f1.jpg
   ...
```

**Members sharing a stem are one sample.** That is the whole
[WebDataset](https://github.com/webdataset/webdataset) convention — no index, no
manifest lookup, no database. A reader walks the tar in order and emits a sample each
time the stem changes.

The `.json` sidecar carries everything the training loop needs:

```json
{
  "source_id": "reef-support-benthic-own",
  "partition": "SEAFLOWER_BOLIVAR",
  "labels":      { "taxon": "SC" },
  "label_index": { "taxon": 0 },
  "supervised":  ["taxon"],
  "ignore_index": -100
}
```

`label_index` is the integer the model consumes; `labels` is the canonical node id a
human reads; `ignore_index` is PyTorch's `CrossEntropyLoss` default, so an axis this
source never annotated contributes zero gradient with no special-casing.

An unlabelled source's sidecar looks the same, just mostly empty — provenance still
matters even when supervision does not:

```json
{
  "source_id": "sweet-corals",
  "partition": null,
  "labels":      {},
  "label_index": {},
  "supervised":  [],
  "ignore_index": -100
}
```

### Why tar, rather than reading images directly

Latency, not size. A training run reading 500,000 individual objects from S3 pays one
round trip each — at 20–50 ms that starves a GPU no matter how many dataloader workers
you add. One sequential read of a 512 MB shard serves a couple of thousand samples.

### Three properties that were deliberate

- **Keys are self-describing.** `source__partition__stem` means a shard found detached
  from its manifest still says what is in it and where it came from.
- **`mtime` is zeroed.** Identical content produces a byte-identical tar, so shards are
  content-addressable and a rebuild is verifiable rather than merely plausible.
- **Shards are derivatives, not copies.** `write_shards()` runs the licence gate before
  writing a byte, so a no-derivatives source cannot be sharded even by accident. That
  check runs before any sample is read whether you pass it an eager `Dataset` or a
  `StreamingDataset` (`DatasetBuilder.build_streaming()`) — the streaming path knows its
  contributing sources from the scan's counters, so sharding a corpus too large to hold
  in memory does not require holding it in memory a second time.

Written with the standard library — requiring the `webdataset` package to *produce*
shards would be a dependency for nothing. Reading works with `webdataset`, `torchdata`,
or the small `read_shard()` here.

---

## 2. Where the data lives

Three tiers, three lifecycles, one bucket:

```
<private-bucket>/
  cache/<source_id>/…            durable · faithful copies · gate-checked
  releases/<release_id>.json     immutable · tiny · the audit record
  shards/<release_id>/…          derived · rebuildable · expires
```

**`cache/`** is the durable asset: what we fetched, in source shape, so nothing is
re-downloaded. Admits T0–T3 (non-commercial may be cached, never published). Set an
expiry on T3 — an indefinitely-held NC corpus looks less like research caching the
longer it sits.

**`releases/`** is a few KB of JSON per release: which sources, which licences, which
splits, content hashes. This is the thing you keep forever and the thing an auditor
reads.

**`shards/`** is disposable by design. Rebuildable from `cache/` in one command, so it
needs no durability guarantee and should expire aggressively.

The rule that matters: **never put the durable cache and the derived shards under one
lifecycle policy.** One you keep, one you delete.

---

## 3. The property that makes this portable

**Shards are files.** No service, no client library, no network protocol. That single
fact is why the same artifact works in three very different places:

| Environment | How the job gets the bytes |
|---|---|
| RunPod / Lambda / Vast | `rclone` from S3 → local NVMe, once per job |
| Snellius (HPC) | staged to `/scratch-shared` → read over InfiniBand |
| A laptop | already local |

Compare a streaming dataset system — MosaicML Streaming, Ray Data, HF streaming. They
are good tools, and they require the **compute node** to reach object storage. On an HPC
cluster that is often not available, and where it is, it is rarely fast. A tar file has
no such requirement.

That is the whole portability argument, and it is why I would not adopt a streaming
storage layer even when the corpus grows: it would trade a format that runs everywhere
for one that runs in fewer places.

---

## 4. Snellius specifically

[Snellius](https://www.surf.nl/en/services/compute/snellius-the-national-supercomputer)
is Slurm-based with `gpu_a100` and `gpu_h100` partitions, a dedicated `staging`
partition for data movement, and `/scratch-shared/$USER` at an 8 TB quota over
InfiniBand HDR100.

Three constraints shape how we use it:

**Stage, don't stream.** Data movement belongs on the `staging` partition, not inside a
GPU job. Assume compute nodes cannot reach the internet — and design so it does not
matter, because a staged tar needs no network either way. *(Confirm this for your
project's allocation; it is the first thing to check, but the design is robust to the
answer.)*

**`/scratch-shared` is purged after 14 days.** So staging is a **repeatable step, not a
one-off**. Any runbook that says "upload the data once" is wrong here. Our shards are
rebuildable and content-addressed, which is exactly what makes re-staging cheap and
verifiable.

**8 TB quota.** The entire shippable corpus post-resize is tens of GB. This is not a
constraint for us for a long time.

```bash
# ── stage (staging partition, has network) ───────────────────────────────
rclone copy hz:<private-bucket>/shards/2026-08-benthic-v1/ \
            /scratch-shared/$USER/reef/2026-08-benthic-v1/ --transfers 16

# ── train (gpu_h100, no network needed) ──────────────────────────────────
#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%j.out

module load 2023 Python/3.11.3-GCCcore-12.3.0
source $HOME/venvs/reef/bin/activate

export SHARDS=/scratch-shared/$USER/reef/2026-08-benthic-v1
srun python -m reef.train \
    --shards "$SHARDS/shard-{000000..000042}.tar" \
    --manifest "$SHARDS/SHARD_MANIFEST.json" \
    --checkpoint-dir "$HOME/runs/$SLURM_JOB_ID"
```

`SHARD_MANIFEST.json` carries the class lists, head widths and `ignore_index`, so the
job needs no registry, no network and no `marinedata` install to train — only to *build*
the shards.

**Checkpoints go to `$HOME` or project storage, never `/scratch-shared`** — a 14-day
purge would take them.

### Multi-node, when it comes to that

WebDataset's shard-per-rank split is the standard approach: each rank reads a disjoint
subset of shards. Our shards are already uniform (~512 MB) and independently readable,
so this needs no format change — just `--shards` split by `$SLURM_PROCID`.

---

## 5. Any other cloud GPU

The training code does not change. Only the staging verb does:

```bash
rclone copy hz:<private-bucket>/shards/<release>/ /workspace/shards/   # RunPod
aws s3 sync  s3://…/shards/<release>/              /data/shards/         # AWS
gsutil -m cp -r gs://…/shards/<release>/           /data/shards/         # GCP
```

Then the same `--shards` glob. That is the payoff for choosing a boring format: the
portability is free, and it does not depend on any vendor continuing to exist.

**Egress:** at tens of GB per release this is a rounding error on any provider, and
inside Hetzner's included allowance. Do not architect around it. If the corpus ever
reaches a scale where per-epoch transfer matters, the answer is to stage once to local
disk — which is what we already do — not to adopt a streaming layer.

---

## 6. What is deliberately not built

A streaming storage system, a feature store, a data lake, or a metadata service. Each
would solve a problem we do not have, and each would reduce the number of places our
data can be read. The current answer — tar files plus a JSON manifest — runs on a
laptop, a rented GPU and a national supercomputer without modification, and will still
open in ten years with `tar -xf`.
