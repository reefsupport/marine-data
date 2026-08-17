# Should we upload everything to a data lake?

**Yes for some of it, never for other parts, and the licence tier decides which.** There
is no single answer because "upload to storage" is not one act.

The project's headline is "registry, not a data lake". That remains true for anything
*published*. But a registry that forbids all copying is useless for training: a run that
reads 500,000 individual objects from object storage spends its wall-clock on request
latency, not compute, and a GPU in one region reading a bucket in another burns egress for
no benefit. So we do need copies. The question is which copies are lawful.

---

## Two acts, not one

| | Private working cache | Public mirror |
|---|---|---|
| What it is | Our own storage, served to nobody | Published for others |
| Legally | **Internal copying** — not distribution | **Redistribution** |
| Admits | T0, T1, T2, **and T3 non-commercial** | T0, T1, T2 only |
| Never admits | ND, provenance-defective, TDM-basis, prohibited | same, plus all NC |

**This is the distinction that matters.** Non-commercial material may be cached for our
own research — no commercial exploitation, no redistribution — but publishing it would
breach the licence. Conflating the two is the single most likely way this project causes
real harm, so it is enforced in code rather than documented and hoped for:

```bash
marinedata mirror --target private-cache      # what may we cache?
marinedata mirror --target public-mirror      # what may we publish?
marinedata mirror --target public-mirror --attribution ATTRIBUTION.md
```

```python
from marinedata.mirror import MirrorTarget, plan_mirror
plan = plan_mirror(list(registry), MirrorTarget.PUBLIC_MIRROR)
print(plan.summary())      # includes, excludes with reasons, and obligations carried
```

Four absolute bars apply to **every** target, checked before tier:

- **`no_derivatives`** — a cache would be unusable (masks, crops and shards are all
  derivative acts) and a mirror unlawful. Excluded outright rather than stored and left
  as a trap.
- **`provenance_defective`** — copying propagates a grant the licensor never held, and
  makes *us* a redistributor of it.
- **`legal_basis: tdm`** — Art. 4(2) permits retention only as long as the mining
  requires. A permanent lake is the opposite of that. Mirror derived features or weights,
  never the corpus.
- **`legal_basis: unknown`** — storage makes an unresolved question permanent.

---

## Three targets

### `private-cache` — a faithful copy for throughput

Our own bucket, mirroring source structure. Purpose: stop re-fetching, and put bytes near
compute. Admits NC.

### `training-shard` — the format that actually makes training fast

Same admission as the cache, tracked separately because a shard is a **derivative**, not a
copy — which is why share-alike propagates through it to any released weights.

```python
from marinedata.shard import write_shards
result = write_shards(dataset, "s3-staging/shards", split="train")
print(result.summary())
```

[WebDataset](https://github.com/webdataset/webdataset) tar shards, ~512 MB each: one
sequential read serves thousands of samples instead of thousands of GETs. Written with
the standard library only — requiring the `webdataset` package to *produce* shards would
be a dependency for no benefit; reading works with `webdataset`, `torchdata`, or the small
`read_shard()` here.

Each shard carries:

- `<key>.jpg` / `<key>.mask.png` / `<key>.json` grouped by stem, per the convention
- keys shaped `source__partition__stem`, so a shard found detached from its manifest still
  says what is in it and where it came from
- `SHARD_MANIFEST.json` — class lists, head widths, `ignore_index`, and the full
  `LINEAGE`
- `ATTRIBUTION.md`, generated rather than hand-maintained, because an attribution file
  that drifts from the bucket asserts compliance falsely

`mtime` is zeroed, so identical content produces byte-identical shards — shards become
content-addressable and reproducible.

### `public-mirror` — ~100-item samples, permissive only

The systemic fix for verification: it makes every permissive source CI-verifiable
regardless of how awkward its origin is, and it survives upstream outages we cannot fix
(the CoralVQA `datasets-server` failure, for one). Permissive tiers only, enforced by the
gate.

Publishing this needs a decision, not just engineering — it means standing behind the
redistribution analysis per source under our own name.

---

## Where to put it

Cost and latency, not law:

- **Training shards belong near the GPUs.** `rs-ai` runs on RunPod; shards in Hetzner
  Helsinki mean cross-provider egress on every epoch. Stage shards in a bucket in the
  training region and treat them as disposable — they are rebuildable from the cache in
  one command, so they need no durability guarantee.
- **The private cache belongs where it is cheapest**, since it is read rarely once shards
  exist. Hetzner is already fine.
- **Never store the two in one bucket with one lifecycle.** The cache is the durable
  asset; shards are derived artefacts that should expire.

## Retention

- **Private cache:** indefinite for T0–T2. **For T3 (non-commercial), set an expiry** and
  re-fetch on demand — an indefinitely-held NC corpus looks less like research caching
  the longer it sits there.
- **Training shards:** expire aggressively. They are derived and reproducible.
- **TDM-basis material:** never enters storage at all; the gate refuses it. If a counsel
  opinion later permits mining, the `pretrain-eu` profile already carries
  `retention_days: 180`.

## What we deliberately do not build

A general-purpose lake that ingests everything and sorts out licences later. That is the
industry default and it is exactly how the provenance problems in MarineInst20M happened:
by the time anyone audits, the corpus is load-bearing and nobody wants the answer.

Every copy here passes a gate, carries attribution, and records why it was allowed.
