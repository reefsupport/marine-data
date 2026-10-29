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
print(plan.summary())  # includes, excludes with reasons, and obligations carried
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

## `metadata.parquet` in a staged tree: the optional `partition` column

A staged `sources/<id>/<version>/metadata.parquet` is the wide `sample_schema` table (37 columns,
schema v2; the older 34-column v1 shape is also accepted). Trees restaged from the legacy layout
(bucket reorganisation, 2026-10-06) carry **one extra trailing `partition` column** (38
columns): the `images/<partition>/` segment (`SEAVIEW_ATL`, a site, a legacy sub-directory) that
`StagedTreeLoader` pairs images, masks and labels on. `sample_schema.validate_table` accepts the
table with or without that trailing column (`OPTIONAL_COLUMNS`); it is string-typed, must come last,
and is not a `SampleRow` field. A flat table (no `partition`, `image_path` set) stays valid and
resolves to `images/<stem>.<ext>`; a row with neither raises (`staged_partition`).

Sample ids are `<source_id>/<stem>`. Where a stem repeats across partitions (for example the 483
`labelled_data/segments` images of `seaview-survey-imagery`) the id is
`<source_id>/<partition>/<stem>`, because `validate_rows` rejects duplicate ids. The licence string of
an unlicensed source is written verbatim as `NO-LICENCE-STATED`; the schema's own null form is
`NOASSERTION`.

## Coralscapes V2 tree (`sources/coralscapes-v2/2.0/`)

`scripts/stage_coralscapes_v2.py` (`src/marinedata/ingest_coralscapes_v2.py`) writes this tree on VM 900,
only under this prefix. `<stem>` is the HF frame stem `site<N>_<frame>_<ts>`; the files keep the V1 naming
(`_leftImg8bit`), so a V1 twin has the same file stem:

```text
sources/coralscapes-v2/2.0/
  images/default/<stem>_leftImg8bit.png              image, bytes verbatim from the HF parquet shard
  labels/masks/default/<stem>_leftImg8bit.png        semantic ids 1-95, 0 = unlabelled; uint8 (mode L)
  labels/instances/default/<stem>_instanceIds.png    cityscapes instance ids; uint16 (mode I;16); fish class 25, divisor 2000
  metadata.parquet                                   wide sample_schema v2 table: 37 columns + trailing `partition`
  ANNOTATIONS.json  SOURCE.json  LICENSE
  CHECKSUMS.sha256                                   written last; a tree without it is an interrupted run
```

- `partition` is `default` in every row, so `StagedTreeLoader` pairs `images/default/` with
  `labels/masks/default/`. The `stem` column holds `<stem>_leftImg8bit`, the V1 file stem: it is what links
  a V2 frame to its V1 twin (sha256 cannot, the bytes and pixels differ).
- Instance PNGs sit under `labels/instances/`, not `labels/instance_masks/` as in older trees; the generated
  map below lists the older names.
- The tree holds 2,433 frames: the HF split (`train` 1,790, `validation` 170, `test` 473) goes to
  `upstream_split`. Staging behaviour, flags and the resume rules: `docs/ingest-howto.md`, section 5.

<!-- BEGIN GENERATED: storage-map -->
### Canonical storage map (generated from `registry/sources/*.yaml`)

One location per id: `sources/<id>/<version>/` (images under `images/<partition>/`, labels under
`labels/{masks,instance_masks,exports,points}/`, then `metadata.parquet`, `CHECKSUMS.sha256`,
`INGEST.json`). Non-open sources live in `<private-bucket>`. `—` means unknown, never zero.
Regenerate with `python scripts/gen_storage_docs.py`; check offline with
`marinedata registry verify --listing <bucket listing>`.

| id | canonical location | version | access class | n_images | n_annotations | n_files | checksums |
|---|---|---|---|---:|---:|---:|---|
| `aqqua-baltic-holo` | `s3://<open-bucket>/sources/aqqua-baltic-holo/record-18405776/` | record-18405776 | open | 88,068 | — | — | pinned |
| `aqua20` | `s3://<open-bucket>/sources/aqua20/rev-5ccc9d6b8d87/` | rev-5ccc9d6b8d87 | internal-only | 8,171 | — | — | pinned |
| `aquascan-1k-sss` | `s3://<open-bucket>/sources/aquascan-1k-sss/record-18771165/` | record-18771165 | open | 1,033 | — | — | pinned |
| `aris-didson-fish-td` | `s3://<open-bucket>/sources/aris-didson-fish-td/record-5046708/` | record-5046708 | open | 19,226 | — | — | pinned |
| `auv-flc-fls-sunboat` | `s3://<open-bucket>/sources/auv-flc-fls-sunboat/record-10544811/` | record-10544811 | open | 17,676 | — | — | pinned |
| `benthoz15` | `s3://imos-data/IMOS/AUV/` | unversioned | open | 9,874 | — | — | unpinned |
| `brackishmot` | `s3://<open-bucket>/sources/brackishmot/zip-e717dc1438aa/` | zip-e717dc1438aa | unknown | 76,618 | — | — | pinned |
| `coral-bleaching-detection-v2i-multiclass` | `s3://<private-bucket>/coral_bleaching/others.tar.gz` | unversioned | unknown | 736 | — | — | unpinned |
| `coral-health-classification` | `s3://<private-bucket>/coral_bleaching/others.tar.gz` | unversioned | unknown | 1,637 | — | — | unpinned |
| `coralscapes` | `s3://<open-bucket>/sources/coralscapes/1.0/` | 1.0 | open | 2,075 | — | — | pinned |
| `coralscop-masks-rs` | `s3://<open-bucket>/sources/coralscop-masks-rs/2026-09-23-3e8612678469/` | 2026-09-23-3e8612678469 | internal-only | 38,928 | — | — | pinned |
| `coralseg-ucsd-mosaics` | `s3://<open-bucket>/sources/coralseg-ucsd-mosaics/unversioned/` | unversioned | internal-only | 4,922 | — | 9,845 | pinned |
| `coralvqa` | `s3://<open-bucket>/sources/coralvqa/rev-3da50a4429e4/` | rev-3da50a4429e4 | restricted-nc | 11,804 | — | 11,809 | pinned |
| `csiro-cots` | `s3://<open-bucket>/sources/csiro-cots/zip-eb9e280ea1ca/` | zip-eb9e280ea1ca | internal-only | 23,501 | — | — | pinned |
| `deepseagrass` | `s3://<open-bucket>/sources/deepseagrass/dap-47653-72004/` | dap-47653-72004 | restricted-nd | 72,003 | — | — | pinned |
| `elliott-bay-benthic` | `s3://<open-bucket>/sources/elliott-bay-benthic/rev-cd20e627ba77/` | rev-cd20e627ba77 | restricted-nc | 1,000 | — | — | pinned |
| `estonefish-scenes-real` | `s3://<open-bucket>/sources/estonefish-scenes-real/1.0/` | 1.0 | open | — | — | — | pinned |
| `fathomnet-fgvc23` | `s3://<open-bucket>/sources/fathomnet-fgvc23/rev-4636bed20b3b/` | rev-4636bed20b3b | restricted-nd | 5,950 | — | — | pinned |
| `fathomnet-fgvc25` | `s3://<open-bucket>/sources/fathomnet-fgvc25/2025/` | 2025 | restricted-nd | 9,306 | — | — | pinned |
| `floating-marine-debris` | `s3://<open-bucket>/sources/floating-marine-debris/git-427059e47f99/` | git-427059e47f99 | open | 11 | — | — | pinned |
| `galapagos-insas` | `s3://<open-bucket>/sources/galapagos-insas/record-17393662/` | record-17393662 | open | 21 | — | — | pinned |
| `ibf` | `s3://<private-bucket>/sources/ibf/2026-10-06/` | 2026-10-06 | internal-only | 496 | — | 579 | pinned |
| `kaggle-healthy-bleached-corals` | `s3://<private-bucket>/coral_bleaching/others.tar.gz` | zip-5e15452254a9 | internal-only | 923 | — | — | pinned |
| `koi-rgb-sonar` | `s3://<open-bucket>/sources/koi-rgb-sonar/rev-4120e0f1b59d/` | rev-4120e0f1b59d | open | 326,076 | — | — | pinned |
| `marineeval` | `s3://<open-bucket>/sources/marineeval/rev-cb94f5a61594/` | rev-cb94f5a61594 | open | 2,672 | — | — | pinned |
| `marineevt` | `s3://<open-bucket>/sources/marineevt/rev-37488d3c7690/` | rev-37488d3c7690 | open | 169,409 | — | — | pinned |
| `marineinst20m-web` | `s3://<open-bucket>/sources/marineinst20m-web/manifest-c596e4376e6f/` | manifest-c596e4376e6f | internal-only | 8,217 | — | — | unpinned |
| `mas3k-hf` | `s3://<open-bucket>/sources/mas3k-hf/rev-1dff3d4b1fb8/` | rev-1dff3d4b1fb8 | internal-only | 8,730 | — | — | pinned |
| `mermaid-aws` | `s3://<open-bucket>/sources/mermaid-aws/2026-09-19-bc53d5a2c0b6/` | 2026-09-19-bc53d5a2c0b6 | restricted-nc | 18,561 | — | — | pinned |
| `meuvs-video-summ` | `s3://<open-bucket>/sources/meuvs-video-summ/record-20746075/` | record-20746075 | open | 40,587 | — | — | pinned |
| `muot3m` | `s3://<open-bucket>/sources/muot3m/rev-4294cb6a2537/` | rev-4294cb6a2537 | restricted-nd | 3,074,016 | — | — | unpinned |
| `nes-plankton-2022` | `s3://<open-bucket>/sources/nes-plankton-2022/rev-18fb2e57fd77/` | rev-18fb2e57fd77 | open | 97,026 | — | — | pinned |
| `noaa-dsc-csv` | `s3://<open-bucket>/sources/noaa-dsc-csv/zip-5257227aef7d/` | zip-5257227aef7d | unknown | — | — | — | pinned |
| `noaa-esd-coral-bleaching` | `s3://<open-bucket>/sources/noaa-pifsc-bleaching/1-image-labels/` | v1 | open | 1,568 | — | — | unpinned |
| `noaa-oceaneyes` | `s3://<open-bucket>/sources/noaa-oceaneyes/list-a11ed85b698b/` | list-a11ed85b698b | unknown | 20,065 | — | — | pinned |
| `noaa-pifsc-bleaching` | `s3://<open-bucket>/sources/noaa-pifsc-bleaching/1-image-labels/` | 1-image-labels | open | 10,419 | — | — | pinned |
| `noaa-pifsc-esa-coral-icra` | `s3://<open-bucket>/sources/noaa-pifsc-esa-coral-icra/rev-b5e722f7c11e/` | rev-b5e722f7c11e | open | 470 | — | — | pinned |
| `oceangym` | `s3://<open-bucket>/sources/oceangym/rev-ca7a8b677cdb/` | rev-ca7a8b677cdb | open | 1,220 | — | — | pinned |
| `pangaea-cvap-lander-bbox` | `s3://<open-bucket>/sources/pangaea-cvap-lander-bbox/PANGAEA.961314/` | PANGAEA.961314 | open | 2,509 | — | — | unpinned |
| `pingmapper-sss-seg` | `s3://<open-bucket>/sources/pingmapper-sss-seg/v1.0.0/` | v1.0.0 | open | 31,014 | — | — | pinned |
| `plankton-interaction-videos` | `s3://<open-bucket>/sources/plankton-interaction-videos/rev-f7d40238215e/` | rev-f7d40238215e | open | 21,221 | — | — | pinned |
| `qut-fish` | `s3://<open-bucket>/sources/qut-fish/zip-ce6985436f9b/` | zip-ce6985436f9b | restricted-nc | 13,231 | — | — | pinned |
| `reef-support-benthic-own` | `s3://<open-bucket>/sources/reef-support-benthic-own/2026-10-06/` | 2026-10-06 | open | 1,250 | 14,517 | 18,276 | pinned |
| `reef-support-bleaching` | `s3://<open-bucket>/sources/reef-support-bleaching/2026-09-24/` | 2026-09-24 | open | 658 | — | — | pinned |
| `reef-support-seaview-labels` | `s3://<open-bucket>/sources/reef-support-seaview-labels/2026-10-06/` | 2026-10-06 | open | 2,707 | 50,603 | 56,025 | pinned |
| `reefolution` | `s3://<private-bucket>/sources/reefolution/2026-09-23-2c84cb0c9cda/` | 2026-09-23-2c84cb0c9cda | internal-only | 870 | — | 872 | pinned |
| `rmas-hf` | `s3://<open-bucket>/sources/rmas-hf/rev-7eed10d36cd0/` | rev-7eed10d36cd0 | internal-only | 3,014 | — | — | pinned |
| `roboflow-aquarium` | `s3://<open-bucket>/sources/roboflow-aquarium/zip-5d30fed3dd5a/` | zip-5d30fed3dd5a | open | 638 | — | — | pinned |
| `roboflow-coral-bleaching-final-v6i` | `s3://<open-bucket>/sources/roboflow-coral-bleaching-final-v6i/v6i-image-labels-r2/` | v6i-image-labels-r2 | open | 2,551 | — | — | pinned |
| `roboflow-coral-bleaching-general-v1-yolov8s` | `s3://<open-bucket>/sources/roboflow-coral-bleaching-general-v1-yolov8s/v1-yolov8s-image-labels-r2/` | v1-yolov8s-image-labels-r2 | open | 2,543 | — | — | pinned |
| `roboflow-coral-classification-copy-changed-v13i` | `s3://<open-bucket>/sources/roboflow-coral-classification-copy-changed-v13i/v13i-image-labels/` | v13i-image-labels | open | 2,789 | — | — | pinned |
| `roboflow-coral-reef-bleach-detection-v2i` | `s3://<open-bucket>/sources/roboflow-coral-reef-bleach-detection-v2i/v2i-image-labels/` | v2i-image-labels | open | 10,781 | — | — | pinned |
| `roboflow-coral-reef-classification-v3i` | `s3://<open-bucket>/sources/roboflow-coral-reef-classification-v3i/v3i-image-labels/` | v3i-image-labels | open | 4,556 | — | — | pinned |
| `rs-labelled-masks` | retired (no read path) | unversioned | internal-only | 63,167 | — | — | unpinned |
| `ruod` | `s3://<open-bucket>/sources/ruod/rev-c22094e45b7f/` | rev-c22094e45b7f | internal-only | 14,000 | — | — | pinned |
| `seattle-aquarium` | `s3://<open-bucket>/sources/seattle-aquarium/rev-78701c4efe73/` | rev-78701c4efe73 | restricted-nc | 6,522 | — | — | pinned |
| `seaturtleid2022` | `s3://<open-bucket>/sources/seaturtleid2022/rev-91cd6c1126c7/` | rev-91cd6c1126c7 | unknown | 8,729 | — | — | pinned |
| `seaview-survey-imagery` | `s3://<open-bucket>/sources/seaview-survey-imagery/2026-10-06/` | 2026-10-06 | open | 11,870 | — | 12,376 | pinned |
| `sonarsweep` | `s3://<open-bucket>/sources/sonarsweep/rev-350b20a9acaf/` | rev-350b20a9acaf | internal-only | 127,396 | — | — | pinned |
| `sss-mine-detection` | `s3://<open-bucket>/sources/sss-mine-detection/2/` | 2 | open | 1,170 | — | — | pinned |
| `suim` | `s3://<open-bucket>/sources/suim/2020/` | 2020 | open | 1,598 | — | — | pinned |
| `swdd-sss-wall` | `s3://<open-bucket>/sources/swdd-sss-wall/record-13692547/` | record-13692547 | open | 10,771 | — | — | pinned |
| `synthetic-seabed-debris` | `s3://<open-bucket>/sources/synthetic-seabed-debris/1.0.0/` | 1.0.0 | restricted-nc | 4,830 | — | — | pinned |
| `uatd` | `s3://<open-bucket>/sources/uatd/3/` | 3 | open | 9,212 | — | 9,205 | pinned |
| `uiis` | `s3://<open-bucket>/sources/uiis/rev-44ca5db46599/` | rev-44ca5db46599 | open | 4,628 | — | — | pinned |
| `uiis10k` | `s3://<open-bucket>/sources/uiis10k/rev-5c4316368a8e/` | rev-5c4316368a8e | open | 10,048 | — | — | pinned |
| `usis10k` | `s3://<open-bucket>/sources/usis10k/rev-b10f41ab819b/` | rev-b10f41ab819b | open | 10,632 | — | — | pinned |
| `usod` | `s3://<open-bucket>/sources/usod/zip-fb2a1c738fd7/` | zip-fb2a1c738fd7 | unknown | 14,600 | — | — | pinned |
| `uw-stereodepth-40k` | `s3://<open-bucket>/sources/uw-stereodepth-40k/rev-6aa807c87f85/` | rev-6aa807c87f85 | internal-only | 40,000 | — | — | unpinned |
| `uwbench` | `s3://<open-bucket>/sources/uwbench/rev-84af94500308/` | rev-84af94500308 | internal-only | 3,819 | — | — | unpinned |
| `uwstereo-syn` | `s3://<open-bucket>/sources/uwstereo-syn/rev-62dd4c52418c/` | rev-62dd4c52418c | open | — | — | — | pinned |
| `uwturbidity-fnu` | `s3://<open-bucket>/sources/uwturbidity-fnu/1.0.0/` | 1.0.0 | unknown | 675 | — | — | pinned |
| `uxo-acoustic-optical` | `s3://<open-bucket>/sources/uxo-acoustic-optical/1.1/` | 1.1 | unknown | 3 | — | — | pinned |
<!-- END GENERATED: storage-map -->
