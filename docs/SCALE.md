# Scale — preparing for growth

> **Read this first.** An earlier version of this document answered the question "is a
> terabyte available today, and would SSL on it fix the production bug?" That was not the
> question. The question was **how to build so nothing breaks as data grows**, and those
> are separable. The analysis below is still useful about *today's* corpus and *today's*
> failure — but the engineering answer is in §0, and it is: the ceiling is not bytes, it
> is samples, and it has been raised.

## §0 — What was actually built (2026-08-18)

The blocker was never storage. Loaders were already generators; only the builder was
eager. `build()` holds every `Sample` in a list, and measured deep-sized with realistic
paths that is **7,772 bytes each**:

| corpus | eager | streaming |
|---|--:|--:|
| 10,000 | 17 MB | 0.98 MB |
| 100,000 | 171 MB | 1.93 MB |
| **1,000,000** | **1,693 MB** | **1.93 MB** |

So the real ceiling was about **one million samples** — and BenthicNet-1M is already in
the registry, already CC-BY, and crosses it alone. Nothing to do with terabytes.

`scan.py` fixes it. What genuinely needs whole-corpus knowledge is the label index and the
split assignment, and both need only **counters**: one int per group, one per class. Both
are O(groups + classes), never O(samples).

```python
plan = builder.build_streaming(by="site", ratios={"train": 0.7, "val": 0.15, "test": 0.15})
print(plan.summary())  # counts and splits, from the scan — no second pass
for sample in plan.split_stream("train"):
    ...  # streamed; memory flat
```

Memory is flat at 1.93 MB from 100k to 1M samples because it is bounded by group count.
Splits land exactly 70/15/15, and the eager and streaming paths call the **same**
`assign_splits`, so they cannot drift — a split that differed between them would be
nearly impossible to notice and would invalidate every comparison between runs.

`build()` stays the default below ~1M samples: simpler, random-access, and correct.

**What is deliberately still not built:** a streaming *storage* system — MosaicML
Streaming, LitData, Ray Data, a region-pinned shard volume. Those solve byte throughput,
which is not the constraint at 0.6 TB and would not have been the constraint at 1M
samples either. When a single epoch genuinely cannot read from local NVMe, revisit
§2 below, which specs it.

---

# FINAL IMPLEMENTATION BRIEF — marine-data at scale
**Date:** 2026-08-18 · **Status:** decisions, not options · Supersedes the "combine all major datasets to train on TB of data" framing.

---

## VERDICT: No. Do not do TB-scale training. The premise is wrong twice over.

**Wrong on the arithmetic:** there is no TB. Measured today, the shippable (T0+T1+T2) imagery corpus is **1.25 TB nominal, 0.59 TB reef-relevant, and 86% of that reef fraction is two datasets** (sweet-corals 339.8 GB + the unregistered Coralscapes videoframes 163.2 GB). The multi-TB mass — CoralNet ~2.4 TB, BenthicNet-11M ~1.14 TB, ReefNet-species 277 GB — is all T4/per-source-varies, and `pretrain-eu` (the only profile that admits T4) carries `retention_days: 180` under Art. 4(2). A permanent region-pinned shard volume is the permanent lake that policy forbids.

**Wrong on the diagnosis, which matters more:** the measured production failure is ~50% "unknown hard substrate" / ~32% rubble on Caribbean transects. That is a **vocabulary defect**, not a feature-quality defect. `registry/sources/reef-support-own.yaml:77` already states it: *"Fixing the ontology makes soft coral EXPRESSIBLE but supplies no substrate examples… That needs funded substrate annotation."* No quantity of unlabelled pixels creates a decoder class. Continue-pretraining on Indonesian and Red Sea reefs — three distributions whose shared defining property is the **near-absence of octocorals** — and then fine-tuning a head on 1,250 biota-only Caribbean masks yields a model with no substrate output unit. The collapse is relocated, not reduced.

**Re-scope, adopted:** the money goes to Caribbean substrate labels; the engineering goes to four correctness bugs that are live at today's 4,000-sample scale; the SSL question gets **one bounded experiment on sweet-corals alone**, and only after the eval that could falsify it exists. Everything below is scoped to that.

---

## 1. THE CORPUS

**Build set — `research` profile, ~52 GB, this is the whole training corpus:**

| source | tier | items | GB | role |
|---|---|---|---|---|
| `reef-support-benthic-own` | T0 | 1,250 | ~3.8 | Caribbean, biota-only, **the only in-region masks** |
| `coralscapes` | T2 → **T1** | 2,075 | 5.86 | dense seg, Red Sea |
| `reef-support-seaview-labels` | blocked → **open, approved 2026-10-06** (images CC-BY-3.0-AU, masks CC-BY-4.0) | 705 | ~2.7 | 46.9% Soft Coral — strongest octocoral evidence we own |
| `deolhonoscorais` | T1 | 1,411 | 12.89 | Brazil, western Atlantic |
| **NEW** Caribbean AGRRA point set | T0 | 2,000 | ~6 | §8 — the actual unlock |
| `sweet-corals` (SSL only, P4) | T1 | 90,289 | 339.8 → **~5.4 resized** | in-modality GoPro + COLMAP poses + metric scale |

**Registry corrections — apply all, they are true independently of strategy.** `coralscapes` **T2→T1**: Zenodo API returns `license.id = apache2.0`, HF tags `license:apache-2.0`, `eceo-epfl/coralscapesScripts` agrees; the registry's "dispute" cites one of those three as saying the opposite. Register the **163.21 GB `coralscapes_videoframes.7z`** on the same record as a separate `general-pretraining` source. `reefnet`/`reefnet-hf` **T1→T3** (repo tags `cc-by-nc-sa-4.0`; the paper is not the primary source). `reef-guidance-system` **T1→T3** (CC-BY-NC-SA-4.0, and it *is* labelled — the caveat is false). `reefnet-species-images` **T1→T4** (no licence tag, 277 GB of CoralNet-derived imagery). `benthicnet-1m` imagery → **PER-SOURCE-VARIES** pending `all_licenses_refs.csv`. Re-verify `seatizen-atlas` against a per-session record — the cited 11125847 returns `cc-by-4.0` and is 7.85 GB of GeoPackage, not the 1.6M images. Backfill `size_bytes` from the measured column; fix the `sweet-corals` GiB/GB error (377,957,122,048 → 339.8 GB); fix `fathomnet` (**480,531 images / 1,292,031 boxes** live, vs 100,000 / 6,600,000 registered).

**Excluded, permanently.** MarineInst20M (TX, Getty/Shutterstock). MUOT-3M (CC-BY-NC-**ND** — ND bars the derivative act, not merely the commercial one; same trap as Seatizen). NOAA Ocean Exploration (>271 TB) and Ocean Networks Canada (>20,000 h, CC-BY): free, enormous, and photometrically disjoint from sunlit reef — volume for its own sake. CoralNet and BenthicNet-11M: T4, and Art. 4(2) retention makes them unusable as a standing corpus.

**Add to registry anyway** (completeness, flagged non-reef): `planktonzilla-17M`, 17,404,047 items in **94.13 GB** — 5.4 kB/image. It is the cleanest available proof that "17M images" is not "a TB".

**One supervised exception worth a single experiment:** BenthicNet's 188,688 labelled images / 2.6M CATAMI point labels **do** carry substrate classes, and `registry/crosswalks/catami-1.4.yaml:96-100` already maps Sand→SD, Rubble→RB. That is a supervised auxiliary head on 188k labelled images, not SSL on a TB. Temperate, per-source-varies licence — caveat it.

---

## 2. STORAGE AND STREAMING

**Decision: stage everything on local NVMe. Build no streaming system.**

The rule, as a function of post-resize corpus bytes **D** and node NVMe **C**:

- `D ≤ 0.8·C` → **stage once, map-style Dataset, true `torch.randperm` per epoch.** ← we are here, by 100×.
- `D > C`, hot subset ≤ `0.8·C` → NVMe as LRU shard cache over a **same-region** bucket.
- `D > C`, every epoch touches everything → stream, and only if `B_net ≥ 1.3 × GPUs × img/s × bytes/img`.
- Cross-provider, cross-continent streaming inside the training loop → **never**.

**Resize is mandatory and is the whole trick.** SSL trains at 2×224 global + 8×96 local crops with a 512-px adaptation phase. A 4000×3000 GoPro frame at 3.9 MB carries nothing the model consumes. At 512-px short side / q90 JPEG, sweet-corals goes **339.8 GB → ~5.4 GB**; the fine-tune set is ~15 GB. One-time cost: ~90k images × 25 ms ≈ **40 CPU-minutes on 16 cores**.

**Where bytes live.** Native originals: Hetzner private cache, read rarely, durable, `~€5/mo`. Training shards: **the training pod's local NVMe**, staged at job start. Nothing else. No RunPod network volume, no R2, no MosaicML Streaming, no LitData, no Energon, no Ray Data. Format stays WebDataset tar (`mtime=0`, content-addressable) because it is the lingua franca and `shard.py` already emits it — but the *dataloader* is a plain map-style Dataset under `torchdata.StatefulDataLoader`, which is the supported survivor of the DataPipes deprecation.

**Egress number: ~€0.** Total transfer for the entire programme is ~20 GB of shards, inside Hetzner's included 1 TB. Restate `docs/ARCHITECTURE.md:95` as: *egress cost is second-order; egress bandwidth is first-order* — and at 20 GB neither binds. For the record, had we streamed 1 TB × 50 epochs it would have been ~€49 against $185–280 of GPU: the bill never decides, the 17-min-per-epoch WAN transfer would have.

---

## 3. CODE CHANGES

**Fix this week — these are defects at 4,000 samples, independent of scale.**

```python
# integrations/pandas.py:56 — O(N²), measured 0.284s@10k → 20.5s@80k
- position = list(positions)[offset] if split else offset
+ positions = list(positions)          # hoisted above the loop
+ position = positions[offset] if split else offset

# shard.py:132-134 — licence-gate fail-open
- except Exception:
-     continue
+ except Exception as exc:
+     raise ShardError(f"{source_id}: not resolvable in registry; no shard may be written") from exc

# tensorflow.py:74-75 — sorted(index.axes) called twice per sample inside the loop → hoist

# builder.py:116-129 — slice groups by cumulative SAMPLE weight, not group index
```

The split bug is the sharpest: simulated with real corpus sizes, a requested **70/15/15 returns 23.5/76.4/0.1**, BenthicNet lands entirely in val, and the whole labelled Caribbean set is the *only* thing in test. The `empty splits` guard at `builder.py:136-144` does not fire because no split is empty.

**Signature-only, lazy, zero behaviour change** (a `list` is an `Iterable`, so existing callers and tests are untouched): `LabelIndex.from_samples` and `class_weights` → `Iterable[Sample]`; transpose `class_weights`' loops so `encode` runs once per sample (O(N·A²) → O(N·A)); `Dataset.class_counts` and `supervision_coverage` → iterate `self._iter()` with a carried `self._n`; `pandas.leakage_report` → take the group→split map, not `dataset.samples` (it is a pure #groups reduction).

**Stays eager, deliberately:** `to_dataframe` (exploration tool, chunk it if it ever hurts), `to_tf_dataset` (unused on our path; note `tf.constant(paths)` hits protobuf's 2 GB ceiling around 15–20M paths — a hard wall, not a slope), the whole `DatasetBuilder.build` path.

**Deferred until a build actually exceeds 2M samples — do not build now:** the streaming `iter_samples()` generator, the Arrow manifest replacing `Dataset.samples`, per-shard sidecars, `ProcessPoolExecutor` shard writers, `_images_under` de-`sorted()`ing, the `Sample.supervised` bitmask. Every one is engineer-weeks serving a corpus we have decided not to build. `builder.py:306 list(loader)` / `:312 extend` measures **1,108 B/sample** for flat-image loaders — 4.5 GB at 2.4M samples, which survives. It is a real ceiling; it is not this quarter's ceiling.

**Tests: 281 pass today (285 selected of 295 collected).** Expected breakage from the fixes above: **the `builder.split` tests only.** Weighting by sample count changes assignments for any fixture with unequal group sizes — those assertions must be re-derived, not deleted, and a new test must assert the 70/15/15 → 23.5/76.4/0.1 case now holds proportion within tolerance. `shard.py` gains one new test (unresolvable `source_id` raises `ShardError`); nothing existing depended on the swallow. The `pandas`/`tensorflow`/`labelindex` fixes are behaviour-preserving — if any test breaks, that test encoded a bug.

---

## 4. THE LICENCE GATE UNDER STREAMING

The gate itself scales for free: `gate.evaluate`, `_permitted`, `_check_mappable`, `build_lineage`, `plan_mirror` are all **O(#sources) = 60** and run before the first `__iter__`. Three changes make it hold at the write and read boundaries.

**Write-time, streaming-native.** Replace the `{s.source_id for s in samples}` full scan with a running set checked on first sight inside the write loop, memoising `evaluate_mirror` per source id — O(1) amortised, bounded by 60, and it works on an `Iterable`. Hoist `Registry.load()` (43 ms, currently re-parsed per call) to a parameter. Combined with the `raise ShardError`, this closes the demonstrated bypass: today a single re-key (`seatizen-atlas` → `seatizen-atlas-2024`, exactly what the citation fix will cause) lets an ND source shard silently.

**Read-time, new.** Write `profile`, `allow_tiers` and a `{source_id: tier}` map to the **top level** of `SHARD_MANIFEST.json` — today profile is buried at `manifest['lineage']['profile']` and nothing reads it. Ship `load_shards(dir, *, profile) -> Iterator[dict]` that raises `LicenceViolation` when the manifest's tiers are not a subset of the profile's, enforced **per shard**. This is how a T3 sample is made unable to reach a batch: `_ADMITTED[TRAINING_SHARD]` correctly admits NC for a research cache, so the artifact must carry its own profile and the reader must refuse. The per-sample `licence_tier` at `shard.py:85` currently has **zero readers** and is `None` on every non-`generic.py` path — treat it as decorative, do not rely on it.

**Parallel writers (only if P4 ever needs them).** Each worker gets an explicit `shard_index_start` (kill `len(shards)` at `shard.py:152`, which makes every worker write `shard-000000.tar` and truncate) and writes a `shard-NNNNNN.json` sidecar with source ids, keys, skips, byte count and tar sha256. `SHARD_MANIFEST.json` and `ATTRIBUTION.md` are written **only** by a finaliser that reduces over sidecars and fails if any tar lacks one — no worker touches those paths. Switch `shard.py:230` from `lineage.attribution_text()` to `mirror.attribution_document()`, which carries licence id, tier, source URL and the non-relicensing statement; `mirror.py:227` is explicit that a drifting attribution file *"asserts compliance falsely"*, and under naive parallelism it drifts by construction. Add `marinedata attribution <dir> --verify` to CI: reconstruct the source set from the tars, diff against `ATTRIBUTION.md`.

**Also close the backbone hole.** `ls registry/models` → no such directory. `facebook/dinov3-*` is released under Meta's **bespoke DINOv3 License**, not Apache-2.0; `facebook/dinov2-*` **is** Apache-2.0. Create `registry/models/*.yaml` reusing `Tier`/`LicenceFlags`, make `gate_weights` fail closed on a missing base manifest, and **default to the DINOv2 backbone**. Without this, `LINEAGE.json` asserts a clean T1 chain over LVD-1689M — 1.689 billion unrepresented images, ~1700× the marine corpus.

---

## 5. SPLITS, SHUFFLE, RESUME

They do not conflict once identity is **content-derived rather than positional**. Everything that breaks does so because `Dataset.splits` is `dict[str, list[int]]` of stream positions and shard keys embed the enumeration index.

**Splits win; positional convenience loses.** Freeze `registry/SPLIT_MAP.json` mapping `group_key = "{source_id}/{partition}"` → split. Generate once by cumulative-sample-weight assignment; thereafter **append only** — a new group buckets by `blake2b(f"{seed}:{group_key}").int / 2**64` against cumulative ratios, O(1) and independent of the group universe. Existing groups never move. Today a single source failing to load (caught at `builder.py:307`, dropped, recorded in `skipped`) shifts slice boundaries and reassigns unrelated groups: simulated, a transient NFS error moved `src8/site7` from **train → test** — evaluation on data the model trained on, and nothing raises. Group keys come from `SourceLoader.partitions()` without reading a single sample.

**Shuffle.** Epoch order = sort by `blake2b(f"{seed}:{epoch}:{source_id}/{key}")`, not `randperm(N)`. Invariant to N, to world size, to which samples are missing. At 52 GB on NVMe this is a true global permutation, so the WebDataset shuffle-buffer problem never arises — worth naming anyway, because it is the trap for anyone who reverts to streaming: `DEFAULT_SHARD_BYTES = 512 MB` at 90 kB/sample is ~5,700 samples/shard against a conventional 1,000-sample buffer, and DINO/iBOT centering plus **KoLeo** (a nearest-neighbour spread term over the batch) collapse silently on near-duplicate transect frames while the loss curve looks fine.

**Resume.** `StatefulDataLoader.state_dict()` + model + optimizer + **EMA teacher** + per-worker RNG states (python/numpy/torch — omit these and resumed crops correlate with pre-crash ones, invisibly). ViT-L ≈ 4.8 GB, 2.5 s to local NVMe, every 15 min, SIGTERM handler. **CI test:** resume from checkpoint, assert the next 100 steps' loss matches an uninterrupted run to <1e-3. If it does not, the data layer is not resumable regardless of what the README says.

**Bind the audit to the build.** `LineageEntry.items` reads the registry YAML, and `content_hash` covers only the build *plan*: a 1-sample build of `coralvqa` emits `items: 9682` and a hash byte-identical to the full build. Rename to `items_declared`, add `items_consumed` from the actual stream, and extend the hash payload with consumed count, skip count, split seed, split-map version and sorted per-shard sha256s. And validate bytes, not existence — `shard.py:169` checks only `is_file()`, so a truncated JPEG, a zero-byte file and an HTML 403 page named `.jpg` all shard clean and the manifest declares 3 good samples. Add magic-byte + trailing-marker checks, record each rejection as `{key, reason}`, and abort above `--max-skip-fraction 0.005`.

---

## 6. TRAINING PLAN

**Decision: fine-tune, do not pretrain.** Baseline = the published EPFL Coralscapes DINOv2/DPT checkpoint (Apache-2.0 backbone), re-headed to `registry/schemas/rs-benthic-v1.yaml` (119 lines, SC branch with four octocoral children and WoRMS ids — the file exists), trained on re-headed own masks **plus the new AGRRA substrate points** under a point-supervised loss (partial cross-entropy + CoralSRT-style propagation, IGNORE where unsupervised).

**Cost:** 10–30 GPU-hours per run, **€30–90**. `docs/ARCHITECTURE.md:3h` already says this out loud, and it *"kills every argument for building infrastructure to avoid retraining."* Budget 10 runs: **~€600 all-in.**

**The one SSL experiment, gated.** Continued pretraining of DINOv2 ViT-L on **sweet-corals alone** (resized, 5.4 GB, 20–40 GPU-hours, **~€100**) — chosen not for pixel count but because it is the only pool with **COLMAP camera poses and correct metric scale**, which is exactly the defect behind `pixel_size_m` being null everywhere, and which supports multi-view geometric SSL rather than undifferentiated-pixel SSL. Not from scratch: DINOv3's ViT-7B teacher cost **61,440 GPU-hours** on 1.689B images; DINOv2 ViT-L *distillation alone* is 8,000. Three preconditions, all required: (i) E1 and E3 exist and eval-v2 is frozen; (ii) ≥8 Caribbean sites carry substrate labels so a site-level bootstrap CI is narrower than the effect sought; (iii) the fine-tuned baseline has visibly plateaued on E1.

**Eval that would tell you it worked.** E1 leave-one-site-out mIoU with site-level bootstrap CIs; **E3 class-fraction calibration** and `oov_gt_pixel_fraction` as first-class gated metrics — `ARCHITECTURE.md:3g` calls E3 *"the instrument that would have caught the collapse on day one"*. Success = substrate classes present in the output at all, and predicted cover fractions within ±5 points of AGRRA-counted truth on held-out Tayrona sites.

**Null hypothesis, stated so it can be lost.** *SSL pretraining on sweet-corals changes E1 mIoU by less than the site-level bootstrap CI, and changes E3 substrate calibration not at all.* If that holds, the TB was wasted and so was the 0.5 TB — stop, and say so in the decision record. Be honest that at 4 sites the CI is ±8–15 points, which means **today the experiment is unfalsifiable**: that, not cost, is why it comes after the labels.

---

## 7. PHASING

**P0 — this week, zero cost.** Change `registry/crosswalks/coralscapes-39.yaml:28` `"unknown hard substrate" → fidelity: unmappable` and delete `"unknown hard substrate"`/`"hard substrate"` from the RK alias list at `rs-benthic-v1.yaml:62`. Today ~50% of pixels that are an epistemic non-answer are laundered into confident Rock/pavement cover in the MRV number. Add E3 + `oov_gt_pixel_fraction` as gated metrics. Send two emails: EPFL-ECEO (Coralscapes licence, three concurring primary sources) and XL Catlin Seaview (the 705 SEAVIEW_ATL masks at 46.9% Soft Coral, currently `legal_basis: unknown` and gate-blocked).

**P1 — week 2.** The four code fixes (§3), the gate fixes (§4), `SPLIT_MAP.json` (§5), `registry/models/`, and all registry corrections from §1.

**P2 — weeks 2–10, the money.** Caribbean substrate annotation (§8). Runs in parallel with everything.

**P3 — week 10+.** Re-head, fine-tune, measure on E1–E4, freeze eval-v2.

**P4 — conditional, week 16+.** The single sweet-corals SSL run, if and only if P3 plateaued and ≥8 sites carry labels.

**Do NOT build:** the streaming/manifest refactor, the resize-and-reshard pipeline for anything beyond sweet-corals, MosaicML MDS, LitData, Energon, Ray Data, the RunPod network volume, parallel shard writers, any CoralNet/BenthicNet-11M ingestion, any T4 corpus retention. Each serves a corpus we have decided not to build.

**Record the reversal.** `~/Documents/Obsidian/Projects/ReefSupport/Decisions/2026-08-18-reject-tb-scale-ssl.md` plus a supersede note in `docs/ARCHITECTURE.md` — §3h and Open Decision 4 are the decisions this brief upholds, and the TB plan silently inverted them. Without the record, this gets re-litigated in three months.

---

## 8. THE CHEAPER ALTERNATIVE — and it is the plan

**2,000 Caribbean images × 50 AGRRA points ≈ 100,000 substrate-inclusive labels.** AGRRA because `registry/crosswalks/agrra-benthic.yaml` already maps SAND→SD, RUB→RB, CCA→CCA at `fidelity: exact` — no new schema, no new crosswalk, both committed. Points not masks because the failing metric **is a cover fraction**, which point counts estimate directly at 4–8× the throughput of dense masks. Add a ~200-image dense-mask subset, substrate-inclusive, purely as the frozen eval-v2 set.

**Cost: 65–135 annotator-hours. €1.6–6.8k at EU expert rates, ~€0.5–2k through the existing UNAL Tayrona partnership. 4–8 weeks calendar.** Plus ~€600 of GPU.

**Against:** the TB programme at €12–25k and 2–4 months, dominated by engineering, targeting a metric it structurally cannot move. The cheaper intervention is also the only one with a guaranteed, measurable effect — and it is the only line item that makes the SSL question answerable later. Fund it.