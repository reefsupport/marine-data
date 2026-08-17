# Platform architecture — training, provenance and the flywheel

> Produced 2026-08-17 by a 16-agent design workflow: both repos mapped, external
> practice researched, four independent proposals (MLOps-first, open-science-first,
> ruthless-minimalism, flywheel-first), each adversarially reviewed twice, then
> synthesised. **Verified against the code before adoption** — six of its claims
> were real defects in this repo; two are fixed in the same commit as this document.

---

# Recommended Architecture: Training, Provenance, and the Flywheel

## 0. The decision in one paragraph

Adopt the **MINIMAL spine — two repos, training inside `marine-data` behind an optional `[train]` extra** — and graft onto it three things it lacked: the **structured, recursive obligation model** from MLOPS, the **abstention-preserving fusion** design from FLYWHEEL, and the **publish-the-manifest-not-the-bytes** citation pattern from RESEARCH. Reject the third repo (`rs-train`/`rs-models`/`reef-vision`) in all three variants: it buys a clean dependency diagram and costs a permanent three-way version dance that one person will not maintain. Reject the public benchmark (ReefBench) as a v1 deliverable — the corpus cannot populate it honestly and the paper is not on the critical path to a sellable product.

**The single most important finding across all four proposals and eight critiques is not architectural.** It is this: *the ontology fix cannot fix the production failure.* The registry contains zero Caribbean substrate ground truth — own annotations are Hard Coral / Soft Coral / Milleporid / Other Sessile Invertebrates / SCALE, and only Tayrona carries all five. Adding a soft-coral class makes soft coral *expressible*; it gives the model no Caribbean sand, rubble, pavement, turf or CCA example. The ~50% unknown-hard-substrate / ~32% rubble collapse is *both* an ontology defect *and* a supervision hole, and only the first is an engineering problem. Every plan that sequences 6–10 phases behind "the ontology fix unblocks the production failure" is wrong about its own critical path.

---

## 1. Topology (answer to **a**)

**Two repos. One new module tree. No new services.**

```
marine-data (existing)                    rs-ai (existing, inference only)
├─ registry/                              ├─ ModelEndpoint + weights_sha256,
│  ├─ sources/*.yaml   ← partition-level  │    weights_lineage_uri, ontology_id
│  │                      rights (NEW)    ├─ worker checkpoint SHA verify
│  ├─ models/*.yaml    ← base ckpts (NEW) ├─ stats layer off Coralscapes strings
│  ├─ schemas/rs-benthic-v1.yaml (EXISTS) └─ upload_run_artifacts widened
│  └─ crosswalks/
├─ src/marinedata/
│  ├─ release.py       ← DatasetRelease (NEW, boundary object)
│  ├─ splitledger.py   ← append-only site→split JSONL (NEW)
│  ├─ obligations.py   ← structured Obligation (NEW)
│  ├─ weights.py       ← ObligatedArtifact + gate_weights (NEW)
│  ├─ rasters.py       ← dense-mask crosswalk applicator (NEW, largest item)
│  └─ train/           ← behind [train] extra: recipe, run, eval, card, flywheel
├─ releases/*.json     ← git-committed, append-only, CI-enforced
├─ splits/ledger.jsonl ← git-committed, append-only
└─ docker/train.Dockerfile
```

`rs-ai` imports `marinedata` for the ontology and the weights gate only — which is exactly why `marinedata`'s runtime deps must stay `pydantic + pyyaml`. Training deps live behind the extra and never enter the serving image's resolution graph.

**Training submission:** a ~50-line `scripts/train_submit.py` against the RunPod **Pods** API. Do *not* reuse `rs_ai_core.compute.RunPodBackend` — every critique that checked confirmed it is hard-bound to the Serverless `/{endpoint_id}/run` shape. The "reuse" is hollow and would couple the trainer to the customer-facing service's core package. (Taken from the MINIMAL critique; all four proposals got this wrong in different ways.)

---

## 2. How this avoids the fatal flaws

| Flaw found | Where | How this design avoids it |
|---|---|---|
| `reef-support-benthic` is T0_OWN but ~2,700 of 3,957 images are Seaview pixels; every gate passes and mints a *false* clean certificate | MLOPS crit 1, RESEARCH crit 1 | **Rights attach to `(source, partition)`.** Split into `reef-support-benthic-own` (4 Colombian sites) and `reef-support-benthic-seaview-labels` (labels ours, pixels not, tier resolved against real Seaview terms). One day of registry work; blocks Phase 1. |
| Split key too fine → same colonies both sides | FLYWHEEL crit 1 (reconstruction_id), MLOPS crit 1 (partition null) | **Split unit = persistent `site_id`**, declared on Source, carried on Sample, never derived. Reconstructions and repeat visits nest inside. Temporal holdout = last visit of each *train* site. |
| Eval-contamination guard cannot fire (keys/hashes differ across sources for identical pixels) | MLOPS crit 1, RESEARCH crit 5 | Guard is **site-set disjointness** plus a `content_sha256` collision check that raises if one hash lands in two splits. Not key intersection. |
| Flywheel generates pseudolabels *from* held-out imagery | FLYWHEEL crit 2 | Fusion job refuses any reconstruction whose `site_id` is not in the train partition. Enforced at generation, asserted again in builder. |
| Dense masks bypass harmonisation entirely; the crosswalk never touches a pixel | RESEARCH crit 2, FLYWHEEL crit 2 | `rasters.py` — `apply(mask, crosswalk, index) -> (per-axis index rasters, ignore_mask)`. This is the **largest unlisted engineering item in all four plans** and it is scoped explicitly here. |
| Eval flattered because out-of-vocab GT pixels silently become IGNORE | FLYWHEEL crit (sci) 1 | Eval uses a **frozen union vocabulary, `min_count=0`**, and reports `oov_gt_pixel_fraction` as a gated metric beside mIoU. |
| Base pretrained backbone has no licence representation; recursion bottoms out in a hole | FLYWHEEL crit 3 | `registry/models/*.yaml` reusing `Tier`/`LicenceFlags` verbatim. `gate_weights` **fails closed** on a missing base manifest. |
| Orchestrator-boot licence gate is not on the weights-load path, is bypassable, and makes a YAML edit a production outage | MINIMAL crit 3 | Enforcement moves to **CI at worker-image build/push**. Serving-side is a cheap `weights_sha256` verify after download, which is what `docs/MRV.md` already falsely claims happens. |
| Blind-annotation bias probe is unexecutable with one human | MLOPS crit 5 | Replaced by an **ordering constraint**: annotate the 2% holdout first, commit it timestamped and hashed, mint verifies the artifact predates the fusion run. Machine-checkable. |
| Phase 0 saves argmax rasters; Phase 7 needs posteriors | FLYWHEEL crit 5 | Phase 0 persists **top-5 logits + residual mass + model sha + temperature**, not argmax labels. Fixing this later is impossible; the data is gone. |
| Immutability CI gate deadlocks on the first licence correction | RESEARCH crit (sci) 7 | Gate is a **required accompaniment**, not a refusal: the same PR must add `registry/corrections/<date>-<source>.yaml` and write SUPERSEDED markers into affected releases. |

**Accepted costs (not solved):**
- **Region is confounded with corpus, annotator and vocabulary.** Unavoidable with this corpus. Mitigation is naming, not fixing (§4g).
- **n is tiny.** ~8 own sites. Every headline number gets a site-level bootstrap CI and a printed `n`. Some eval cells will read "insufficient".
- **Statutory retention vs immutable releases.** `pretrain-eu` has `retention_days: 180`. Split releases into a permanent *metadata* tier and an expiring *bytes* tier; refuse a DOI on any release containing a TDM legal basis.
- **Zenodo DOI over a manifest whose bytes are mostly private.** State it explicitly in the release rather than letting the DOI imply openness.

---

## 3. Answers (b)–(h)

**(b) Immutable, citable dataset release.** `marinedata release mint` → content hash over `{registry_digest, marinedata version, full profile snapshot, per-(source,partition) {resolved licence incl. LicenceFlags, legal_basis, items_consumed, key digest}, frozen label index + digest, split-ledger digest, per-shard sha256, per-sample content_sha256 AND label_digest}`. Three surfaces: `releases/<id>.json` in git (working), `rs-storage-private/releases/<id>/` (bytes), Zenodo version DOI on the metadata bundle at milestones only. **Label digests are mandatory** — for a segmentation project the labels are what drifts, and hashing only images means a "frozen" eval can silently improve (RESEARCH crit (sci) 6).

**(c) Weights + obligation propagation.** `rs-storage-private/weights/<model_id>/<version>/` — one directory is one auditor bundle: `model.safetensors`, `WEIGHTS.json`, `MODEL_CARD.md`, `metrics.json`, `RECIPE.yaml`, `RELEASE.json` copy, `LICENCES/`, `SHA256SUMS`. Version = content hash. `WEIGHTS.json` carries recursive `base_model.{id, sha256, lineage_uri}`. Obligations become structured objects — `{kind, sources, required_weights_licence, expires_at}` — with `non_commercial` as a kind (missing from every proposal's list, and it is the one that must hard-block). Three raising gates: run start, release mint, CI image push. `revision=` on every HF call site; `image_digest` captured. Record in `docs/LEGAL.md` that share-alike-in→share-alike-out is a **policy** choice stricter than settled law.

**(d) Flywheel.** Hard order: **ontology + raster crosswalk → eval-v2 frozen → fusion.** Non-overridable `ontology_id` check in the job. Fuse per-point **posteriors**, emit `IGNORE_INDEX` where fused mass is weak or view-disagreement is high — a pseudolabel that is 60% ignore is a good pseudolabel. Register fused output as its own source with `derived_from` and `generation: N`, inheriting the producing checkpoint's obligations (**not** T0_OWN — that would be a laundering path). Cap `generation > 2` without new human labels. `Provenance.AUTO` (which already exists with the right docstring — do not add a parallel `MODEL_DERIVED` enum) barred from eval by a raise. Develop the fusion code on wildflow/sweet-corals (CC-BY-4.0, poses, correct metric scale) so plumbing carries zero sequencing risk.

**(e) Cadence.** Three clocks. Weekly bot: `marinedata verify --sample` over declared layouts + **licence-page content digests** (requires fixing `sample_digest`, which hashes filenames + `st_size` and cannot see a content change) — opens an issue, nothing more. Release mint: event-driven, human only. Retrain: event-driven — ontology bump, ≥10 new annotated sites, or an eval regression. Licence review staggered on entry (do **not** create a single-day cliff — all 70 `verified_on` values are `2026-08-17` today).

**(f) Contribution.** One door: a GitHub PR adding a `registry/sources/*.yaml` entry, CI-gated by pydantic + `gate.evaluate` across all profiles + the bounded ~100-item live verify. Non-maintainer tier assertions default to `provenance_defective`. **Door 2 (bytes/annotations) is deferred indefinitely** — governance for zero contributors is the clearest year-one abandonment candidate. Before making `marine-data` public, scrub bucket names, the Hetzner endpoint, object prefixes and `credentials_env` keys.

**(g) Evaluation.** Four cells, no public benchmark, no leaderboard, no sealed tier.
- **E1 in-domain**: leave-one-site-out over own Colombian sites (same camera, same annotators, same schema). This is the *honest* generalisation estimate.
- **E2 cross-corpus transfer**: own ↔ Coralscapes ↔ sweet-corals, on a fixed common label subset (HC / SC / other), **labelled as cross-corpus, not cross-region**, with `n` per cell. Gated as a veto on regression beyond one measured between-site SD — not a fixed percentage.
- **E3 class-fraction calibration + coverage@precision**: the instrument that would have caught the collapse on day one. mIoU is never the headline.
- **E4 MRV metrics**: cover-% error and area-m² error; `pixel_size_m` recovery against the 654 SCALE polylines (Tayrona-only — so E4 measures within-site recovery and says so).

Keep `eval-v1 (coralscapes-39)` frozen forever for the before/after story, but score the *old* model on the *new* eval set through the published crosswalk with unmappable→abstain. eval-v1 alone rewards the defect being fixed.

**(h) Physical data.** Shards staged **once per release** onto the training pod's local NVMe (or a region-pinned network volume if GPU capacity there is confirmed first). Labelled corpus is ~10–30 GB sharded; egress is under €1/quarter. **Egress is a rounding error — do not architect around it.** The real bill is GPU: a SegFormer/DINOv3-DPT finetune on ~5k dense masks is 10–30 GPU-hours, tens of dollars per run. Say that number out loud, because it kills every argument for building infrastructure to avoid retraining. Four prefixes, four lifecycles: `releases/` (immutable), `weights/` (indefinite), `runs/` (tiered — this is the only unbounded growth), `rs-storage-open/` (public samples).

---

## 4. Phasing

**P0 — days, do this week, independent of everything.** Widen `upload_run_artifacts`' 7-name whitelist to persist rectified frames, keep-masks, camera poses and **top-5 logits + temperature + model sha** before the tempdir dies; add a tiered retention rule in the same PR. *Unblocks:* the flywheel having any usable input, and stops shipping auditors a `run_manifest.json` that references files existing nowhere. **Gated on the CTO confirming a training-rights clause exists (§5.1).**

**P1 — 1–2 weeks.** Partition-level rights split of `reef-support-benthic`. `release.py`: real `registry_digest`, inlined `LicenceFlags`, `items_consumed`, per-sample image **and label** digests, frozen label index in the release, `LabelIndex.load(release)` append-only replacing `from_samples`. Fix the `shard.py` fail-open (`except Exception: continue` around the registry lookup) and make `Dataset` carry its own `Registry`. Write `tests/test_lineage.py`, `test_shard.py`, `test_labelindex.py`. *Unblocks: everything auditable.*

**P2 — 1 week.** `splitledger.py` — append-only site→split JSONL in git (not Postgres; the durability argument that kills MLflow kills Postgres here too). Content-hash collision raise. `Provenance.AUTO` barred from eval by raise. *Unblocks: comparable metrics.*

**P3 — 2–3 weeks, the real work.** `rasters.py` dense-mask crosswalk applicator with per-axis ignore channels. Fix `"unknown hard substrate" → RK` to `fidelity: unmappable` (currently it launders an epistemic non-answer into a confident Rock claim). Author the 4 missing crosswalks. **`rs-benthic-v1.yaml` already exists** — 117 lines, SC branch, four octocoral children, WoRMS ids, SCL class. Three proposals budgeted 2–4 weeks to write a file that is written. *Unblocks: the ontology fix actually reaching pixels.*

**P4 — 2 weeks + annotation.** Freeze eval-v2 under rs-benthic-v1. **This is the schedule risk**: it needs expert dense annotation including substrate classes that do not exist yet, and AI agents cannot draw benthic masks. Ship the first checkpoint measured against eval-v1 + E3 calibration if eval-v2 slips; do not let the hardest human task block the first shippable artifact.

**P5 — 2 weeks.** `marinedata[train]`: recipe, trainer, E1–E4 harness, `WEIGHTS.json` emitter, `registry/models/*.yaml`, `gate_weights`, `train.Dockerfile`, submit script. Flat immutable run directories in Hetzner; **no MLflow, no W&B, no tracker.**

**P6 — 1 week, ships *with* P5.** rs-ai stats-layer rewrite off Coralscapes substring matching and literal class id `"12"` (all silently return 0.0 under a new head), `ontology_id` in cover JSON, `pipeline_version` bump, `weights_sha256` verify, CI image-build licence gate. Coordinate the breaking cover-JSON change with MariMap and the backend.

**P7 — later.** Flywheel v1, gated on P3 + P4. **Turning multi-view fusion on before the raster crosswalk and eval-v2 exist bakes "gorgonian = unknown hard substrate" into 30× more pixels at 30× confidence, permanently, and contaminated eval means you cannot detect that it happened.** Fusion removes variance, not bias.

**Do NOT build:** third repo; workflow orchestrator; DVC/lakeFS/Pachyderm; MLflow/W&B; ReefBench as a public benchmark; sealed eval tier with annual re-annotation; SPDX AI-BOM; EU AI Act Art. 53 template (Art. 53 binds GPAI providers — a benthic segmentation model is not in scope); Croissant until a named funder asks; contribution Door 2; annotation platform; `PipelineExecutor` DAG rewrite; `chain_hash` beyond dropping `duration_s` and binding pipeline/version/orchestrator_sha/model config.

---

## 5. Open decisions — CTO only

1. **Customer training-rights audit + GDPR retention.** The flywheel's fuel is customer footage with divers in frame. `LegalBasis` has no CONTRACT member; `Provenance` has no CUSTOMER member. Deferring blocks P0, and P0 is the only irreversible loss in progress. *Cost of deferral: every processed run destroys ~965 rasters permanently.*
2. **Coralscapes licence.** One email to EPFL-ECEO. It is also already `disputed: true`, so `gate.py` denies it on all shipping profiles *today* — meaning the legal commercial training corpus is ~1,250 own images with no substrate labels. *Cost of deferral: P5 trains a model that cannot ship.*
3. **`huggingface.co/reefsupport/CoralSCOP`.** Verify and take down or document. *Cost: a funder's first due-diligence grep.*
4. **Fund Caribbean substrate annotation.** This is the money decision the architecture cannot make. Without it, no ontology fix and no flywheel repairs the 32% rubble. *Cost of deferral: the product stays unsellable regardless of how good the provenance chain is.*
5. **Seaview rights position** — are those 4 sites' pixels usable commercially at all?
6. **Publish `marine-data` or not.** Real credibility asset; costs a scrub and a support surface.

---

## 6. Operational burden — honest

**What survives one person:** ~2,500 lines of new Python across nine modules, four flat storage prefixes, three git-tracked append-only artifacts, one Dockerfile, one submit script, one weekly bot, zero services. Everything is JSON and Markdown in object storage and git, readable with `cat` and `sha256sum`. If every tool in this design vanished you would lose convenience, not evidence.

**What I cut because it does not:** the third repo, all four standards exporters, the tracking server, the public benchmark with its annual re-annotation ritual, the three-door contribution governance, the 2% blind-annotation probe as a human process. Each was defensible; collectively they were four proposals designing for a team that does not exist.

**Build one thing that is not in any proposal:** `marinedata audit WEIGHTS.json` — re-derives the registry digest, release hash, split-ledger digest and obligation set from scratch and exits non-zero on any drift. The design creates roughly a dozen hand-maintained agreements between artifacts and no proposal asserted them together. That single command is what still works in three years when everything else is forgotten.