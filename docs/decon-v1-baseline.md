# Decon v1 baseline (P2, 2026-09-25)

`marinedata decon check` (WP-12/P2) run for real against the two staged benchmark
manifests — `coralscapes` (557 records) and `suim` (110 records) — versus v1.
Design: `docs/design/eval-decontamination-split-v2.md` §2/§5.

## Data source: local export, not the published HF repo

The intended source (per the brief) was anonymous HTTPS against v1's published HF
release, `reefsupport/open-marine-imagery@v1` (D-A). Both
`https://huggingface.co/api/datasets/reefsupport/open-marine-imagery` and the
`/tree/v1` variant returned `{"error":"Invalid username or password."}` with no
credentials sent. A control request against `https://huggingface.co/api/datasets/mnist`
succeeded, and `~/.netrc`/`HF_TOKEN`/`.curlrc` were all absent, isolating this to the
repo itself: **it is not publicly readable yet** (private or unpublished), not a
network or environment problem. This is a real finding, not a package failure — no
attempt was made to authenticate around it, per the brief's "anonymous HTTPS"
instruction and the charter's "no HF create/upload/push" constraint.

**Fallback used:** the real local v1 export at `~/dev/reefsupport/data/_hf/v1/`
(13GB, referenced by `docs/dedup-report.md` as the same corpus WP-10's dedup run
used) — `data/metadata/{train,validation,test}-*.parquet` (69,600 rows total,
`image_sha256` + `upstream_id`, no image bytes) and `data/images/{train,validation,test}-*.parquet`
(same 69,600 rows, with decodable image bytes). This is real v1 production data, not
a synthetic fixture and not a proxy corpus.

## Scope of this run

- **S0 (upstream_id) / S1 (sha256):** full coverage — all 69,600 v1 rows, all three
  splits, `metadata` config only (no image bytes decoded; a plain dict join).
- **S2 (pixel sha256) / S3 (dHash/pHash + entropy guard):** bounded to the
  `validation` + `test` shards of the `images` config — 9,868 of 69,600 rows
  (14%). The 24 `train` shards (~59,732 rows) were **not** decoded in this pass;
  this is a disclosed wall-clock scope limit, not a silent gap. Re-running with the
  train shards included is a follow-up, not part of this gate.
- **S4 (SSCD embedding cosine):** skipped — no SSCD weights were fetched for this
  run (`--embed-weights` unset). `thresholds.s4_embedding_model` is fixed to
  `sscd_disc_mixup` (matching D-V; see `registry/benchmarks.yaml`) so a future run
  with weights present is a drop-in.
- **S5 (patch/crop, D-T):** off — this run used `dedup_crop=False`, the default.
  WP-10c's `--dedup-crop` flag (merged from `feat/wp10-dedup` @ `8c26170`) now
  gates decon's own S5 stage the same way; flipping it on for the real v2 build is
  D-T2's call for the integrator, not this gate.

## Results

| benchmark | manifest_n (hashed_n) | eval_n | S0 | S1 | S2 | S3 | S4 | S5 | status |
|---|---|---|---|---|---|---|---|---|---|
| coralscapes | 557 | 392 | 0 | 0 | 0 | 0 | skipped | off | clean |
| suim | 110 | 110 | 0 | 0 | 0 | 0 | skipped | off | clean |

`eval_n` is the registry's declared upstream eval-split count
(`upstream_split.counts`); `manifest_n`/`hashed_n` is the size of the staged
manifest actually checked (all records, not just the eval split) — the two differ
for coralscapes because its manifest carries more than the declared `test` rows.

Zero overlaps were found on every stage run (S0-S3) for both benchmarks against
the validation+test deep-pass sample and the full-corpus S0/S1 metadata scan. This
is a real, disclosed result for the scope above — it does **not** cover S4, S5, or
the v1 `train` split's image bytes, so it is a partial clearance, not a full
zero-contamination proof for v1 end-to-end.

Run script (scratch, not part of this repo):
`/private/tmp/claude-501/.../scratchpad/p2/real_run2.py`. Wall time: 11s.
