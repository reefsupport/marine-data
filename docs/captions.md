# Captions layer

Two tiers, one schema (`registry/captions/schema.yaml`, per-image parquet keyed by
`image_sha256`): Tier A (`caption_template`, `label_origin=derived`) is a deterministic
sentence built only from released metadata and labels; Tier B (`caption_vlm`,
`label_origin=model`) is a pilot of open-weights VLM dense captions grounded in Tier A's
facts. Code: `src/marinedata/captions/{facts,template,consistency,vlm,schema,cli}.py`.

## Tier A — `caption_template`

`facts.py` extracts a `CaptionFacts` dataclass per row from three sources already on
disk: the v1 `metadata` config (`source_id`, `habitat`, `depth_m`, `meow_ecoregion`,
`platform`), the `bleaching-condition` task config's resolved `label` column, and the
`benthic-coarse`/`benthic-l2` task configs' resolved `label` column. A field that is
null on the source row is omitted — never guessed — and every clause in the caption
traces back to one JSON key in `caption_facts`. `template.py` renders the populated
facts into one sentence in a fixed clause order (source → habitat → depth → ecoregion →
benthic → bleaching → taxa); a fact-less row gets an empty string, never a placeholder.

**Two known-empty fields on v1, by design, not a bug:**
- `benthic_dominant`/`benthic_present` read the `benthic-coarse`/`benthic-l2` `label`
  column. In the current tree that column is **100% null for
  all 63,330 rows** — the points/mask rollup is not wired yet (see
  `docs/task-layers.md`). This module deliberately does not recompute that rollup
  itself (the config wiring owns it); the fact appears automatically once the config wiring
  lands, with no code change here.
- `top_taxa` has no populated per-image field anywhere in v1; always empty.

**`label_status` proxy.** The project decision gates `bleaching_status` on
`label_status=ok`. No `label_status` column exists yet anywhere in the codebase (grep
confirms — it is a planned name, not a landed one). This module uses the
`bleaching-condition` task's own resolved-vs-null `label` column as the proxy: a null
`label` (an ambiguous multi-class box/mask, `label_reason` null) contributes no fact; a
resolved `label` (`label_reason="exact"`) does. Switch to a real `label_status` column
once the confident-learning gate lands — the call site is `facts.build_facts_table`.

### Measured counts (real run, this branch, 2026-09-25)

| Corpus | rows | ≥3 facts | coverage |
|---|---|---|---|
| v1 release (train+validation+test, all 8 sources) | 69,600 | 25,467 | **36.6%** |
| staged `mermaid-aws` (see below) | 18,561 | 18,561 | **100.0%** |
| **combined** | **88,161** | **44,028** | **49.9%** |

v1's low coverage is arithmetic, not a bug: `source_id` + `habitat` are 100%-covered
(2 facts on every row) but `meow_ecoregion` (16.4%) and `bleaching_status` (~28.4%,
17,665/62,244 `bleaching-condition` rows resolved) are the only other populated
fields, and `benthic_dominant`/`top_taxa` are always empty (above). A row needs one of
those two thin fields to cross the 3-fact line; 36.6% of v1 rows have at least one.

Facts extraction is pandas table joins over parquet already on disk — no image I/O —
measured at **41,292 rows/s** on this machine; the full 69,600-row v1 run and the
staged corpora both complete in well under a second. This throughput is not the
bottleneck for "the full run" (see below); it is reported for completeness only.

### `mermaid-aws` and `coralscapes` (brief item 2, staged sources)

- **mermaid-aws**: `~/.cache/marinedata/mermaid-aws/mermaid_confirmed_annotations.parquet`
  is 464,025 point-level rows (`image_id`, `benthic_attribute_name`, `growth_form_name`,
  ...) over 18,561 distinct images — real point data, not yet ingested
  (no `image_sha256` exists for this source). `facts.rollup_points_to_facts` implements
  the points→image rollup directly (dominant ≥50% of points, present ≥10%, unknown
  points excluded, image dropped if >50% unknown) and keys the output
  `native:mermaid-aws:<image_id>` rather than a sha256, so it can never be silently
  merged into the sha256-keyed schema before ingest assigns one. Output:
  `<out>/captions_template_mermaid_aws.parquet` (18,561 rows, kept separate from
  the v1 output for the same reason).
- **coralscapes**: `~/.cache/marinedata/coralscapes/_shards/*.parquet` has only `image`
  (pixels) and `label` (a segmentation mask) — no per-image id, source metadata, or
  geography beyond pixel content, and the metadata note
  already closes out coralscapes geography as "none". There is no field here to build a
  fact from without guessing from pixels, which Tier A does not do. Skipped; not a gap
  in this module.

## Tier B — `caption_vlm` pilot (harness built, **not executed this pass**)

`vlm.py` implements the full pipeline: `build_prompt` grounds the model in a row's
Tier-A facts and instructs it never to contradict them; `stratified_sample` draws a
deterministic proportional sample by `(source_id, habitat)`; `load_backend` prefers the
MLX 4-bit build of `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` on MPS and falls back to
`transformers`'s `Qwen/Qwen2.5-VL-3B-Instruct` on MPS; `run_pilot` records
`vlm_model`/`vlm_revision`/`prompt_sha256`/`seed`/`label_origin=model` per row;
`cooldown_then_requeue` implements D-AA's backoff (≥15 min cool-down, requeue at the
tail, ≤3 attempts before it is listed as needing a decision). All of it is unit-tested against a fake
`VLMBackend` (`tests/test_captions_vlm.py`) with no network or model download.

**The 500-image pilot itself was not run this pass.** Neither `mlx-vlm` nor
`transformers` is installed (neither is a `pyproject.toml` dependency, and this layer does
not edit `pyproject.toml`); downloading and running a 3B-class VLM is out of this
worker's tool-call budget alongside everything above. To run it:

```
uv pip install mlx-vlm   # or: uv pip install transformers accelerate
uv run python -m marinedata.captions.cli pilot \
  --metadata <metadata>.parquet --images-root <staged-images-dir> \
  --n 500 --seed 0 --out <out>/captions_vlm_pilot.parquet
```

Once real `caption_vlm` rows exist, `marinedata captions check` runs the automated
consistency checker (`consistency.py`) over them and reports the flagged rate.

## Hallucination rate

- **Automated consistency check (all rows, real run).** `consistency.check_caption`
  flags a caption that *names* a bleaching state or coarse benthic class contradicting
  the row's own facts. Run against the 69,600-row v1 `caption_template` output:
  **0/69,600 flagged (0.0%)** — expected, since a template caption is built from the
  same facts it is checked against; this is a regression guard, not a measurement of
  real hallucination (that requires `caption_vlm` rows, not run this pass).
- **Blind audit (150-caption contact-sheet review).** Not run this
  pass — it needs real `caption_vlm` output to audit, which does not exist yet (above).
  Once the pilot runs, the protocol is: sample 150 pilot rows, render one contact sheet
  of thumbnails per ~15-20 rows, review each blind (caption hidden until after the
  visual call), record `audit_verdict` (`consistent`/`hallucination`/`uncertain`) per
  row in `registry/captions/audit.tsv`, and report the per-claim rate with
  `consistency.wilson_ci(successes, n)` — the same 95% Wilson interval used for
  its 87.1% figure. No `audit.tsv` is shipped this pass since there is nothing real to
  put in it; shipping a fabricated one would be worse than shipping none.
- **Human verification (≥1,000 captions).** This needs human annotators and is an
  **external-annotator** item. The protocol is the audit TSV format
  above at 1k+ scale, ideally with more than one rater per row for inter-rater
  agreement; `registry/captions/schema.yaml`'s `audit_verdict` column is the landing
  spot for the result.

## The full run

By project decision, the full corpus is **not** captioned locally. Tier A is
already run in full above (69,600 + 18,561 rows, <1s). Tier B's cost is VLM inference,
not facts extraction; its throughput was not measured this pass (no pilot run), so no
projection is given here rather than a fabricated one — measuring `img/s` on even a
5-20 image smoke run of the pilot command above gives the number to multiply out for
both a local run (`rows / measured_img_per_s`) and a server Job (the same formula,
substituting the Job's GPU throughput). That projection is the v2 build's job.
