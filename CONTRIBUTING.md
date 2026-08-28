# Contributing

The most valuable contribution is **a verified licence**, not a large upload. We do not
host data — see the README on why.

## Adding a source

Add an entry to the appropriate file under `registry/sources/`, then run:

```bash
pytest
marinedata check --profile research
```

### Two rules

**1. `verified_by` must cite a primary source.**

A licence file you opened. A dataset card. Written permission. A terms-of-use page, with
the date you read it.

Not: "the paper says", "it's on HuggingFace so it's open", "everyone uses it".

This rule exists because we got it wrong ourselves. Our internal catalog recorded
MarineInst20M as "Mixed / Open" on the strength of a paper's phrasing. The `LICENSE.txt`
in the repository said CC-BY-NC-SA — over an image pool including Getty and Shutterstock.
Secondary sources are how that happens.

**2. Tier changes require a second reviewer.**

Raising a source's tier makes it usable in more contexts. That should be as deliberate as
changing a licence.

### Record what you don't know

An honest gap is more useful than a confident guess:

- Unsure of the licence → `NO-LICENCE-STATED` and `legal_basis: unknown`.
- Two sources disagree → set `disputed: true` with a `dispute_note`. The gate will block
  it on shipping profiles until resolved, which is the correct outcome.
- Unknown item counts → leave `items` unset rather than estimating.

### Worked example: a source with scalar labels

Most of the friction in adding a dataset is the crosswalk, not the source entry itself.
Here is the whole path for a source whose labels are per-image or per-point strings
(dense masks and point clouds skip step 2 entirely — their classes live in the raster,
which the label index never sees, so `marinedata check` never asks for a crosswalk).

1. **Add the source.** For a HuggingFace dataset, start with a draft:

   ```bash
   marinedata add owner/my-new-source          # prints a draft to stdout
   marinedata add owner/my-new-source -o draft.yaml
   ```

   This probes the Hub API and the datasets-server for a licence tag and column
   schema, and infers a layout — but every value it prints is a starting point, not a
   fact. In particular the licence is a dataset-card *claim*: rule #1 above still
   applies, and the draft's `verified_by` says so explicitly. Fill in the `TODO`s
   (description, capabilities, coverage, annotations — a probe cannot know what the
   labels mean scientifically), then paste the result under `registry/sources/<file>.yaml`:

   ```yaml
   - id: my-new-source
     name: My New Source
     licence: CC-BY-4.0
     legal_basis: licence
     provenance: public
     verification:
       verified_on: 2026-08-28
       verified_by: "HuggingFace card owner/my-new-source states license:cc-by-4.0"
       method: dataset-card
     access: { method: huggingface, uri: https://huggingface.co/datasets/owner/my-new-source,
               params: { hf_id: owner/my-new-source } }
     modalities: [image]
     capabilities: [benthic-classification]
     annotations:
       - kind: image-label
         supervises: [taxon]
     loader: { layout: image-folder }
   ```

   Run `marinedata doctor --incomplete-only` — it will list `my-new-source` with
   `layout=✗ crosswalk=✗`, which is exactly right: nothing has been verified against
   real data yet, and there is no crosswalk.

2. **Verify the layout against real data.**

   ```bash
   marinedata verify my-new-source
   ```

   This fetches a small bounded sample and confirms the declared layout actually reads
   it. If it fails, fix `loader.params` (or the layout itself) until it passes — do not
   guess twice; `marinedata fetch my-new-source` alone will show you what is really on
   disk.

3. **Add the crosswalk.** One entry under `registry/crosswalks/<file>.yaml` mapping the
   source's native label strings onto canonical node ids:

   ```yaml
   crosswalks:
     - id: my-new-source
       source_schema: dataset-native
       target_schema: rs-benthic-v1
       edges:
         - { source_label: "Hard Coral", targets: { taxon: HC }, fidelity: exact }
         - { source_label: "Soft Coral", targets: { taxon: SC }, fidelity: exact }
   ```

   Reference it from the source entry: `loader: { layout: image-folder, schema_id:
   dataset-native, crosswalk_id: my-new-source }`.

   An edge that cannot map cleanly is not a blocker — see "Record what you don't know"
   below and `docs/LABELS.md` for `fidelity` and abstention.

4. **Verify the crosswalk against real data.**

   ```bash
   marinedata labels my-new-source
   ```

   This audits the crosswalk's edges against the labels a real fetched sample actually
   contains — catching a mis-registered label string before it silently drops
   supervision. It has already caught two mis-registrations in this registry.

5. **Done.** `marinedata doctor --incomplete-only` should no longer list
   `my-new-source`, and `pytest` should still pass. If the source has no annotations at
   all (a pretraining-only image dump), skip steps 3–4 entirely — `doctor` reports
   `crosswalk=·` (not applicable) rather than asking for one that would be fictional.

### Domain shift is part of the entry

If a source is regionally biased, say so. `missing_classes` in particular: a schema that
omits a taxon abundant in some region will fail there in a way accuracy metrics on the
source's own test set will never reveal. This field is the one thing in the registry that
no other catalog carries, and it is the most common cause of deployment failure in marine
CV.

Measured drops (`known_drops`) need a real citation — a paper, or reproducible internal
measurement described specifically enough to check.

## Code

- `ruff check . && ruff format --check .`
- `pytest` — the gate tests are non-negotiable. If you change gate semantics, the matrix
  in `tests/test_gate.py` must change with it, deliberately and in the same commit.
- Models are frozen (`ConfigDict(frozen=True)`). A registry is a set of assertions; code
  that mutates one in place can silently disagree with a lineage report it already emitted.

## Scope

In scope: marine and aquatic datasets of any modality — imagery, video, geometry, audio,
molecular, remote sensing.

Out of scope: hosting data, model weights (a separate registry may come later), and
anything requiring us to redistribute restrictively-licensed material.

## Conduct

Be accurate, be correctable. If you think a tier is wrong, open an issue — being
corrected is the point.
