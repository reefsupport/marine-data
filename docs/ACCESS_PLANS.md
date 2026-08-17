# Access plans for sources that cannot be auto-fetched

`marinedata verify` reports four states. `verified` and `metadata-only` need nothing.
This document covers the rest: what the actual obstacle is per source, and the specific
route through it.

The obstacles are not interchangeable. Treating "no small file published" the same as
"a human must accept terms" produces either wasted engineering or a licence breach, so
they are separated here by **mechanism**, not by dataset.

---

## Route A — HuggingFace mirrors (do this first)

The field is actively mirroring onto HuggingFace, and a mirror needs no new code: the
existing `huggingface` fetcher handles it. Searching before building a bespoke fetcher
has already changed three answers.

**Confirmed live 2026-08-17:**

| Source | Mirror | Structure |
|---|---|---|
| `uieb` | `Hikari0608/UIEB` | `raw:Image`, `gt:Image` — exactly `image-mask-pairs` |
| `brackish-dataset` | `moondream/brackish_underwater` | `image:Image`, `objects:List` |
| `benthicnet-1m` | `ndotm/benthicnet_labelled` | `url, source, dataset, site, image, lat, lon` |
| `reefnet` | `ReefNet/ReefNet-1.0` | tabular annotations — already registered as `reefnet-hf` |

**BenthicNet is the one that matters.** That mirror is the annotation index *with image
URLs*, which makes 2.6M CATAMI annotations addressable — and CATAMI is the highest-value
missing crosswalk. It needs one new layout: a **URL-following reader** for index-style
datasets whose rows carry links rather than bytes (~40 lines).

**Not found on HF:** `ozfish`, `deepfish`, `seaclear-marine-debris`,
`atlantis-synthetic-depth`. Re-check periodically; mirrors appear.

**Method:** search `huggingface.co/api/datasets?search=<name>`, then confirm via
`datasets-server.huggingface.co/first-rows` before writing an entry. Read the real
features — do not trust the dataset card, and do not trust ours either.

---

## Route B — declare a `sample_url` (no new code)

For sources publishing at least one small directly-downloadable file. One line in
`access.params`, and the existing `http`/`zenodo` fetcher takes it.

```yaml
access:
  method: http
  params:
    sample_url: https://example.org/dataset/sample_100.zip
```

Candidates: `suim`, `squid`, `fish4knowledge`, `labeled-fishes-in-the-wild`,
`sea-urchin-detection`, `heron-reef-benthic`, `fishnet-2023`.

**Rule:** the URL must be a *stable, publisher-hosted* file. Do not point at a mirror
someone might delete, and do not scrape a page hunting for one — an entry that silently
rots is worse than one honestly marked unfetchable.

---

## Route C — Kaggle single-file API

[`kaggle datasets download -d owner/name -f <file>`](https://github.com/Kaggle/kaggle-api/issues/313)
downloads **one file** rather than the whole archive — which is precisely the bounded
sample we want. Needs `KAGGLE_USERNAME` / `KAGGLE_KEY`.

Relevant: `flsea` (`viseaonlab/flsea-vi`), plus several fish sets. Note many Kaggle
*competition* datasets carry competition-only terms — check before adding, because
"available on Kaggle" is not a licence.

**Work:** a `kaggle` access method plus a fetcher, ~60 lines. Credentials belong in env
vars via the existing `credentials_env` pattern, never in the registry.

---

## Route D — FathomNet API

[`fathomnet-py`](https://github.com/fathomnet/fathomnet-py) is a real REST client:
`images.find_by_concept('Nanomia')` returns records with URLs. Fully programmatic.

Deep-sea, so low reef relevance — worth doing for completeness and for the megafauna
capability, not for benthic transfer. **Work:** ~40 lines, or an optional dependency.

---

## Route E — GitHub raw + tree API

Same shape as the `hf-files` fetcher already written: list the tree, download small
files. Applies to `floating-marine-debris` (data in-repo) and
`atlantis-synthetic-depth`.

---

## Route F — publish our own sample mirror ⭐ the systemic fix

Routes A–E each unlock a few sources. This one unlocks **all permissive sources at
once**, and it is the only route that scales.

For CC-BY, CC0 and public-domain sources we may lawfully redistribute a ~100-item
sample. Publishing those to a `reefsupport/marine-data-samples` HuggingFace repo makes
every one of them CI-verifiable regardless of how awkward its origin is, and it survives
upstream outages — the CoralVQA failure is a datasets-server bug we cannot fix from here.

**Hard constraint:** only `T0_OWN`, `T1_PERMISSIVE` and (with share-alike honoured)
`T2_COPYLEFT` may be mirrored. Mirroring an NC or ND source would be exactly the
redistribution breach the whole registry exists to prevent. The gate must enforce this
mechanically — a `mirrorable` property derived from tier and flags, not a human
remembering.

**Also required:** per-item attribution in the mirror, and a `MIRROR_LINEAGE.json`
recording origin, licence and fetch date for every sampled file.

This needs a decision, not just engineering: it means publishing under our name and
standing behind the redistribution analysis per source.

---

## Genuinely unreachable — and correctly so

### Zenodo monoliths

Zenodo returns no `accept-ranges` (verified 2026-08-17), so partial extraction from a
zip is impossible — you either take the whole archive or nothing.

| Source | Size | Route |
|---|---|---|
| `deolhonoscorais` | 5.1 GB single zip | one-off manual download |
| `deepreefmap-data` | 10.3 GB | one-off manual download |
| `varos` | 18.0 GB | one-off manual download |
| `reefset-v1` | 1.6 GB | borderline — could be a whole-file `sample_url` |

**Plan:** download once, extract to a stable path, point `roots` at it, and record the
verified layout. This is not a tooling gap; it is a one-time human action, and pretending
otherwise would mean building a downloader that pulls 18 GB in CI.

### CoralNet — scraping

`coralnet.ucsd.edu/api/` returns **404**; there is no public API. Access means scraping
per source, as the CoralNet-Toolbox does.

**Prerequisites, in order:**

1. **Counsel opinion on Art. 15o Auteurswet.** `pretrain-eu` already refuses this source
   without a `legal_opinion_ref`, so the code will not let it in early.
2. **Re-check `robots.txt`** at scrape time. It was absent (404) on 2026-08-17, which is
   favourable under the Dutch machine-readable test but is a fact with a date, not a
   standing permission.
3. **Per-source licence check.** CoralNet publishes no blanket licence and sources set
   their own terms.
4. **Rate limiting and identification.** A real User-Agent with contact details, modest
   concurrency, backoff. We are asking a small academic service for a favour.
5. **Log lawful access per item.** That log *is* the defence.

**Only then** is a scraper worth writing. Building it before step 1 would be building
something we might have to delete.

### Gated forms

`coralmask` requires completing a Google Form. A human must accept the terms, and that
assent is plausibly a contract that overrides the EU TDM exception (DSM Art. 7 protects
Arts. 3/5/6 but not Art. 4). The `contract_gated` flag exists to make this visible;
automating past it would be the one failure mode with real legal consequence.

**Plan:** do not automate. If we want it, someone signs, and we record the permission in
`verification.method: written-permission`.

### Request-only

`seamapd21` needs an email to the maintainers. Worth sending — Gulf of Mexico is the
closest large fish-detection set to Caribbean assemblages.

---

## Priority

1. **Route A** — HF mirror sweep + the URL-following layout. Highest verified-count per
   hour, and it unlocks BenthicNet/CATAMI.
2. **Route B** — `sample_url` pass. Hours, no new code, no decisions.
3. **Routes C/E** — Kaggle and GitHub fetchers.
4. **Route F** — the sample mirror. Needs a decision first.
5. **Zenodo manual downloads** — as the datasets are actually needed, not speculatively.
6. **CoralNet** — after counsel. Not before.

Realistic ceiling: routes A–E reach roughly 30 verified of ~37 loadable. Route F reaches
essentially all permissive sources. The remainder are correctly unreachable, and the
registry should keep saying so.
