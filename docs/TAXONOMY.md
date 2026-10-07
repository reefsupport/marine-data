# OBIS / WoRMS linkage — decision record

> Produced 2026-08-17 from OBIS API verification, an AGRRA/CoralNet conflict
> investigation, scale analysis, an independent critique and a synthesis.
>
> **Independently verified before adoption.** Every AphiaID was re-checked against the
> live WoRMS REST API, which confirmed all three reported errors. The OBIS `taxonID` =
> AphiaID linkage and the Montastraea→Orbicella resolution were confirmed against
> `api.obis.org/v3/taxon` directly.
>
> Implemented so far: the three AphiaID corrections, frozen WoRMS verification fields,
> the offline-guarantee test, and the CSafeLoader change. The rest is spec.

---

# FINAL DECISION — Linking the Label Library to OBIS/WoRMS

Status: decided. Every line is a directive.

---

## 1. Q1 — OBIS linkage: CONFIRMED, but AphiaID is an ATTRIBUTE, never a key

**How it works.** OBIS `taxonID` *is* the WoRMS AphiaID — same integer, no mapping table. Verified: `/v3/taxon/Orbicella annularis` → `taxonID: 758260`; `AphiaRecordByAphiaID/758260` → `AphiaID: 758260`. Every OBIS occurrence carries `aphiaID`, `scientificNameID` (LSID form), and an AphiaID-keyed rank ladder (`kingdomid … speciesid`). One integer joins us to OBIS occurrences, WoRMS taxonomy, and any other WoRMS-backed dataset, at every rank.

**Decision: node ids stay opaque project strings** (`HC_ORBICELLA`, `SD`, `TL`). AphiaID is a nullable field on `LabelNode` — which `src/marinedata/schema.py` already models correctly. Reason, measured: `registry/schemas/rs-benthic-v1.yaml` is 78 nodes, of which at most 29 can ever carry an accepted AphiaID. Promoting AphiaID to primary key orphans 63% of the library including every node with a committed crosswalk edge.

**Decision: the committed node is the pin.** "Pin to a WoRMS snapshot version" is not implementable — the REST API has no version/as-of parameter and silently ignores unknown params. Reproducibility comes from git. Every taxon node freezes five fields at resolve time:

```yaml
- id: HC_ORBICELLA
  name: Orbicella
  parent: HC
  axis: taxon
  worms_aphia_id: 758259
  worms_scientificname: Orbicella
  worms_rank: Genus
  worms_status: accepted
  worms_modified: "2015-07-20T15:23:11.000Z"
```

**What we fetch, when.** Nothing at load. Nothing in CI-on-PR. WoRMS is hit only by `marinedata taxa resolve`, a maintainer-run codegen step that rewrites YAML, and by a scheduled drift job that re-resolves every id and fails on any change to those five fields. OBIS is hit only by `marinedata prior build`, writing to `~/.cache/marinedata` (§6).

**Fix before anything is built on it** — 3 of 8 hand-typed AphiaIDs are wrong:

| node | wrong id | actually resolves to | correct |
|---|---|---|---|
| `MIL` | 196197 | Cryptonemiaceae — a **red alga** family, `unaccepted` | **205902** |
| `HC_PORITES` | 206970 | *Seriatopora stricta*, a species, `taxon inquirendum` | **206485** |
| `HC_ORBICELLA` | 1400001 | *Aspidostoma fallax* — a **bryozoan** | **758259** |

A 37.5% hand-entry error rate is the argument for the drift job, and for keeping the node set small enough that it runs in seconds.

---

## 2. Q2 — Scheme-scoped membership: JUSTIFIED, but it is the crosswalk we already have

The claim is true and understated. `Millepora`: AGRRA surveys it as a stony coral (shape group **FIRE CORALS**: `MILL`, `MCOM`, `MSQU`) while filing *M. alcicornis* under **"AGGRESSIVE" INVERTEBRATES** on the same page. Reef Check: "Also include fire coral (Millepora)" in Hard Coral. CATAMI 1.4: `Hydrocorals (11 077000)` is a **sibling** of `Corals (11 168000)`. NCRMP: Millepora is top-level category 10, distinct from category 1 Corals. CoralNet holds both simultaneously — label `79 Millepora` → Other Invertebrates (161,229 annotations) vs `1976 Millepora spp.` → Hard coral. Confirmed contested beyond Millepora/Cliona: `Heliopora`, `Tubipora`, `Stylaster`/`Distichopora`, zoanthids/`Palythoa` (four-way), corallimorphs, and the turf/substrate boundary.

**Two decisions follow, one of which deletes work.**

**(a) Scheme is not a fine enough key — the source label is.** CoralNet is one scheme carrying both answers, because the assertion rides on the *label*, not the scheme. Practitioners already hand-roll this: label `5384 Tubipora (HC)` → Hard coral exists alongside `634 Tubipora musica` → Other Invertebrates.

**(b) DO NOT BUILD A THIRD ARTIFACT.** `CrosswalkEdge` in `schema.py` is already keyed on `source_label` and already carries per-axis targets and a `Fidelity`. That *is* the scheme-scoped membership table. Adding a parallel `membership.yaml` duplicates the join and creates a second thing to keep consistent. Extend the existing edge with the source's own verbatim assertion:

```yaml
- source_label: "coralnet:79"
  targets: { taxon: MIL }
  fidelity: exact
  source_group: "Other Invertebrates"   # verbatim, never interpreted
  contested: true
```

`source_group` is opaque text. ReefSupport's canonical membership is the tree plus one flat concept-membership file for cross-clade groups (`hard_coral` = `HC` ∪ `MIL`, i.e. AphiaID 1363 ∪ 205902 — unreachable by single-parent traversal, which is the structural fact the whole design rests on). Comparability is then a query: any rollup whose contributing edges disagree on `source_group` is flagged non-comparable rather than silently pooled. This matters today — `registry/sources/coral-benthic.yaml` already carries `catami-1.4` (BenthicNet, Millepora-excluding) next to `coralnet-labelset` (both).

**Turf is not a membership problem.** AGRRA scores the overgrowth ("do not put RUB, DCS or PAVE"); Reef Check scores the substrate beneath. Same pixel, different class, by *rule*. Model it as a `scoring_rule: overgrowth | underlying` field on the source, not as a concept.

---

## 3. Q3 — Scale: **440 nodes now, ~1,100 ceiling. YAML in git. Permanently.**

The 17,357 framing was wrong twice: FishNet is not reef fish (8 taxonomic classes, freshwater included, 5.4 images/species — and `registry/sources/fish.yaml:107` already says it is not transect imagery), and dataset class counts are crosswalk *inputs*, never nodes.

The real ceilings: **114 zooxanthellate Scleractinia genera worldwide** (Veron), 85 well-defined. **ReefNet — 38 genera + Fungiidae = 39 classes — is the published state of the art** for coral from imagery, at ~56% cross-source macro recall. Nobody has demonstrated more. Caribbean scleractinians: ~25–30 genera. Video-scoreable Caribbean reef fish: ~80–100 species (GCRMN Level 2 scores **four families**).

**Tier 0 (build now): ~440 nodes** = 337 taxon (90 higher-rank spine, 28 Caribbean coral genera, 25 coral species, Millepora +3 spp, 16 octocoral genera, 12 sponge, 20 algae, 5 seagrass, 15 mobile inverts, 90 fish, 32 Indo-Pacific coral genera) + ~105 concept nodes. **Tier 1 ceiling ~1,100.** Tier 2 — FishNet 17,357, CoralNet 1,275, full WoRMS — are crosswalk inputs and live lookups, never nodes.

**Storage: sharded YAML in git, indefinitely. The Parquet backbone is cancelled.** Measured on this machine: 1,000 nodes = 145 KiB / 310 ms (40 ms with `CSafeLoader`). The binding constraint is *human review of a bulk edit*, ~400 nodes per file — not parsing, not GitHub's 3,000-line collapse, not the 1 MiB rich-diff limit. Shard by clade and region, 150–300 nodes per file. Git-diffable review is preserved as a first-class principle.

One-line perf change: `registry.py:28` `yaml.safe_load` → `yaml.load(fh, Loader=CSafeLoader)` with a pure-Python fallback. ~4× free.

---

## 4. The offline guarantee

**The artifact that makes it true is the committed YAML itself.** There is no runtime resolver, no cache warm-up, no network fallback.

- Resolution is codegen: `marinedata taxa resolve` (maintainer, local) → rewrites `registry/schemas/*.yaml` → reviewed as a PR diff.
- The drift job runs on `schedule` only, behind the existing `MARINEDATA_INTEGRATION=1` boundary already used by `verify-layouts`. `pyproject.toml:54` already defaults to `-m 'not integration'`. Reuse it; invent nothing.
- Enforcement test (write first):

```python
def test_registry_loads_with_network_hard_down(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("network access at load time")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    monkeypatch.setattr(socket, "socket", _boom)
    reg = Registry.load()
    assert reg.schema("rs-benthic-v1").node("HC_ORBICELLA").worms_aphia_id == 758259
    harmonize(...)  # and every loader
```

---

## 5. Non-taxon labels

`SD` (sand), `RB` (rubble), `RK` (rock/pavement, aliased "unknown hard substrate"), `TL` (transect line), `SCL` (scale reference), `DCA` (dead coral with algae) are identified by their **project node id — the primary key for every node, taxon or not**. They have no AphiaID and never will. `DCA` is two organisms plus a condition and already has a committed edge (`coralscapes:algae covered substrate`); an AphiaID-keyed tree would delete its parent `TRANSITION` and orphan it.

Add to `LabelSchema._validate_structure`, next to the existing cross-axis check:

```python
NON_TAXON_ROOTS = {"ABIOTIC", "TRANSITION", "NON_BENTHIC"}
if node.worms_aphia_id is not None:
    if node.axis is not Axis.TAXON:
        raise ValueError(f"{node.id}: worms_aphia_id on axis '{node.axis.value}'")
    if NON_TAXON_ROOTS & set(self.ancestors(node.id)) or node.id in NON_TAXON_ROOTS:
        raise ValueError(f"{node.id}: worms_aphia_id under a non-taxon root")
```

Plus: no two nodes may share a `worms_aphia_id`.

---

## 6. Q6 — Biogeographic priors: BUILD, as a cache, never as a mask

Build them — discrimination is real (Southern Caribbean vs Raja Ampat Scleractinia share only 10 of ~100 genera; *Orbicella* 2063/0, *Pocillopora* 0/25). But **recorded ≠ present**, and the error is anti-correlated with monitoring need: Haiti south coast returns **0 records / 0 species / 0 datasets** — a mask suppresses 100% of hard-coral prediction on a Haitian reef. Jardines de la Reina shows 21 genera vs Curaçao's 35 on the same fauna. Pure effort.

**Rules, all mandatory:**

1. **Never a mask, filter, or hard zero.** Permitted uses: (a) a bounded log-odds offset, `|Δlogit| ≤ 1.0`, incapable of moving a confident prediction across the boundary; (b) a **flag** — out-of-prior predictions route to human review as candidate range extensions. That flag is a product feature.
2. **Effort gate.** From `/statistics`: if `datasets < 3` or `records < 100` for a (region, clade) cell, the prior is uninformative and disabled → uniform. Haiti (0/0), São Tomé (2/11), N Mozambique (2/365) all correctly fail.
3. **Never weight by raw `records` across cells.** Curaçao = 8,414 records / 60 species; Raja Ampat = 729 / 146. Records is effort. Use within-cell relative frequency only.
4. **Temporal window, re-derived on schedule.** *Pterois volitans* in the wider Caribbean: 0 records pre-1985, 1 record 1995–2004, 2,735 in 2015–2026 — OBIS lagged the invasion by ~20 years. Regression test: assert a 1995-vintage prior does not suppress lionfish.
5. **Storage: regenerable Parquet in `~/.cache/marinedata`**, gitignored, manifest mirroring the existing `SHARD_MANIFEST.json` pattern. 232 MEOW × ~1,100 taxa × decade slices is ~255k rows — it must not be YAML. Commit only the one-time MEOW polygon set (Spalding 2007; not available from OBIS `/area`, and VLIZ geometry endpoints 404 — vendor the shapefile) and the query parameters.
6. **Licence: record OBIS at dataset granularity.** Data policy says mostly CC BY but some CC BY-NC/-ND/-SA; the AWS Open Data bulk export is labelled **CC BY-NC 4.0**. `dataset_id` is on every record. WoRMS taxonomy itself is CC-BY, so the backbone is clean even where occurrence data is not.

---

## 7. Q7 — Synonyms: resolve to accepted, record the origin

```
Montastraea annularis  AphiaID 207479  status "superseded combination"  family Montastraeidae
                                       valid_AphiaID → 758260
Orbicella annularis    AphiaID 758260  status "accepted"                family Merulinidae
```

**Always follow `valid_AphiaID`. Commit the accepted id. Record the origin.**

```yaml
worms_aphia_id: 758260
resolved_from: { queried: "Montastraea annularis", aphia_id: 207479, status: "superseded combination" }
```

This makes resolution idempotent and auditable. It matters operationally because AGRRA's codebook says "use ORBI for the *O. annularis* complex" while NCRMP scores to species — both strings genuinely arrive. OBIS resolves either transparently (all three query forms return the identical 28,307 occurrences), so a stale id still retrieves; the risk is two node ids for one organism, which the uniqueness assertion catches.

**Homonyms: the resolver REFUSES, it does not pick.** `AphiaRecordsByName/Turbinaria` returns two accepted genera and returns the **brown alga (206630, Ochrophyta) first**; the coral is 206641. `HC_TURBINARIA` currently has no AphiaID and `coralscapes-39.yaml` maps the bare string `"turbinaria"` to it. First-hit resolution files a brown alga under Hard coral. Contract, tests first:

```python
resolve("Turbinaria")  # raises AmbiguousTaxon
resolve("Turbinaria", expect_phylum="Cnidaria")  # -> 206641
```

Disambiguation is committed as `(name, authority)` or `(name, expect_phylum)` in the node.

---

## 8. Implementation order

**Phase 0 — correctness, no new features (do first, one PR).**
1. Fix `MIL` → 205902, `HC_PORITES` → 206485, `HC_ORBICELLA` → 758259.
2. Add the five `worms_*` provenance fields + `resolved_from` to `LabelNode`.
3. Add the two validators: AphiaID forbidden off-taxon/under non-taxon roots; AphiaID uniqueness.
4. Add the network-hard-down offline test.
5. `CSafeLoader` in `registry.py:28`.

**Phase 1 — resolver + drift.** `marinedata taxa resolve` (codegen, refuses ambiguity, follows `valid_AphiaID`); scheduled drift job behind `MARINEDATA_INTEGRATION=1`. Backfill AphiaIDs for the existing ~29 eligible nodes.

**Phase 2 — Tier 0 backbone to ~440 nodes**, sharded ≤300 nodes/file by clade and region. Each shard is its own reviewable PR. Concept layer and cross-clade membership (`hard_coral` = `HC` ∪ `MIL`) land here.

**Phase 3 — crosswalks as membership.** Add `source_group` + `contested` to `CrosswalkEdge`. Ingest the full CoralNet 12,263-label dump (already extracted) and AGRRA/CATAMI/NCRMP/Reef Check codebooks. Add the non-comparability query.

**Phase 4 — MEOW vendoring + `marinedata prior build`**, with the effort gate, offset cap, temporal window and the lionfish regression test.

**Deferred indefinitely:** Parquet backbone (cancelled), full WoRMS mirror, FishNet-as-nodes, a standalone membership artifact, any runtime network resolution.

🛠 Tools Used: None (synthesis of prior research + direct reads of `src/marinedata/schema.py`, `registry.py`, `registry/schemas/rs-benthic-v1.yaml`, `pyproject.toml`)