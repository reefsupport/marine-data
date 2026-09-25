# SPEC-w3 — deep, large sources (measured 2026-09-25)

Measured = read from upstream metadata (API counts, HF trees, bucket listings, CSV passes); `bytes_basis` in each spec says how bytes were obtained. Status = `scripts/spec_dryrun.py --ids-file` → `registry/ingest-specs/_queue-w3.tsv` (D-R4: ok with 0 items = `needs_adapter`). Fetch bytes = what an ingest moves: full measured bytes, or measured × target/items for subset sources, except treeoflife-10m (streams all shards) and planktonzilla (stages all, subsets at build).

## W3 catalog rows

| id | access | measured items | measured GB | subset target | licence mix | status | fetch GB |
|---|---|---:|---:|---:|---|---|---:|
| seattle-aquarium | hf | 6,522 | 59.0 | — | CC-BY-NC-4.0 (HF card) | ok | 59.0 |
| fathomnet | needs:fathomnet-api | 481,126 | 453.2 | — | per box: CC-BY-NC-4.0 / CC0-1.0 / CC-BY-4.0 (ann | needs_adapter:fathomnet-api | 453.2 |
| benthicnet | needs:frdr-globus | — | — | 400,000 | CC-BY-4.0 (mixed per source survey) | needs_adapter:frdr-globus | — |
| imos-auv | needs:manifest | — | — | 300,000 | CC-BY-4.0 (IMOS data licence) | needs_adapter:manifest | — |
| noaa-oer-video | needs:video | — | — | 500,000 | public domain (US government work) | needs_adapter:video | — |
| inat-marine | needs:manifest | 9,594,509 | 2,387.6 | 1,130,256 | per photo (photos.csv `license`): CC0 / CC-BY /  | needs_adapter:manifest | 281.3 |
| gbif-marine | needs:gbif-occurrence-media | 346,842 | — | 346,842 | per record: CC0-1.0 / CC-BY-4.0 / CC-BY-NC-4.0 ( | needs_adapter:gbif-occurrence-media | — |
| treeoflife-10m | needs:member-filter | 10,988,032 | 1,984.9 | 571,962 | HF card licence null; per image per upstream (EO | needs_adapter:member-filter | 1,984.9 |
| wikimedia-underwater | needs:commons-api | 14,247 | 77.0 | — | per file (Commons extmetadata LicenseShortName): | needs_adapter:commons-api | 77.0 |
| muot3m | hf | 3,074,016 | 256.6 | 303,000 | CC-BY-NC-ND-4.0 (HF card) | needs_adapter:video | 25.3 |
| planktonzilla | hf | 17,404,047 | 91.2 | 759,694 | other (HF card); per-sample licence = the origin | ok | 91.2 |

## NY rows (open-mirror check)

| id | access | measured items | measured GB | subset target | licence mix | status | fetch GB |
|---|---|---:|---:|---:|---|---|---:|
| marineinst20m | needs:annotation-overlay | — | 0.4 | — | annotations: repo LICENSE.txt; images: per upstr | needs_adapter:annotation-overlay | 0.4 |
| coralnet-public | needs:coralnet-browse | — | — | — | per source (set by each source owner) | needs_adapter:coralnet-browse | — |

## Discovery (open-direct specs)

| id | access | measured items | measured GB | subset target | licence mix | status | fetch GB |
|---|---|---:|---:|---:|---|---|---:|
| afrl-stereo-vi | hf | — | — | — | mit | needs_adapter:unsupported-format | — |
| elliott-bay-benthic | hf | 1,000 | 45.3 | — | cc-by-nc-4.0 | ok | 45.3 |
| nautdata | hf | — | — | — | apache-2.0 | needs_adapter:unsupported-format | — |
| nes-plankton-2022 | hf | 10 | 1.6 | — | cc-by-4.0 | ok | 1.6 |
| noaa-gfisher | hf | — | — | — | cc0-1.0 | needs_adapter:unsupported-format | — |
| noaa-oceaneyes | bucket | 20,065 | 1.8 | — | public domain (US government work) — NODD | ok | 1.8 |
| noaa-plankton-shadowgraphs | bucket | — | — | — | public domain (US government work) — NODD | needs_adapter:unsupported-format | — |
| plankton-interaction-videos | hf | 21,221 | 5.7 | — | apache-2.0 | ok | 5.7 |
| seaturtleid2022 | hf | 4 | 1.8 | — | ? | ok | 1.8 |
| uwbench | hf | 7 | 9.8 | — | ? | ok | 9.8 |

## Totals

- measured items 41,951,648 · measured 5,375.9 GB · ingest target 4,855,956 items (subset target, else all measured) · fetch 3,038.2 GB
- Job at 200 MB/s: 4.2 h · Mac at 10 MB/s: 84.4 h
- Unmeasured (null bytes): benthicnet (Globus listing), noaa-oer-video (NCEI 000), imos-auv (listing unfinished), coralnet-public, gbif-marine (per-publisher URLs), marineinst20m image side.
- Status counts: {'ok': 8, 'needs_adapter': 15}

Discovery rows (20, same 26 columns as the catalog): `registry/ingest-specs/w3-discovery.tsv`.
