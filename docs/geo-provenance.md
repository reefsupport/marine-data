# Geo provenance — where each source's coordinates live (WP-2c, 2026-09-25)

EXIF is not a source of geography here: 0/69,600 v1 images carry EXIF GPS (WP-2b full scan), and the
bucket originals are EXIF-stripped too (`reef-support-benthic-own` JPGs are JFIF, no APP1; Range-GET
re-checked 2026-09-25). Geography therefore comes only from what upstream *documents*. Joins are
implemented in `src/marinedata/geo_backfill.py` → `registry/geo/backfill/<source>.parquet`, consumed
by `metadata_release.build_rows()` (join key `<partition>/<stem>`). Precision classes:
`image` > `site` > `source_centroid` (a whole source, or one upstream partition naming one documented
place, only if ≤ 500 km across) > `none`. Location-sensitive 0.1° rounding applies on top.

| Source (bucket version) | Where coordinates live | Precision | Join key | Evidence |
|---|---|---|---|---|
| reef-support-benthic-own (2026-09-24; v1) | Upstream partition names one place each: `UNAL_BLEACHING_TAYRONA` → Tayrona NNP; `SEAFLOWER_BOLIVAR` → Cayo Bolívar; `SEAFLOWER_COURTOWN` → Cayos del Este Sudeste; `TETES_PROVIDENCIA` → Providencia. Whole source spans ~830 km (Tayrona ↔ San Andrés) → no source-level centroid. No coords in Labelbox `export-result.ndjson` (`metadata_fields: []`). Capture date from filename (`20220912_…`, `…_26sep22_…`) for 1,143/1,250 | source_centroid (partition) | partition/stem | [Tayrona](https://en.wikipedia.org/wiki/Tayrona_National_Natural_Park) · [Cayo Bolívar](https://www.openstreetmap.org/way/20257555) · [Courtown](https://www.openstreetmap.org/relation/10757870) · [Providencia](https://en.wikipedia.org/wiki/Providencia_Island,_Colombia) |
| reef-support-bleaching (2026-09-24) | Single partition `UNAL_BLEACHING_TAYRONA` → Tayrona NNP; dates from filename (658/658) | source_centroid | partition/stem | as above |
| reefolution (2026-09-23-2c84cb0c9cda) | Project site: "restored corals … on the Kenyan coast"; Kenyan coast Vanga↔Kiunga ≈ 409 km (≤ 500) → source centroid (endpoint mean). No site names in `annotations.csv` (Name,Row,Column,Label) | source_centroid | stem | [reefolution.org](https://reefolution.org/) · [Vanga](https://www.openstreetmap.org/way/257169285) · [Kiunga](https://www.openstreetmap.org/node/44933213) |
| noaa-pifsc-bleaching (1-image-labels; v1) | Filenames carry ESD site ids (`MAI-B4022`, `FFS-B009`, `LaehouShallow`; 18 island prefixes incl. Guam, Maug, Howland, Kure). NCEI accession 0269246 was **unreachable on the first retry (2026-09-25 ~02:26 UTC), back up on the second (~02:37 UTC)**; its `NOAA_ESD_CoralBleachingClassifier_ImageMetadata.csv` (184 sites, `registry/geo/sites/noaa-esd.csv`) joins via `geo_backfill.noaa_site_id` + `table_join` → 10,186/10,419 rows (97.8%) at `site` precision. The unjoined ~2.2% are Guam/Maug codes, out of scope for these two Hawaiian-Archipelago accessions | site | site id from stem | [HF card](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset) · [InPort 67962](https://www.fisheries.noaa.gov/inport/item/67962) · [NCEI 0269246](https://www.ncei.noaa.gov/archive/accession/0269246) |
| coralscop-masks-rs (2026-09-23-3e8612678469; v1) | 20-char anonymised ids, 1044² crops; per-image COCO JSONs carry only file_name/size; no cross-source near-dup link to any geo-bearing source (dedup 2026-09-25) | none | — | `benthic_datasets/mask_labels/coralscop_masks/*/jsons/` |
| roboflow ×5 (v2i, v3i, v6i, v13i, v1-yolov8s; v1) | Roboflow exports: no location fields; some stems are Flickr photo ids (geotags need a Flickr API key → login, skipped) | none | — | `upstream/README.roboflow.txt` |
| rf100-coral-lwptl (rev-83f0a33679b0) | Roboflow-100 via HF; WP-6 schema, lat/lon null upstream | none | — | [HF](https://huggingface.co/datasets/LibreYOLO) |
| usis10k (rev-b10f41ab819b) | Salient-instance benchmark of web-sourced frames; no location | none | — | — |
| mermaid-aws (2026-09-19-bc53d5a2c0b6) | `mermaid/mermaid_confirmed_annotations.parquet` (re-fetched WP-2d, 481,575 rows) and the per-image `_annotations.csv` sidecars carry only `image_id`/`point_id`/`benthic_attribute_*`/`region_id`/`region_name` — **no** sample-event, site or project id anywhere in the bucket. `region_name` is realm only (WP-2d re-check: 3 realms + 750 null; Central/Western Indo-Pacific, Tropical Atlantic — thousands of km, no centroid). WP-2d re-tried the manager's lead, `GET /v1/summarysampleevents/` (public, 200, no auth): it lists MERMAID's own surveys (`sample_event_id`, `site_id`+lat/lon, `project_id`) but carries **no image id** to join against — there is no field in either direction linking an `image_id` UUID to a `sample_event_id`. The one endpoint that would (`/v1/images/<id>/`) still 401s (confirmed again 2026-09-25) → skipped per D-E. Source coverage is `global`, so no source-level centroid is legal either | none (realm known, unjoinable) | image uuid | `s3://coral-reef-training/mermaid/` (anon) · [summarysampleevents](https://api.datamermaid.org/v1/summarysampleevents/) (public, no image-id field) |
| coralscapes (1.0) | 35 numbered dive sites (`site10_…`) in Djibouti, Eritrea, Sudan, Jordan and Israel (arXiv 2503.20000 §3: "All imagery was collected during scuba dives at 35 sites… using GoPro Hero 10 cameras… **the location of the sites is withheld and instead replaced by an ID**" — an explicit anti-poaching/overtourism redaction, not a missing-metadata gap). Neither the HF card nor the paper/appendix publishes a site→country map or coordinates, so even a country-level centroid can't be assigned; the 5 countries' Red Sea coastline spans Djibouti (~11.5°N) to the Gulf of Aqaba (~29.5°N), ≈ 2,200 km ≫ 500 km, so a centroid would fail D-W even if the mapping were known | none (site id known, coords withheld by design) | site number | [HF card](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) · [arXiv 2503.20000 §3](https://arxiv.org/abs/2503.20000) |
| reef-support-seaview-labels (legacy prefix, not staged) | SEAVIEW quadrat ids (`10001001601.jpg`) join Zenodo 3839924 quadrat CSVs (lat/lng per quadrat) → `image` via `table_join` once staged | image (when staged) | quadrat id | [Zenodo 3839924](https://zenodo.org/records/3839924) |

## Needs a maintainer decision / external
- NCEI accessions 0269246 / 0259266 (NOAA ESD site coordinates) — resolved 2026-09-25. WP-2d's bounded
  retry (up to 3×, 10 min apart, `perl alarm 700`) found NCEI down on attempt 1 and back up on attempt 2;
  `registry/geo/sites/noaa-esd.csv` + `registry/geo/backfill/noaa-pifsc-bleaching.parquet` now carry the
  join (10,186/10,419 rows, `site` precision). Nothing further needed here.
- MERMAID image → site: requires a MERMAID login (`api.datamermaid.org/v1/images/<id>/` returns 401) —
  skipped per D-E. WP-2d confirmed the public `/v1/summarysampleevents/` endpoint the manager found does
  not carry an image id either, so there is no lawful join today; would need MERMAID to publish one.
- Coralscapes: site coordinates are deliberately withheld by the authors (anti-poaching/overtourism). Do
  not attempt to re-derive them (e.g. from EXIF-adjacent metadata, dive-log correlation, etc.) even if a
  future upstream release adds a country field — respect the redaction; a country-level centroid would
  still fail D-W's 500 km rule regardless.

## iNaturalist: coordinates are generalised on purpose (WP-U14b/U14c decision)
The inat-marine manifest stages `latitude`, `longitude` and `observed_on` but no geoprivacy flag, so
we cannot tell an obscured observation from an exact one. Every iNat position is therefore kept at
0.1 deg (`gps_precision_m = 22000`, `location_generalized = true`) and the time is date only
(`observed_on` becomes 00:00Z; provenance says "date only"). Coordinates are not nulled.

## Next-5 sources (WP-U14c, inspected 2026-10-05 in `metadata.parquet` heads; extractor code in
`metadata_norm/next5.py`)
| Source | Per-row geo / depth / time staged | Result |
|---|---|---|
| pingmapper-sss-seg | recording name `<site>_<YYYYMMDD>_<unit>_Rec<n>_wcp_ss_<side>_<chunk>` in `upstream_id` | `capture_datetime` = that date (date only). No position: the site is a river/lake name |
| marineevt | none: video frames; `upstream_id` has the video id, QA json is not joined | none staged (`sources/marineevt/rev-37488d3c7690/metadata.parquet`) |
| sonarsweep | none: simulated sonar, degraded optical frames and USD textures | none staged (`sources/sonarsweep/rev-350b20a9acaf/metadata.parquet`) |
| nes-plankton-2022 | none: `upstream_id` is `data/train-NNNNN.parquet#row`; no IFCB sample id | none staged (`sources/nes-plankton-2022/rev-18fb2e57fd77/metadata.parquet`) |
| aqqua-baltic-holo | month and station only (`Finland_April2024_St15`); the per-station csv is not staged | none staged (`sources/aqqua-baltic-holo/record-18405776/`: no csv in the tree) |

Instrument (`camera` column) comes from registry `default_instrument`, provenance `registry_default`:
aris-didson-fish-td = imaging-sonar, pingmapper-sss-seg = side-scan-sonar, aqqua-baltic-holo =
holographic-imager, nes-plankton-2022 = ifcb. sonarsweep has none (mixed simulated modalities).
