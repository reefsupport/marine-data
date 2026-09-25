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
| noaa-pifsc-bleaching (1-image-labels; v1) | Filenames carry ESD site ids (`MAI-B4022`, `FFS-B009`, `LaehouShallow`; 18 island prefixes incl. Guam, Maug, Howland, Kure → Pacific-wide, no centroid). Site coordinates are in NCEI accession 0269246 (annotations+imagery) / 0259266 (2019 benthic cover): **unreachable 2026-09-25** (503 / timeouts). CoralNet source 2947 is public but exposes Region/Island/Site codes only, no per-image lat/lon | none (site-ready: `geo_backfill.noaa_site_id` + `table_join`) | site id from stem | [HF card](https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-CORAL-Bleaching-Dataset) · [InPort 67962](https://www.fisheries.noaa.gov/inport/item/67962) · [NCEI 0269246](https://www.ncei.noaa.gov/archive/accession/0269246) · [CoralNet 2947](https://coralnet.ucsd.edu/source/2947/) |
| coralscop-masks-rs (2026-09-23-3e8612678469; v1) | 20-char anonymised ids, 1044² crops; per-image COCO JSONs carry only file_name/size; no cross-source near-dup link to any geo-bearing source (dedup 2026-09-25) | none | — | `benthic_datasets/mask_labels/coralscop_masks/*/jsons/` |
| roboflow ×5 (v2i, v3i, v6i, v13i, v1-yolov8s; v1) | Roboflow exports: no location fields; some stems are Flickr photo ids (geotags need a Flickr API key → login, skipped) | none | — | `upstream/README.roboflow.txt` |
| rf100-coral-lwptl (rev-83f0a33679b0) | Roboflow-100 via HF; WP-6 schema, lat/lon null upstream | none | — | [HF](https://huggingface.co/datasets/LibreYOLO) |
| usis10k (rev-b10f41ab819b) | Salient-instance benchmark of web-sourced frames; no location | none | — | — |
| mermaid-aws (2026-09-19-bc53d5a2c0b6) | `mermaid/mermaid_confirmed_annotations.parquet` gives `region_name` per image (4 MEOW **realms**, e.g. Central Indo-Pacific) — realm only, no lat/lon; the schema's all-or-nothing MEOW triple forbids a realm-only row. Image → sample-event → site needs `api.datamermaid.org/v1/images/<id>/` = **401 (login) → skipped** | none (realm known) | image uuid | `s3://coral-reef-training/mermaid/` (anon) |
| coralscapes (1.0) | 35 numbered dive sites (`site10_…`) in 5 Red Sea countries; neither the HF card nor arXiv 2503.20000 publishes site coordinates; Red Sea ≫ 500 km → no centroid | none (site id known) | site number | [HF card](https://huggingface.co/datasets/EPFL-ECEO/coralscapes) · [arXiv](https://arxiv.org/abs/2503.20000) |
| reef-support-seaview-labels (legacy prefix, not staged) | SEAVIEW quadrat ids (`10001001601.jpg`) join Zenodo 3839924 quadrat CSVs (lat/lng per quadrat) → `image` via `table_join` once staged | image (when staged) | quadrat id | [Zenodo 3839924](https://zenodo.org/records/3839924) |

## Needs Yohan / external
- NCEI accessions 0269246 / 0259266 (NOAA ESD site coordinates) — retry when NCEI is back; then write
  `registry/geo/sites/noaa-esd.csv` and call `table_join(keys, noaa_site_id, sites, "site", url)`.
- MERMAID image → site: requires a MERMAID login (`api.datamermaid.org` returns 401) — skipped per D-E.
