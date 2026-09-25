# `registry/geo/` provenance

## MEOW — Marine Ecoregions of the World

- **Citation:** Spalding MD, Fox HE, Allen GR, Davidson N, Ferdaña ZA, Finlayson M,
  Halpern BS, Jorge MA, Lombana A, Lourie SA, Martin KD, McManus E, Molnar J, Recchia CA,
  Robertson J (2007) *Marine Ecoregions of the World: a bioregionalization of coast and
  shelf areas.* BioScience 57: 573-583.
- **Source tried, in the brief's order:**
  1. TNC ArcGIS Hub GeoJSON/shapefile export — the item id used by the public "Marine
     Ecoregions of the World" ArcGIS Hub page (`opendata.arcgis.com` REST export) returned
     HTTP 500 on the anonymous export endpoint; the FeatureServer query guess returned 400.
     No working anonymous URL found without going through the ArcGIS Hub UI.
  2. `marineregions.org` / VLIZ geoserver WFS `Ecoregions:ecoregions` — not tried directly
     this pass; `docs/TAXONOMY.md` (WP-2, Phase 4) already recorded these endpoints as
     dead/404 from an earlier check.
  3. **GitHub mirror (used):** `seananderson/paleobaselines` vendors the canonical
     `meow_ecos.shp` shapefile (with its original Esri XML metadata) verbatim at
     `data/MEOW2/meow_ecos.{shp,shx,dbf,prj}`. Verified record/attribute counts match the
     brief's requirement exactly: 232 ecoregions, 62 provinces, 12 realms.
- **Retrieved:** 2026-09-25T00:46Z, via `raw.githubusercontent.com` (anonymous HTTPS GET,
  no login).
- **Retrieved URLs:**
  - `https://raw.githubusercontent.com/seananderson/paleobaselines/master/data/MEOW2/meow_ecos.shp`
  - `https://raw.githubusercontent.com/seananderson/paleobaselines/master/data/MEOW2/meow_ecos.dbf`
  - `https://raw.githubusercontent.com/seananderson/paleobaselines/master/data/MEOW2/meow_ecos.shx`
  - `https://raw.githubusercontent.com/seananderson/paleobaselines/master/data/MEOW2/meow_ecos.prj`
  - `https://raw.githubusercontent.com/seananderson/paleobaselines/master/data/MEOW2/meow_ecos.shp.xml`
    (Esri FGDC metadata — carries the licence text below, quoted from the `<useconst>` tag)
- **sha256 of the raw downloads (not committed — only the simplified derivative is):**
  - `meow_ecos.shp`: `9e958e389308ff49b49e3faca3c594d004e446b22d6830070a97c80c75ba791`
  - `meow_ecos.dbf`: `a1988be1ddffba3001f4f048be5eb281291d1380481110f042b1c5233bfe82d`
  - `meow_ecos.shx`: `7c8a9c5b70dcd496021397265dfca4695b3dda231bd2f5a5ffb235152be171b`
  - `meow_ecos.prj`: `a02a27b1d1982c8516d83398e85a3c8b1aef1713c13ef4d84d7bde17430c07c`
- **Licence text (verbatim, from `meow_ecos.shp.xml` `<useconst>`):**

  > Any modification of the original map to ecoregion boundaries, units, names, or realm
  > and province classes must be noted and explained alongside the original citation.
  > The MEOW Working Group (authors Spalding et al.) shall not be held liable for improper
  > or incorrect use of the data described and/or contained herein. **Any sale,
  > distribution, loan, or offering for use of these digital data, in whole or in part, is
  > prohibited without the approval of the MEOW Working Group.** The use of these data to
  > produce other GIS products and services with the intent to sell for a profit is
  > prohibited without the written consent of the MEOW Working Group. All parties
  > receiving these data must be informed of these restrictions. The MEOW Working Group
  > and its members' respective organizations shall be acknowledged as data contributors
  > to any reports or other products derived from these data.

- **What this means for us:** the polygon geometry itself is used only inside this
  registry, as an internal classifier that derives `meow_realm`/`meow_province`/
  `meow_ecoregion` string labels for staged images that carry GPS. Those three derived
  string fields — not the polygon geometry — are what could ever reach `metadata_release`
  output or an HF card. **`registry/geo/meow-2026-09-25.parquet` (the vendored, simplified
  polygon set) must not itself be published, redistributed, or bundled into the HF
  release or S3 imagery bundles** — that would be exactly the "distribution... in whole
  or in part" the licence prohibits without the MEOW Working Group's approval. It is
  vendored here for internal, non-commercial ecoregion classification only. Flagging for
  Yohan: if MEOW-derived per-sample labels are later published, credit the citation above
  alongside them (the licence's one attribution requirement that *is* compatible with
  publishing derived labels).
- **Vendored file:** `registry/geo/meow-2026-09-25.parquet` — columns `eco_code` (int),
  `ecoregion` (str), `prov_code` (int), `province` (str), `rlm_code` (int), `realm` (str),
  `wkb` (bytes, WKB Polygon/MultiPolygon simplified to 0.01° tolerance,
  `shapely.simplify(..., preserve_topology=True)`). 232 rows, 0.26 MB (well under the 20 MB
  cap). CRS: WGS84 (`GCS_WGS_1984`, confirmed from `meow_ecos.prj`).
