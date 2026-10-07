# SPEC-w4 — deep-sea, plankton-at-scale and catalog-gap discovery (2026-09-25)

**DISK-CRITICAL STOP.** Free disk dropped from 29.6 GiB to 9.6 GiB during this wave (not caused by
this worker — no downloads were made; a `uv sync` venv, since removed, was the only local write
before this). The coordinator's hard floor (`< 20 GiB`) and the brief's own floor (`< 29 GiB`) are
both breached. `ruff`, the full `pytest` suite and the commit were **not run**; see the report.

40 candidates specced, 0 downloaded (verification was anonymous HEAD/API/listing calls only, per the
Do-NOT). Sorted by value: labels > depth novelty > size.

## Status totals
| status | n |
|---|---|
| ok (adapter exists: hf/zenodo) | 12 |
| needs_adapter | 15 |
| needs_yohan (login/token/request) | 8 |
| unreachable | 4 |
| dedup (already covered elsewhere) | 1 |
| **total** | **40** |

Deep-sea (>200 m) candidates: **20** of 40 (`habcam` is catalogued deep-sea-adjacent but its
HabCam shelf survey operates 40–120 m, so it is excluded from this count).

## Top 10 to stage first
1. **deepsea-mot** (ok, hf, 20.0 GB) — only genuine deep-sea multi-object-tracking benchmark found; official MBARI org, cc-by-sa-4.0.
2. **whoi-plankton** (ok, hf, 27.3 GB, 956,867 images) — largest net-new plankton-at-scale set with a clean adapter.
3. **plc-beijbom2015** (ok, zenodo, 9.08 GB, 5,090 images, CC0) — CoralNet-lineage reef points, highest labels-per-GB.
4. **fathomnet-vme** (ok, hf, 0.06 GB) — Vulnerable Marine Ecosystem indicator-taxa benchmark, zero-cost add.
5. **fathomnet-megalodon** (ok, hf, 0.29 GB) — object-detection benchmark, zero-cost add.
6. **noaa-pifsc-esa-coral-icra** (ok, hf, 10.3 GB, 944 files) — official NOAA reef-structure/disease imagery.
7. **coralnet-small** (ok, hf, 6.47 GB, 450 images) — small but official CoralNet-lineage slice; verify content density before committing disk.
8. **whoi-plankton-small** (ok, hf, 1.11 GB, MIT-licensed) — cleanest-licence plankton option if whoi-plankton's unstated licence is a blocker.
9. **planktonset-1-0** (ok, hf, 0.11 GB, 60,736 images) — smallest, cheapest plankton add; closest open equivalent to Kaggle NDSB 2015.
10. **fathomnet-fgvc23** (needs_adapter:coco-remote-fetch, ~16 GB, 10.7k images, 290 taxa) — highest-value deep-sea item still needing a small COCO-loader adapter.

Everything else (needs_adapter with unresolved size, needs_yohan, unreachable) is listed in
`_queue-w4.tsv` with its blocker in the `dry_run` column.

## Notable dedup / non-findings
- `zooscannet`, and very likely `planktonset-1-0` and the rest of the `project-oceania` per-instrument
  HF org (`uvp6net`, `flowcamnet`, `isiisnet`, `zoocamnet`, `globaluvp5net`, `syke_ifcb_2022`,
  `sykezooscan2024`, `medplanktonset`, `jedi_system_oceans_cpics`, `zoolake`, `planktoscope`) appear to
  be the ~15 instrument feeds already pooled into `planktonzilla-17M` (W3, staged in full at 91 GB).
  Only `zooscannet` got its own spec (brief named it explicitly); the rest are **not** speced separately
  to avoid ~10 near-duplicate entries — flagging here instead.
- `kaggle-ndsb-2015`: no anonymous public mirror of the exact 2015 competition split was found;
  `planktonset-1-0` is the nearest open equivalent from the same lineage.
- `marinedet`: no canonical public repository was identified (GitHub search inconclusive) — needs
  a maintainer to name the exact source before it can be specced for real.
- `pangaea-ofos-msm77`: this is the *baseline* the brief said to search "beyond" — it lacked an
  ingest-spec, so it is specced here for completeness, not counted as a new discovery.
- 6 PANGAEA "dataset publication series" (`pangaea-ccz-gsr`, `pangaea-ofos-msm77`,
  `discol-peru-basin-ofos`, `hacon-aurora-vent-ofos`, `so268-ofos-analysis-ready`) share one blocker:
  each parent DOI fans out to 1–13 child dive/profile DOIs with no flat bulk-zip. A single
  `needs:pangaea-series-enum` adapter would unblock all 6 at once (~worth building first among the
  `needs_adapter` group).
