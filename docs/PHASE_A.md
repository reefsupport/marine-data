# Phase A: what landed in `feat/phase-a`

P1.13 merged `feat/registry-truth` (864f1ef) and `feat/card-croissant` (3af3ffc, holding `feat/join` 8b35ac9 and `feat/gates` fca3376).
Briefs and reports (`reports/2026-10-08-PA-<id>.md`) live in the session state, outside the repo.

## Landed, one line per brief
- P1.2: registry is the single truth: every ingest-spec id has an entry (`scripts/registry_fold_specs.py`).
- P1.2b: evidence-backed licences, root digests, CoralscapesV2 and the reconstructed v1.0 lock.
- P1.2c: `feat/registry-truth` made green.
- P1.2d: licence evidence for every entry that sets a class, plus guards (`marinedata check`: 174 permitted / 79 expected / 0 unexpected).
- P1.3: one report with P1.5 (`PA-P1.3-5`): `marinedata check` splits expected from unexpected denials.
- P1.4: signed bucket reads and fetch integrity (`s3_auth`, `s3_client`).
- P1.4b: every S3 constructor goes through the resolver; `fetch.py` became the `fetch/` package.
- P1.5: see P1.3: `release lock` / `release build --lock` and the masks gate.
- P1.6: release gate suite and docs/GATES.md.
- P1.7: release build image, VM bootstrap, `release publish` (dry run unless `--go`).
- P1.9: dataset card and Croissant generators, with golden files.
- P1.9b: gate suite merged into the card branch; one Croissant emitter; one `release gate` subparser.
- P1.10: generated docs (`make docs`), CI gates (`make ci`), SETUP leftovers, seqinfo revert.
- P1.13: this integration: `docs/cli.md`, `tests/test_cli_docs.py`, the `task_layers/configs/` package, gate and publish rework.
- P2.1: per-row metadata resolvers: geo, date, depth, site, quadrat.
- P2.2: one report with P2.4 (`PA-P2.2-4`): SampleRow v3 and the per-row licence layer.
- P2.4: see P2.2: `release gate --metadata` (resolver coverage floors, per-row licences).
- P2.5: join integration: points onto join; real-data metadata and licence gate run.
- P2.6: points config carries geo, date and licence; platform evidence.
- P3.1: Seaview benthic points table and config producer.

## CLI surface added
`auth check`, `metadata resolve`, `tasks build-points --metadata-dir`, `release lock|gate|card|croissant|publish`; every flag is in [cli.md](cli.md).

## Gates, lock, credentials, upload
- Gates (`release gate BUILD`): licence, dedup, splits, masks, load, card are required; privacy, croissant, metadata run on request. GATES.json is one dict keyed by gate name plus `summary`.
- `--require-all` runs every gate and requires each one; an absent, skipped or failed gate refuses, privacy included (a gate that cannot run means the build has not passed). `release publish` is fail-closed: it takes its verdict from `marinedata.gates` and refuses unless GATES.json has a passing entry for all eight gates (privacy and croissant included; an absent or skipped gate is a failure).
- `release.lock.yaml`: `release_id`, `created`, `code_sha`, `reconstructed`, `flavour`, `configs`, `sources` (the registry pins; no silent gaps) and, only when non-empty, `excluded` (`{id: reason}`:
  pinned sources the registry keeps out of the release by `release_skip_reason`; the lock, the check and
  the fetch all honour it). `release build --lock` checks the build against it.
- Credentials: only `s3_auth.py` names rclone.conf and boto3 S3 clients exist only in `s3_auth`/`s3_client` (`tests/test_s3_auth_boundary.py`); `auth check` prints no secret.
- Upload layout (`release publish --go`, `HfApi.upload_large_folder`; the repo must already exist, nothing is created): `data/**/*.parquet`, `README.md`, `croissant.json`, `provenance/{RELEASE.json,GATES.json,release.lock.yaml}`, nothing else. GATES.json and the lock go up as they are; RELEASE.json goes up as a public-safe projection (`marinedata.release_provenance`: an allowlist of counts, config names, ids and digests plus the lock's sources; no excluded or skipped sources, buckets, remotes or local paths). The frozen v1.0 needs `--allow-frozen`, also for `release build`.

## Open items
Maintainer follow-ups tracked outside this repository. None of them changes v1.0.
- Hub metadata: re-tag the CoralSCOP model repository (CC BY-NC-SA 4.0, with attribution), add a model card for coral-ai, enable gated access on the non-commercial dataset repository, add the v1.0 tag.
- Registry: 29 sources were downgraded to `access_class: unknown` pending a licence re-check against their upstream pages; five licence ids changed by the audit await confirmation; two reconciliation pairs remain undecided and are kept as `related`; the caddy stub tree and wikimedia-underwater (per-file licences) stay internal-only.
- Seaview image licence: confirm CC-BY-3.0 against CC-BY-3.0-AU on the UQ eSpace record (doi:10.14264/UQL.2019.930).
- Storage: 13 source trees have no CHECKSUMS file yet, so `root_digest` pins 76 trees; access classes for the non-commercial pairs (D20), the platform value for own data (D21) and the bucket access-policy review (D9) are pending.
- Release: build v1.1 (points and metadata configs) on the release VM under `deploy/release-vm/`, run the gates and publish; the x86_64 image has not been exercised yet. Any widening of the v1.1 points config stays additive.
- Repository: issue labels, the takedown wording in SECURITY.md, branch clean-up after merge.

## Evidence pointing outside the repo
24 `licence_evidence:` lines cite the S57 catalog (`reports/data/2026-09-25-5star-catalog.tsv`), which is not in the repo; they are left as written:
fathomnet-coverage, soneva-corals, benthoz15, fathomnet-megalodon, noaa-dscrtp, noaa-goa-2019, pangaea-ccz-gsr, pangaea-ofos-msm77, bodc-smartex, oceancv-rovtransect, deepsea-coral-cornerrise, ntnu-arl-uw, subpipe, uw-img-imu-sonar, galapagos-3d, port-royal, aqualoc, seagrassfinder, posidonia, openimages-marine, smartbay, caddy, vdd-c, sodd.
