# Registry reconcile, 2026-10-08 (P1.2)

INV-REG (2026-10-08) holds no explicit list of 45 flagged ids, so the pairs below are found by the same-record test (shared URL, DOI, HF repo or Zenodo record) and by id similarity. Only shared-record pairs with exactly one match are applied (`superseded_by` on the spec-folded stub). `reefnet-species-images` was tightened open -> restricted-nc (ReefNet is CC-BY-NC-SA-4.0 per `reefnet`). Nothing is deleted. Every other row is ambiguous: pick reading A or B.

| spec-folded | established | evidence | reading A | reading B |
|---|---|---|---|---|
| benthicnet (unknown, PER-SOURCE-VARIES) | benthicnet-1m (open, CC-BY-4.0) | similar id only; ambiguous | one dataset: `benthicnet` superseded_by `benthicnet-1m`, `benthicnet-1m` keeps class and counts | two releases or subsets: keep both, `benthicnet` keeps its staged tree |
| brackish (open, CC-BY-4.0) | brackish-dataset (open, CC-BY-SA-4.0) | similar id only; ambiguous | one dataset: `brackish` superseded_by `brackish-dataset`, `brackish-dataset` keeps class and counts | two releases or subsets: keep both, `brackish` keeps its staged tree |
| coralnet-public (internal-only, PER-SOURCE-VARIES) | coralnet (internal-only, NO-LICENCE-STATED) | similar id only; ambiguous | one dataset: `coralnet-public` superseded_by `coralnet`, `coralnet` keeps class and counts | two releases or subsets: keep both, `coralnet-public` keeps its staged tree |
| coralnet-public (internal-only, PER-SOURCE-VARIES) | coralnet-small (internal-only, unknown) | similar id only; ambiguous | one dataset: `coralnet-public` superseded_by `coralnet-small`, `coralnet-small` keeps class and counts | two releases or subsets: keep both, `coralnet-public` keeps its staged tree |
| deepreefmap (open, CC-BY-4.0) | deepreefmap-data (open, CC-BY-4.0) | same URL; APPLIED: superseded_by | one dataset: `deepreefmap` superseded_by `deepreefmap-data`, `deepreefmap-data` keeps class and counts | two releases or subsets: keep both, `deepreefmap` keeps its staged tree |
| fathomnet-fgvc23 (restricted-nd, PER-SOURCE-VARIES) | fathomnet (restricted-nd, PER-SOURCE-VARIES) | similar id only; ambiguous | one dataset: `fathomnet-fgvc23` superseded_by `fathomnet`, `fathomnet` keeps class and counts | two releases or subsets: keep both, `fathomnet-fgvc23` keeps its staged tree |
| fathomnet-fgvc25 (restricted-nd, PER-SOURCE-VARIES) | fathomnet (restricted-nd, PER-SOURCE-VARIES) | similar id only; ambiguous | one dataset: `fathomnet-fgvc25` superseded_by `fathomnet`, `fathomnet` keeps class and counts | two releases or subsets: keep both, `fathomnet-fgvc25` keeps its staged tree |
| fathomnet-vme (restricted-nd, PER-SOURCE-VARIES) | fathomnet (restricted-nd, PER-SOURCE-VARIES) | similar id only; ambiguous | one dataset: `fathomnet-vme` superseded_by `fathomnet`, `fathomnet` keeps class and counts | two releases or subsets: keep both, `fathomnet-vme` keeps its staged tree |
| marineinst20m-web (internal-only, NO-LICENCE-STATED) | marineinst20m (internal-only, PROVENANCE-DEFECTIVE) | same URL; APPLIED: superseded_by | one dataset: `marineinst20m-web` superseded_by `marineinst20m`, `marineinst20m` keeps class and counts | two releases or subsets: keep both, `marineinst20m-web` keeps its staged tree |
| mermaid (unknown, NO-LICENCE-STATED) | mermaid-aws (restricted-nc, CC-BY-NC-SA-4.0) | similar id only; ambiguous | one dataset: `mermaid` superseded_by `mermaid-aws`, `mermaid-aws` keeps class and counts | two releases or subsets: keep both, `mermaid` keeps its staged tree |
| mouss (internal-only, NO-LICENCE-STATED) | mouss-detection (open, US-GOV-PD) | same hf; APPLIED: superseded_by | one dataset: `mouss` superseded_by `mouss-detection`, `mouss-detection` keeps class and counts | two releases or subsets: keep both, `mouss` keeps its staged tree |
| mouss (internal-only, NO-LICENCE-STATED) | mouss-seg (unknown, NO-LICENCE-STATED) | similar id only; ambiguous | one dataset: `mouss` superseded_by `mouss-seg`, `mouss-seg` keeps class and counts | two releases or subsets: keep both, `mouss` keeps its staged tree |
| noaa-pifsc-esa-coral-icra (internal-only, NO-LICENCE-STATED) | noaa-coral-icra (unknown, other-hf) | same hf; APPLIED: superseded_by | one dataset: `noaa-pifsc-esa-coral-icra` superseded_by `noaa-coral-icra`, `noaa-coral-icra` keeps class and counts | two releases or subsets: keep both, `noaa-pifsc-esa-coral-icra` keeps its staged tree |
| reef-guidance (restricted-nc, CC-BY-NC-SA-4.0) | reef-guidance-system (open, CC-BY-4.0) | same hf; APPLIED: superseded_by | one dataset: `reef-guidance` superseded_by `reef-guidance-system`, `reef-guidance-system` keeps class and counts | two releases or subsets: keep both, `reef-guidance` keeps its staged tree |
| reefnet-species (restricted-nc, NO-LICENCE-STATED) | reefnet (restricted-nc, CC-BY-NC-SA-4.0) | similar id only; ambiguous | one dataset: `reefnet-species` superseded_by `reefnet`, `reefnet` keeps class and counts | two releases or subsets: keep both, `reefnet-species` keeps its staged tree |
| reefnet-species (restricted-nc, NO-LICENCE-STATED) | reefnet-hf (restricted-nc, CC-BY-NC-SA-4.0) | similar id only; ambiguous | one dataset: `reefnet-species` superseded_by `reefnet-hf`, `reefnet-hf` keeps class and counts | two releases or subsets: keep both, `reefnet-species` keeps its staged tree |
| reefnet-species (restricted-nc, NO-LICENCE-STATED) | reefnet-species-images (open, CC-BY-4.0) | similar id only; ambiguous | one dataset: `reefnet-species` superseded_by `reefnet-species-images`, `reefnet-species-images` keeps class and counts | two releases or subsets: keep both, `reefnet-species` keeps its staged tree |
| seaclear (open, CC-BY-4.0) | seaclear-marine-debris (open, CC-BY-4.0) | similar id only; ambiguous | one dataset: `seaclear` superseded_by `seaclear-marine-debris`, `seaclear-marine-debris` keeps class and counts | two releases or subsets: keep both, `seaclear` keeps its staged tree |

## P1.2b decisions on the 13 ambiguous pairs (2026-10-08)

Rule: the same upstream DOI, URL, bucket or tree makes the spec-folded id `superseded_by` the entry that owns the staged tree (or the curated entry);
different data is kept and cross-referenced with `related:` (both ways). Nothing is deleted.

| spec-folded | established | decision | evidence |
|---|---|---|---|
| benthicnet | benthicnet-1m | superseded_by | same record: doi:10.20383/103.0614 is in this entry's citation and in benthicnet-1m's access.uri |
| brackish | brackish-dataset | superseded_by | same dataset: this entry's spec says 'via HF mirror dronefreak/Brackish', same Pedersen CVPRW 2019 paper, 14,674 images in both; the mirror says CC-BY-4.0, the original (AAU) CC-BY-SA-4.0, so use the target |
| mermaid | mermaid-aws | superseded_by | same upstream: the spec reads the AWS open-data bucket s3://coral-reef-training that mermaid-aws stages (mermaid-aws owns the tree) |
| coralnet-public | coralnet | keep both, related | public sources of the platform vs the platform entry (a subset, not a duplicate) |
| coralnet-public | coralnet-small | keep both, related | coralnet-small is a 450-image WHOI re-pack on HF, a different object |
| fathomnet-fgvc23 | fathomnet | keep both, related | competition subset with its own tree (5,950 images) vs the live API archive |
| fathomnet-fgvc25 | fathomnet | keep both, related | competition subset with its own tree (9,306 images) vs the live API archive |
| fathomnet-vme | fathomnet | keep both, related | HF VME benchmark repo vs the live API archive |
| mouss | mouss-seg | keep both, related | different HF repos: grayscale imagery vs segmentation masks |
| reefnet-species | reefnet | keep both, related | species-level image set (TsinghuaCorals HF repo) vs the KAUST/MIT ReefNet benchmark (arXiv 2510.16822) |
| reefnet-species | reefnet-hf | keep both, related | species-level image set (TsinghuaCorals HF repo) vs ReefNet/ReefNet-1.0 on HF |
| reefnet-species | reefnet-species-images | keep both, related (undecided) | same repo name under two HF namespaces (TsinghuaCorals vs BobDerBaum), no shared locator in either record |
| seaclear | seaclear-marine-debris | keep both, related (undecided) | different DOIs (10.1038/s41597-024-03286-0 vs -03759-2) for what both name SeaClear marine debris; the stub has no tree (figshare id unresolved) |
