# Licence classes restored from the LIC audit (P1.2b, 2026-10-08)

P1.2 tightened ten specs with an unstated licence to internal-only and Part 0 (0a3e4a3) downgraded three to unknown. The LIC-A/LIC-B audit
(2026-10-05; `lic-A.tsv`, `lic-B.tsv`) had read an upstream licence for all 13; each class below is set with a licence id and a `licence_evidence:` line.
`wikimedia-underwater` is the exception: the audit found a per-file regime (CC0, PD, CC-BY, CC-BY-SA per https://commons.wikimedia.org/wiki/Commons:Licensing),
which has no single licence id, and the research gate refuses `open` with PER-SOURCE-VARIES (legal_basis unknown). It stays internal-only (P1.2); the audit sentence is in its notes.

| id | class before | class after | licence | evidence (URL : quoted sentence) |
|---|---|---|---|---|
| hacon-aurora-vent-ofos | internal-only | open | CC-BY-4.0 | https://doi.pangaea.de/10.1594/PANGAEA.943364 : license: https://creativecommons.org/licenses/by/4.0/ (PANGAEA JSON-LD) |
| marineevt | internal-only | open | MIT | https://huggingface.co/datasets/AnTo2209/MarineEVT/resolve/main/README.md : This dataset is released under the MIT License |
| mouss | internal-only | open | US-GOV-PD | https://huggingface.co/datasets/akridge/MOUSS_fish_imagery_dataset_grayscale_small : License: Public Domain (NOAA Open Data) |
| noaa-pifsc-esa-coral-icra | internal-only | open | US-GOV-PD | https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-ESA-CORAL-ICRA-Dataset/resolve/main/README.md : License: Public Domain (NOAA Open Data Public Access) |
| pangaea-ps101-ofos-arctic | internal-only | open | CC-BY-3.0 | https://doi.pangaea.de/10.1594/PANGAEA.871550 : license: https://creativecommons.org/licenses/by/3.0/ (PANGAEA JSON-LD); one sampled child dataset (873046) exposed no licence field |
| pangaea-ps118-ofobs-peninsula | internal-only | open | CC-BY-4.0 | https://doi.pangaea.de/10.1594/PANGAEA.911904 : license: https://creativecommons.org/licenses/by/4.0/ (PANGAEA JSON-LD) |
| pangaea-ps81-ofos-antarctic | internal-only | open | CC-BY-3.0 | https://doi.pangaea.de/10.1594/PANGAEA.872719 : license: https://creativecommons.org/licenses/by/3.0/ (PANGAEA JSON-LD) |
| pangaea-ps96-ofos-weddell | internal-only | open | CC-BY-3.0 | https://doi.pangaea.de/10.1594/PANGAEA.862097 : license: https://creativecommons.org/licenses/by/3.0/ (PANGAEA JSON-LD) |
| so268-ofos-analysis-ready | internal-only | open | CC-BY-4.0 | https://doi.pangaea.de/10.1594/PANGAEA.957274 : license: https://creativecommons.org/licenses/by/4.0/ (PANGAEA JSON-LD) |
| noaa-coral-icra | unknown | open | US-GOV-PD | https://huggingface.co/datasets/NMFS-OSI/NOAA-PIFSC-ESD-ESA-CORAL-ICRA-Dataset : License: Public Domain (NOAA Open Data Public Access) |
| plasticinwater | unknown | open | MIT | https://huggingface.co/datasets/OceanCV/PlasticInWater : license: MIT (card claim only; upstream image provenance unstated, Roboflow-style export) |
| mouss-seg | unknown | open | US-GOV-PD | https://huggingface.co/datasets/akridge/MOUSS_fish_segment_dataset_grayscale : License: Public Domain (NOAA Open Data) |
