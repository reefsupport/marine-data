#!/usr/bin/env python3
"""W0 slice driver: write one ingest spec per registered-but-not-staged source, dry-run it
through ``marinedata ingest-source`` (metadata/enumeration only, no bytes), and append a
row to ``registry/ingest-specs/_queue-w0.tsv``.

Network-light: HF/GitHub revision resolution and Zenodo/HF/bucket listing calls only.
Never downloads payload bytes. See docs/ingest-howto.md and the 2026-09-25 5star-SPEC-w0
brief for the source list and per-source disposition (run / needs_yohan / needs_adapter).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "registry" / "ingest-specs"
QUEUE = SPEC_DIR / "_queue-w0.tsv"
PY = Path.home() / "dev/.wt/marine-data/splitmap-group/.venv/bin/python3.11"
WORK = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/spec-w0-work")

# kind: run | needs_yohan | needs_adapter | dead
SOURCES = [
    dict(id="coralnet-public", kind="needs_yohan", adapter="http", params={},
         license="NOASSERTION", attribution="CoralNet (Chen et al. 2021, CoralNet 1.0)",
         citation="Chen+ 2021 CoralNet 1.0", homepage="https://coralnet.ucsd.edu/source/about/",
         size_gb=None, value=5, why="account + CoralNet ToS required for bulk/per-source export"),
    dict(id="seatizen-atlas", kind="run", adapter="zenodo",
         params={"zenodo_record": 12819157},
         license="CC-BY-4.0", attribution="Seatizen Atlas (Contini et al., Sci Data 2025)",
         citation="doi:10.5281/zenodo.12819157", homepage="https://zenodo.org/records/12819157",
         size_gb=35.58, value=4),
    dict(id="sweet-corals", kind="run", adapter="hf",
         params={"repo": "wildflow/sweet-corals", "revision": "main"},
         license="CC-BY-4.0", attribution="wildflow sweet-corals",
         citation="wildflow", homepage="https://huggingface.co/datasets/wildflow/sweet-corals",
         size_gb=339.8, value=3),
    dict(id="reefnet", kind="dead", adapter="hf", params={},
         license="CC-BY-NC-SA-4.0", attribution="ReefNet 2025 (arXiv:2510.16822)",
         citation="ReefNet 2025 arXiv:2510.16822", homepage="https://arxiv.org/abs/2510.16822",
         size_gb=None, value=4,
         why="no resolvable download_urls; duplicate registry stub of reefnet-hf (same arXiv id)"),
    dict(id="reefnet-hf", kind="run", adapter="hf",
         params={"repo": "ReefNet/ReefNet-1.0", "revision": "main"},
         license="CC-BY-NC-SA-4.0", attribution="ReefNet 2025 (arXiv:2510.16822)",
         citation="ReefNet 2025 arXiv:2510.16822", homepage="https://huggingface.co/datasets/ReefNet/ReefNet-1.0",
         size_gb=13.58, value=4),
    dict(id="reefnet-species", kind="run", adapter="hf",
         params={"repo": "TsinghuaCorals/reefnet_species_images", "revision": "main"},
         license="NOASSERTION", attribution="TsinghuaCorals reefnet_species_images (HF)",
         citation="-", homepage="https://huggingface.co/datasets/TsinghuaCorals/reefnet_species_images",
         size_gb=277.43, value=3),
    dict(id="deolhonoscorais", kind="run", adapter="zenodo",
         params={"zenodo_record": 7338208},
         license="CC-BY-4.0", attribution="De Olhos nos Corais (Zenodo 7338208)",
         citation="Zenodo 7338208", homepage="https://zenodo.org/records/7338208",
         size_gb=12.89, value=3),
    dict(id="mermaid", kind="run", adapter="bucket",
         params={"endpoint": "s3.amazonaws.com", "bucket": "coral-reef-training", "prefix": ""},
         license="NOASSERTION", attribution="MERMAID / Wildlife Conservation Society (WCS) "
             "coral-reef-training",
         citation="MERMAID / WCS", homepage="https://datamermaid.org/",
         size_gb=68.5, value=5),
    dict(id="marineinst20m", kind="needs_yohan", adapter="http", params={},
         license="CC-BY-NC", attribution="MarineInst20M (Zheng et al., ECCV 2024)",
         citation="Zheng+ ECCV2024 MarineInst", homepage="https://github.com/zhengziqiang/MarineInst20M",
         size_gb=None, value=5, why="README has no data link, no HF dataset found; email the "
             "authors"),
    dict(id="fathomnet", kind="needs_adapter", adapter="http", params={},
         license="NOASSERTION", attribution="FathomNet (Katija et al., Sci Rep 2022)",
         citation="Katija+ Sci Rep 2022", homepage="https://database.fathomnet.org/",
         size_gb=700, value=5,
         why="needs_adapter:fathomnet-api - paginated REST at "
             "https://database.fathomnet.org/api/images/query/filter "
             "returns per-image JSON (uuid, url, boundingBoxes); per-image licence varies "
                 "CC-BY/CC-BY-NC. "
             "W3-scale (Phase A subset) is out of scope for this slice."),
    dict(id="deepsea-mot", kind="run", adapter="hf",
         params={"repo": "MBARI-org/DeepSea-MOT", "revision": "main"},
         license="CC-BY-SA-4.0", attribution="MBARI DeepSea-MOT",
         citation="MBARI 2025", homepage="https://huggingface.co/datasets/MBARI-org/DeepSea-MOT",
         size_gb=20.02, value=4),
    dict(id="benthicnet", kind="needs_adapter", adapter="http", params={},
         license="CC-BY", attribution="BenthicNet (Lowe et al., Sci Data 2025)",
         citation="doi:10.20383/103.0614", homepage="https://doi.org/10.20383/103.0614",
         size_gb=None, value=5,
         why="needs_adapter:frdr-globus - FRDR record exposes a Globus collection + a per-file "
             "HTTPS "
             "listing at https://www.frdr-dfdr.ca/repo/handle/... ; no plain enumerable file "
                 "list yet. "
             "W3-scale (CATAMI-labelled subset) is out of scope for this slice."),
    dict(id="deepfish", kind="run", adapter="http",
         params={"version": "2020", "urls": [
             {"url": "http://data.qld.edu.au/public/Q5842/2020-AlzayatSaleh-00e364223a600e83bd9c3f5bcd91045-DeepFish/DeepFish.tar",
              "key": "DeepFish.tar"}]},
         license="CC-BY-4.0", attribution="DeepFish (Saleh et al., Sci Rep 2020)",
         citation="Saleh+ Sci Rep 2020", homepage="https://alzayats.github.io/DeepFish/",
         size_gb=7.6, value=4),
    dict(id="ozfish", kind="needs_adapter", adapter="bucket",
         params={"endpoint": "storage.pawsey.org.au", "bucket": "public",
                 "prefix": "m/FDFML/frames"},
         license="CC-BY-3.0-AU", attribution="OzFish (AIMS)",
         citation="doi:10.25845/5e28f062c5097", homepage="https://storage.pawsey.org.au/public/m/FDFML/",
         size_gb=None, value=5,
         why="needs_adapter:pawsey-portal - storage.pawsey.org.au/public/... is the Pawsey Data "
             "Portal "
             "front-end (200 on a directory path, 403 with a listing query string), not a raw "
                 "anonymous "
             "S3 ListObjectsV2 endpoint; the bucket adapter's path-style listing 404s. "
                 "frames/crops/labelled "
             "are the three known top-level prefixes; per-file URLs need a portal-specific "
                 "crawler."),
    dict(id="fishnet-2023", kind="needs_yohan", adapter="http", params={},
         license="NOASSERTION", attribution="FishNet (Khan et al., ICCV 2023)",
         citation="Khan+ ICCV2023", homepage="https://fishnet-2023.github.io/",
         size_gb=None, value=3, why="project page only, no anonymous download; request access"),
    dict(id="fish4knowledge", kind="needs_adapter", adapter="http", params={},
         license="NOASSERTION", attribution="Fish4Knowledge (Fisher et al. 2016)",
         citation="Fisher+ 2016 Springer", homepage="http://groups.inf.ed.ac.uk/f4k/",
         size_gb=0.51, value=3,
         why="needs_adapter:site-navigation - homepage links to an HTML index of per-video "
             "archives (no stable file list resolved from here)"),
    dict(id="brackish", kind="run", adapter="hf",
         params={"repo": "dronefreak/Brackish", "revision": "main"},
         license="CC-BY-4.0", attribution="Brackish Dataset (Pedersen et al., CVPRW 2019), via "
             "HF mirror dronefreak/Brackish",
         citation="Pedersen+ CVPRW2019", homepage="https://huggingface.co/datasets/dronefreak/Brackish",
         size_gb=0.71, value=4,
         why="original AAU host has no resolved download URL from here; HF mirror used, note "
             "for WP-6d/Yohan"),
    dict(id="lfitw", kind="needs_adapter", adapter="http", params={},
         license="NOASSERTION", attribution="Labeled Fishes in the Wild (Cutter et al., WACVW "
             "2015)",
         citation="Cutter+ WACVW2015", homepage="https://swfsc-fisheries.github.io/labeled-fishes-in-the-wild/",
         size_gb=0.44, value=3,
         why="needs_adapter:site-navigation - NOAA SWFSC page has no stable direct file list "
             "resolved from here"),
    dict(id="mouss", kind="run", adapter="hf",
         params={"repo": "akridge/MOUSS_fish_imagery_dataset_grayscale_small", "revision": "main"},
         license="NOASSERTION", attribution="NOAA PIFSC MOUSS fish imagery (grayscale small), "
             "via HF akridge/MOUSS_fish_imagery_dataset_grayscale_small",
         citation="NOAA PIFSC MOUSS", homepage="https://huggingface.co/datasets/akridge/MOUSS_fish_imagery_dataset_grayscale_small",
         size_gb=0.09, value=3),
    dict(id="seamapd21", kind="needs_adapter", adapter="http",
         params={"version": "2021", "urls": [
             {"url": f"https://grunt.sefsc.noaa.gov/parr/SEAMAPD21.tar.gz.a{c}"}
             for c in "abcdef"]},
         license="NOASSERTION", attribution="SEAMAPD21 (Boulais et al. 2021), NOAA SEFSC",
         citation="Boulais+ 2021", homepage="https://github.com/SEFSC/SEAMAPD21",
         size_gb=None, value=4,
         why="needs_adapter:multipart-tar - 6 split parts (SEAMAPD21.tar.gz.aa..af) on "
             "grunt.sefsc.noaa.gov; "
             "the http adapter's enumerate() only keeps known streamable/spooled suffixes "
                 "(.tar/.tar.gz/.zip/...), "
             "so '.tar.gz.aa' is silently dropped (0 items) - needs a part-concatenating "
                 "adapter, not a filter tweak"),
    dict(id="whoi-plankton", kind="run", adapter="hf",
         params={"repo": "nf-whoi/whoi-plankton", "revision": "main"}, expected_images=956867,
         license="NOASSERTION", attribution="WHOI-Plankton (Sosik et al.), via HF "
             "nf-whoi/whoi-plankton",
         citation="Sosik+ WHOI-Plankton", homepage="https://huggingface.co/datasets/nf-whoi/whoi-plankton",
         size_gb=27.33, value=2),
    dict(id="seaclear", kind="needs_adapter", adapter="http", params={},
         license="CC-BY-4.0", attribution="SeaClear marine debris (Djuraskovic et al., Sci Data "
             "2024)",
         citation="Djuraskovic+ Sci Data 2024", homepage="https://doi.org/10.1038/s41597-024-03286-0",
         size_gb=1.71, value=4,
         why="needs_adapter:figshare-api - Sci Data supplementary data is on figshare, not Zenodo; "
             "no zenodo/http adapter fits figshare's own REST API"),
    dict(id="suim", kind="run", adapter="hf",
         params={"repo": "SatwikKambham/suim", "revision": "main"},
         license="MIT", attribution="SUIM (Islam et al., IROS 2020), via HF mirror "
             "SatwikKambham/suim",
         citation="Islam+ IROS2020", homepage="https://huggingface.co/datasets/SatwikKambham/suim",
         size_gb=0.53, value=4,
         why="original IRVLab host returns 403 from here; HF mirror used (MIT as stated on the "
             "mirror card)"),
    dict(id="uieb", kind="run", adapter="hf",
         params={"repo": "Hikari0608/UIEB", "revision": "main"},
         license="NOASSERTION", attribution="UIEB (Li et al., TIP 2019), via HF mirror "
             "Hikari0608/UIEB",
         citation="Li+ TIP2019", homepage="https://huggingface.co/datasets/Hikari0608/UIEB",
         size_gb=6.01, value=4),
    dict(id="squid", kind="run", adapter="zenodo",
         params={"zenodo_record": 5744037},
         license="CC-BY-NC-SA-4.0", attribution="SQUID (Berman et al., TPAMI 2021)",
         citation="Berman+ TPAMI2021", homepage="https://zenodo.org/records/5744037",
         size_gb=45.82, value=4),
    dict(id="flsea", kind="needs_yohan", adapter="http", params={},
         license="CC-BY-NC-SA-4.0", attribution="FLSea (Randall et al. 2023)",
         citation="Randall+ 2023", homepage="https://www.kaggle.com/datasets/viseaonlab/flsea-vi",
         size_gb=None, value=4, why="Kaggle login required"),
    dict(id="varos", kind="run", adapter="zenodo",
         params={"zenodo_record": 5567209, "version": "record-5567209"},
         license="CC-BY-4.0", attribution="VAROS (Zwilgmeyer et al., ICCVW 2021)",
         citation="Zwilgmeyer+ ICCVW2021", homepage="https://zenodo.org/records/5567209",
         size_gb=17.96, value=2),
    dict(id="atlantis", kind="needs_yohan", adapter="http", params={},
         license="NOASSERTION", attribution="ATLANTIS (Zhang et al., CVPR 2024)",
         citation="Zhang+ CVPR2024", homepage="https://www.kaggle.com/datasets/zkawfanx/atlantis",
         size_gb=2, value=2, why="hosted on Kaggle; login/API key required (D-E: never create "
             "accounts)"),
    dict(id="deepreefmap", kind="run", adapter="zenodo",
         params={"zenodo_record": 10624794},
         license="CC-BY-4.0", attribution="DeepReefMap (Sauder et al. 2024)",
         citation="Sauder+ 2024", homepage="https://zenodo.org/records/10624794",
         size_gb=10.34, value=4),
    dict(id="reef-guidance", kind="run", adapter="hf",
         params={"repo": "QCR-Underwater-Perception/reef-guidance-system", "revision": "main"},
         license="CC-BY-NC-SA-4.0", attribution="Reef Guidance System (QUT QCR)",
         citation="QUT QCR", homepage="https://huggingface.co/datasets/QCR-Underwater-Perception/reef-guidance-system",
         size_gb=30.31, value=3),
    dict(id="reefset-v1", kind="run", adapter="zenodo",
         params={"zenodo_record": 11071202},
         license="CC-BY-4.0", attribution="ReefSet v1",
         citation="Zenodo 11071202", homepage="https://zenodo.org/records/11071202",
         size_gb=1.71, value=3),
    dict(id="floating-marine-debris", kind="run", adapter="github",
         params={"repo": "miguelmendesduarte/Floating-Marine-Debris-Data", "ref": "main"},
         license="CC-BY-4.0", attribution="Floating Marine Debris Data (Duarte et al.)",
         citation="GitHub miguelmendesduarte/Floating-Marine-Debris-Data",
         homepage="https://github.com/miguelmendesduarte/Floating-Marine-Debris-Data",
         size_gb=0.02, value=3),
    dict(id="heron-reef-benthic", kind="needs_adapter", adapter="http", params={},
         license="CC-BY-4.0", attribution="Heron Reef benthic imagery (Sci Data 2021)",
         citation="doi:10.1038/s41597-021-00871-5", homepage="https://www.nature.com/articles/s41597-021-00871-5",
         size_gb=None, value=3,
         why="needs_adapter:paper-lookup - article landing page only, actual data repository "
             "(likely a supplementary Zenodo/figshare record) not yet resolved"),
    dict(id="coralnet", kind="needs_yohan", adapter="http", params={},
         license="NOASSERTION", attribution="CoralNet API",
         citation="-", homepage="https://coralnet.ucsd.edu/",
         size_gb=None, value=3, why="CoralNet's API requires an account/API token even for "
             "source-scoped exports"),
    dict(id="mft25", kind="needs_yohan", adapter="http", params={},
         license="Apache-2.0", attribution="MFT25 / SU-T (Vranlee et al. 2025)",
         citation="-", homepage="https://vranlee.github.io/SU-T/",
         size_gb=None, value=3, why="project page is request-based (no anonymous download link)"),
    dict(id="sea-urchin-detection", kind="needs_yohan", adapter="http", params={},
         license="CC-BY-4.0", attribution="Sea Urchin Detection (Roboflow Universe)",
         citation="-", homepage="https://universe.roboflow.com/thesis-ztpwb/sea-urchin-10r8i",
         size_gb=None, value=3, why="Roboflow export requires a free API key (account, "
             "forbidden by D-E)"),
    dict(id="noaa-ncrmp-structural-complexity", kind="needs_adapter", adapter="http", params={},
         license="NOASSERTION", attribution="NOAA NCRMP structural complexity",
         citation="-", homepage="https://www.fisheries.noaa.gov/inport/item/67851",
         size_gb=None, value=3,
         why="needs_adapter:site-lookup - InPort metadata record only, the underlying data host "
             "is not resolved"),
    dict(id="whoi-plankton-small", kind="run", adapter="hf",
         params={"repo": "nf-whoi/whoi-plankton-small", "revision": "main"},
         license="MIT", attribution="WHOI-Plankton (small), via HF nf-whoi/whoi-plankton-small",
         citation="Sosik+ WHOI-Plankton", homepage="https://huggingface.co/datasets/nf-whoi/whoi-plankton-small",
         size_gb=1.11, value=3),
    dict(id="underwater-image-enhancement", kind="run", adapter="hf",
         params={"repo": "iamvastava/underwater_image_enhancement", "revision": "main"},
         license="CC-BY-4.0", attribution="Underwater Image Enhancement (HF "
             "iamvastava/underwater_image_enhancement)",
         citation="-", homepage="https://huggingface.co/datasets/iamvastava/underwater_image_enhancement",
         size_gb=0.45, value=3),
]


def hf_sha(repo: str, revision: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{repo}/revision/{revision}",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            info = json.load(r)
        if info.get("gated") or info.get("private") or info.get("disabled"):
            return None
        return str(info["sha"])
    except Exception:
        return None


def github_sha(repo: str, ref: str) -> str | None:
    try:
        req = urllib.request.Request(f"https://api.github.com/repos/{repo}/commits/{ref}")
        with urllib.request.urlopen(req, timeout=20) as r:
            info = json.load(r)
        return str(info["sha"])
    except Exception:
        return None


def build_yaml(s: dict) -> dict:
    spec = {
        "id": s["id"],
        "adapter": s["adapter"],
        "params": dict(s["params"]),
        "license": s["license"],
        "attribution": s["attribution"],
        "citation": s.get("citation", ""),
        "homepage": s.get("homepage", ""),
    }
    if s.get("expected_images"):
        spec["expected_images"] = s["expected_images"]
    return spec


def dump_yaml(spec: dict) -> str:
    # Minimal, deterministic YAML writer (no external pyyaml dependency needed at import time
    # of this script; the CLI itself uses pyyaml to read it back).
    import yaml

    return yaml.safe_dump(spec, sort_keys=False, default_flow_style=False)


def run_dry(s: dict, spec_path: Path) -> tuple[str, dict]:
    """Returns (status, report_or_info)."""
    try:
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        proc = subprocess.run(
            [str(PY), "-m", "marinedata.cli", "ingest-source", s["adapter"], str(spec_path),
             "--dry-run", "--work", str(WORK)],
            cwd=str(ROOT), env=env,
            capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return "dead:timeout", {}
    if proc.returncode == 3:
        line = (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else ""
        return f"needs_yohan:{line}", {}
    if proc.returncode != 0:
        return f"dead:{proc.returncode}:{(proc.stderr or '').strip()[-200:]}", {}
    try:
        report = json.loads(proc.stdout)
    except Exception:
        return f"dead:bad-json:{proc.stdout[-200:]}", {}
    return "ok", report


def main() -> None:
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in SOURCES:
        sid = s["id"]
        # Pin revisions/refs before writing the final spec.
        if s["kind"] == "run" and s["adapter"] == "hf" and s["params"].get("revision") == "main":
            sha = hf_sha(s["params"]["repo"], "main")
            if sha:
                s["params"]["revision"] = sha
            else:
                s["kind"] = "needs_adapter"
                s["why"] = "hf revision resolution failed (network/gated) - retry"
        if s["kind"] == "run" and s["adapter"] == "github" and s["params"].get("ref") == "main":
            sha = github_sha(s["params"]["repo"], "main")
            if sha:
                s["params"]["ref"] = sha
            else:
                s["kind"] = "needs_adapter"
                s["why"] = "github ref resolution failed (network) - retry"

        spec = build_yaml(s)
        spec_path = SPEC_DIR / f"{sid}.yaml"
        spec_path.write_text(dump_yaml(spec))

        pinned = "-"
        items = bytes_ = ""
        if s["kind"] == "run":
            status, report = run_dry(s, spec_path)
            if status == "ok":
                pinned = report.get("version", "-")
                items = report.get("plan", {}).get("items", "")
                bytes_ = report.get("plan", {}).get("declared_bytes", "")
            dry_run = status
        else:
            dry_run = f"{s['kind']}:{s.get('why', '')}"
            if s["adapter"] == "hf" and s["params"].get("revision"):
                pinned = s["params"]["revision"]
            elif s["adapter"] == "zenodo" and s["params"].get("zenodo_record"):
                pinned = f"record-{s['params']['zenodo_record']}"

        size_gb = s.get("size_gb")
        gb = (bytes_ / 1e9) if isinstance(bytes_, (int, float)) and bytes_ else size_gb
        est_hours = round(gb / 360, 2) if gb else "?"
        priority = round(s["value"] / gb, 3) if gb else s["value"]

        rows.append([
            sid, s["adapter"], pinned, items, bytes_ or "", s["license"], s.get("format", ""),
            dry_run, est_hours, priority,
        ])
        print(f"{sid}: {dry_run}", file=sys.stderr)

    with open(QUEUE, "w") as f:
        f.write("id\tadapter\tpinned_version\titems\tbytes\tlicence\tformat\tdry_run\test_hours_at_100MBps\tpriority\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
