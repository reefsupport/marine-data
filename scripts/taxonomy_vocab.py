"""WP-7 step 1: turn fetched label inventories into registry/taxonomy/vocab/*.tsv.

Input: the per-source label caches written by the WP-7 fetchers (label lists and
annotation counts only; no images were downloaded). Output: one TSV per labelled
source — the observed vocabulary the release gate audits a crosswalk against.
Usage: python scripts/taxonomy_vocab.py <cache_dir> <repo_root>
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

DAY = "2026-09-25"


def write(
    repo: Path,
    source: str,
    crosswalk: str,
    counts: dict,
    *,
    kind: str,
    origin: str,
    desc: dict | None = None,
    note: str = "",
) -> None:
    out = repo / "registry/taxonomy/vocab" / f"{source}.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# source: {source}",
        f"# crosswalk: {crosswalk}",
        f"# counts: {kind}",
        f"# retrieved: {DAY} from {origin}",
    ]
    if note:
        lines.append(f"# note: {note}")
    lines.append("label\tcount\tdescription")
    for label, n in sorted(counts.items(), key=lambda t: (-(t[1] or 0), t[0])):
        lines.append(f"{label}\t{'' if n is None else n}\t{(desc or {}).get(label, '')}")
    out.write_text("\n".join(lines) + "\n")


def coralnet_names(cache: Path) -> dict[str, tuple[str, str]]:
    page = (cache / "coralnet_label_list.html").read_text()
    out = {}
    for m in re.finditer(
        r'<tr data-label-id="(\d+)">\s*<td '
        r'class="name"><a[^>]*>([^<]*)</a></td>\s*<td>([^<]*)</td>',
        page,
    ):
        out[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return out


def main(cache: Path, repo: Path) -> None:
    j = lambda n: json.loads((cache / n).read_text())  # noqa: E731
    hf = "https://huggingface.co/datasets/"
    fgvc = [r["name"] for r in csv.DictReader((cache / "fgvc_supercategory_map.csv").open())]
    write(
        repo,
        "fathomnet-fgvc23",
        "fathomnet-concepts",
        dict.fromkeys(fgvc),
        kind="types",
        origin="github.com/fathomnet/fgvc-comp-2023 docs/supercategory_map.csv",
        note="annotation JSONs are Kaggle-only (login) per D-E; coverage is type-weighted",
    )
    write(
        repo,
        "fathomnet-vme",
        "fathomnet-concepts",
        j("fathomnet-vme.labels.json")["counts"],
        kind="annotations",
        origin=hf + "FathomNet/2024-vme-benchmark coco_{train,val,test}.json",
    )
    write(
        repo,
        "ruod",
        "ruod",
        j("ruod.labels.json")["counts"],
        kind="annotations",
        origin=hf + "Mortallll/RUOD coco/annotations/instances_{train,val}.json",
    )
    for sid, repo_id in [("uiis", "UIIS"), ("usis10k", "USIS10K"), ("uiis10k", "UIIS10K")]:
        counts = dict(j(f"{sid}.labels.json")["counts"])
        fg = counts.pop("foreground", None)
        write(
            repo,
            sid,
            "usis-uiis",
            counts,
            kind="annotations",
            origin=hf + f"LiamLian0727/{repo_id}",
            note=(
                f"excluded foreground_annotations/* ({fg} class-agnostic masks duplicating the "
                "multi-class set)"
            )
            if fg
            else "",
        )
    for v, sid in [("instance_version", "trashcan"), ("material_version", "trashcan-material")]:
        write(
            repo,
            sid,
            "trashcan",
            j(f"trashcan-{v}.labels.json")["counts"],
            kind="annotations",
            origin=f"DRUM hdl:11299/214865 dataset.zip dataset/{v}/instances_*_trashcan.json",
        )
    write(
        repo,
        "trash-icra19",
        "trash-icra19",
        dict.fromkeys(["plastic", "bio", "rov"]),
        kind="types",
        origin="DRUM hdl:11299/214366 (class list per Fulton et al. 2019, ICRA)",
        note="per-class counts need the YOLO txt members; coverage is type-weighted",
    )
    yml = (cache / "obsea_23sp_4120img_34945annots_2688res_data.yaml").read_text()
    names = json.loads(re.search(r"names:\s*(\[.*\])", yml).group(1).replace("'", '"'))
    write(
        repo,
        "obsea-fish",
        "obsea-fish",
        dict.fromkeys(names),
        kind="types",
        origin="Zenodo 14888440 obsea_split_YOLO.zip data.yaml (23 classes, 34,945 boxes)",
        note="the 4,120 label files span 1.2 GB of the zip; coverage is type-weighted",
    )
    d = j("deepseagrass_data.json")
    c4, c5 = Counter(), Counter()
    for f in d.get("file", d.get("files", [])):
        p = f.get("filename", "").split("/")
        if p[-1].lower().endswith(".jpg"):
            (c5 if p[0].startswith("For_5") else c4)[p[-2]] += 1
    write(
        repo,
        "deepseagrass",
        "deepseagrass",
        {**c4, **c5},
        kind="annotations",
        origin="CSIRO DAP csiro:47653 file listing (patch counts by class folder)",
    )
    import pyarrow.parquet as pq

    t1 = pq.read_table(cache / "noaa_t1.parquet", columns=["label_name", "label_desc"]).to_pylist()
    write(
        repo,
        "noaa-benthic-t1",
        "coralnet-noaa-pifsc",
        Counter(r["label_name"] for r in t1),
        kind="annotations",
        origin=hf + "NMFS-OSI/noaa-pacific-benthic-cover-t1-all data/all.parquet",
        desc={r["label_name"]: r["label_desc"] for r in t1},
    )
    t3 = Counter(
        r["label_name"]
        for r in pq.read_table(cache / "noaa_t3.parquet", columns=["label_name"]).to_pylist()
    )
    cn = coralnet_names(cache)
    ids = {
        r["Short Code"]: r["Label ID"]
        for r in csv.DictReader((cache / "noaa_t3_class_list_full.csv").open())
    }
    write(
        repo,
        "noaa-benthic-t3",
        "coralnet-noaa-pifsc",
        t3,
        kind="annotations",
        origin=hf + "NMFS-OSI/noaa-pacific-benthic-cover-t3-all data/all.parquet + "
        "coralnet.ucsd.edu/label/list/",
        desc={
            k: " | ".join(cn.get(ids.get(k, ""), ("?", "?"))) + f" | coralnet:{ids.get(k, '?')}"
            for k in t3
        },
    )
    plc: Counter = Counter()
    for site in ["heron_reef", "nanwan_bay", "line_islands"]:
        for k, n in j(f"plc-{site}.labels.json")["counts"].items():
            plc[k.split(",", 1)[1].strip()] += n
    write(
        repo,
        "plc-beijbom2015",
        "plc-beijbom2015",
        plc,
        kind="annotations",
        origin="Zenodo 5000003 {heron_reef,nanwan_bay,line_islands} labelmap.txt + "
        "*/annotations.txt (archived column)",
    )
    mlc = ["CCA", "Turf", "Macro", "Sand", "Acrop", "Pavon", "Monti", "Pocill", "Porit"]
    write(
        repo,
        "mlc-moorea",
        "mlc-moorea",
        dict.fromkeys(mlc),
        kind="types",
        origin="Beijbom et al. 2012 (CVPR) label set; EDI knb-lter-mcr.5006.3 metadata is "
        "login-walled (403)",
        note="coverage is type-weighted until the annotation files are staged",
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
