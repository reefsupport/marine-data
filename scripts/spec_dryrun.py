"""W2b SPEC queue driver (network-light, metadata-only dry runs).

For every ``registry/ingest-specs/*.yaml`` spec in this slice, either:

* runs ``marinedata ingest-source <adapter> <spec> --dry-run`` (bounded timeout,
  no bytes move) for specs whose ``adapter`` is one the CLI understands
  (hf/zenodo/http/bucket/github), or
* records the row directly, without a subprocess, for specs whose ``adapter``
  starts with ``needs:`` — the framework has no adapter for those yet
  (Kitware Girder, Google Drive, a multi-part tar, ...); see each spec's
  ``why_not_supported``.

Appends one row per source to ``registry/ingest-specs/_queue-w2b.tsv``.

Usage: ``PYTHONPATH=src .venv/bin/python scripts/spec_dryrun.py``
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SPECS = ROOT / "registry" / "ingest-specs"
QUEUE = SPECS / "_queue-w2b.tsv"  # default; --queue overrides (w3 writes _queue-w3.tsv)
WORK = Path("/tmp/spec-w2b-work")
TIMEOUT_S = 90

REAL_ADAPTERS = {"hf", "http", "zenodo", "bucket", "github"}

# A dry-run that resolves (access ok, no gate) but enumerates 0 items is a
# decoder gap, not an access problem: `BaseAdapter.enumerate()` only keeps
# STREAMABLE/SPOOLED suffixes (images, tar/tgz, zip, parquet — see
# docs/design/ingestion.md), so a source whose payload is exclusively some
# other container never gets picked up. Downgrade those to needs_adapter with
# the concrete kind, and use the catalog's stated size as declared_bytes since
# the (correctly) filtered enumeration reports 0.
ZERO_ITEM_OVERRIDE = {
    "salmon-cage": "video",  # all files are .mp4/.kva, no decoder for either
    "underwater-images-2542305": "rar",  # payload is .rar archives, no extractor
    "oceaninstruct": "json-captions",  # pure JSON instruction/caption pairs, nothing to stage
}
COLUMNS = [
    "id",
    "adapter",
    "version",
    "items",
    "bytes",
    "licence",
    "format",
    "dry_run",
    "est_hours_at_100mbps",
    "priority",
    "measured_items",
    "measured_bytes",
]

# catalog id -> (value 1-5, size_gb-as-stated, label/format) from
# $T/reports/data/2026-09-25-5star-catalog.tsv (S57). Used only to compute
# `priority` (the catalog's value-per-GB) and to report the stated format
# when a dry-run never ran (needs_adapter/dead).
CATALOG_META = {
    "fish-vista": (2, 60.48, "PNG+CSV"),
    "wildfish": (3, 25.83, "JPEG"),
    "fish-length-stereo": (3, 25.18, "images"),
    "salmon-cage": (2, 5.61, "MP4"),
    "luderick-seagrass": (3, 1.17, "images+json"),
    "med-fish": (3, 0.64, "JPEG+labels"),
    "mouss-seg": (3, 0.09, "YOLO-seg"),
    "urpc": (2, 1.83, "VOC/YOLO"),
    "duo": (3, None, "COCO"),
    "viame-public": (4, None, "mixed"),
    "ozfish": (5, None, "CSV+JPEG+MP4"),
    "seamapd21": (4, None, "YOLO"),
    "marinelife16k": (3, 79.62, "parquet"),
    "ocean-r1": (4, 38.48, "parquet"),
    "marine-animals-mm": (2, 3.73, "parquet"),
    "oceaninstruct": (2, 0.03, "JSON"),
    "seathru": (4, 38.44, "RAW+TIFF depth"),
    "underwater-images-2542305": (2, 33.74, "?"),
    "marine-snow": (2, 14.22, "PNG"),
    "rcaustic": (2, 10.57, "video/imgs"),
    "sea-undistort": (1, 2.27, "PNG"),
    "euvp": (4, 0.19, "JPEG pairs"),
    "lsui": (4, None, "PNG pairs"),
    "u45": (1, 0.1, "PNG"),
    "ruie": (2, None, "JPEG"),
    "usr248": (2, None, "PNG"),
    "ufo120": (2, None, "PNG pairs"),
    "hicrd": (3, None, "PNG"),
    "usod10k": (3, None, "PNG"),
}


def run_dry(sid: str, adapter: str, spec_path: Path) -> dict:
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "marinedata.cli",
        "ingest-source",
        adapter,
        str(spec_path),
        "--dry-run",
        "--work",
        str(WORK),
    ]
    import os

    env = {"PYTHONPATH": str(ROOT / "src"), **os.environ}
    try:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return {"dry_run": "dead:timeout", "version": "-", "items": 0, "bytes": 0}
    if proc.returncode == 3:
        m = re.search(r"NEEDS-YOHAN\t(\S+)\t(.+)", proc.stderr)
        why = m.group(2).strip()[:60] if m else "gated"
        return {"dry_run": f"needs_yohan:{why}", "version": "-", "items": 0, "bytes": 0}
    if proc.returncode != 0:
        status = re.search(r"HTTP Error (\d{3})", proc.stderr)
        code = status.group(1) if status else "error"
        return {
            "dry_run": f"dead:{code}",
            "version": "-",
            "items": 0,
            "bytes": 0,
            "_stderr": proc.stderr[-400:],
        }
    report = json.loads(proc.stdout)
    plan = report.get("plan", {})
    return {
        "dry_run": "ok",
        "version": report.get("version", "-"),
        "items": plan.get("items", 0),
        "bytes": plan.get("declared_bytes", 0),
    }


def zero_item_kind(sid: str, raw: dict) -> str:
    """D-R4: a dry run that is ok but stages 0 items is ``needs_adapter:<kind>``."""
    if sid in ZERO_ITEM_OVERRIDE:
        return ZERO_ITEM_OVERRIDE[sid]
    return str((raw.get("measured") or {}).get("zero_item_kind") or "unsupported-format")


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--glob", default="*.yaml", help="spec filename glob under registry/ingest-specs"
    )
    ap.add_argument("--queue", type=Path, default=QUEUE, help="output queue TSV")
    ap.add_argument(
        "--ids-file", type=Path, help="only run the spec ids listed (one per line) in this file"
    )
    args = ap.parse_args(argv)
    paths = sorted(SPECS.glob(args.glob))
    if args.ids_file:
        wanted = {ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()}
        paths = [p for p in paths if p.stem in wanted]
    rows = []
    for spec_path in paths:
        sid = spec_path.stem
        raw = yaml.safe_load(spec_path.read_text()) or {}
        adapter = raw.get("adapter", "")
        licence = raw.get("license", "NOASSERTION")
        cat_value, cat_size_gb, cat_format = CATALOG_META.get(sid, (None, None, "?"))

        if adapter in REAL_ADAPTERS:
            print(f"[{sid}] dry-run ({adapter})...", file=sys.stderr)
            result = run_dry(sid, adapter, spec_path)
            if result["dry_run"] == "ok" and result["items"] == 0:
                result = {**result, "dry_run": f"needs_adapter:{zero_item_kind(sid, raw)}"}
        elif adapter.startswith("needs:"):
            kind = adapter.split(":", 1)[1]
            result = {"dry_run": f"needs_adapter:{kind}", "version": "-", "items": 0, "bytes": 0}
        else:
            result = {
                "dry_run": f"dead:unknown-adapter:{adapter}",
                "version": "-",
                "items": 0,
                "bytes": 0,
            }

        measured = raw.get("measured") or {}
        n_bytes = result.get("bytes", 0) or 0
        est_hours = f"{n_bytes / (100 * 1024 * 1024) / 3600:.3f}" if n_bytes else "-"
        size_gb = (n_bytes / 1e9) if n_bytes else cat_size_gb
        priority = f"{cat_value / size_gb:.3f}" if (cat_value and size_gb) else "?"

        rows.append(
            {
                "id": sid,
                "adapter": adapter,
                "version": result["version"],
                "items": result["items"],
                "bytes": n_bytes,
                "licence": licence,
                "format": cat_format,
                "dry_run": result["dry_run"],
                "est_hours_at_100mbps": est_hours,
                "priority": priority,
                "measured_items": measured.get("items", "-"),
                "measured_bytes": measured.get("bytes", "-"),
            }
        )

    with args.queue.open("w") as f:
        f.write("\t".join(COLUMNS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLUMNS) + "\n")
    print(f"wrote {len(rows)} rows to {args.queue}", file=sys.stderr)


if __name__ == "__main__":
    main()
