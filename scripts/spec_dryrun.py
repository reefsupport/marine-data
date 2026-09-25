#!/usr/bin/env python3
"""Dry-run driver for the 5-star ingest-spec queue (see docs/ingest-howto.md).

For each requested source, runs ``marinedata ingest-source <adapter> <spec>
--dry-run`` under a bounded subprocess timeout (network-light: metadata/
enumeration only, no bytes move) and reports one row:

    id, adapter, pinned_version, items, bytes, licence, label_format, dry_run,
    est_hours_at_100mbps, priority

Sources whose spec names an adapter the framework does not implement (only
hf, http, zenodo, bucket, github are real) are never invoked — the CLI would
reject an unknown adapter choice outright — their row is filled straight from
the spec's own ``adapter``/``notes`` fields, no subprocess call. A spec whose
``adapter`` is documentation-only (``needs:<kind>``, e.g. ``needs:pawsey``,
``needs:multipart-tar`` — see each spec's ``why_not_supported``) reports
``needs_adapter:<kind>`` with the ``needs:`` prefix stripped.

D-R4 (2026-09-25 5star-charter): a dry-run that resolves (access ok, no gate)
but enumerates 0 items is a decoder gap, not an access problem —
``BaseAdapter.enumerate()`` only keeps STREAMABLE/SPOOLED suffixes, so a
source whose payload is exclusively some other container (video, rar, a pure
JSON caption file, ...) never gets picked up. Any row that comes back
``dry_run == "ok"`` with ``items == 0`` is downgraded here to
``needs_adapter:<kind>``, not left as a false "ok". ``ZERO_ITEM_KIND`` records
the kind for ids already diagnosed by hand; unknown ids fall back to
``needs_adapter:unknown-empty`` so the gap is still visible in the queue.

Usage:
    python scripts/spec_dryrun.py --slice w0                  # every id in
                                                                # _manifest-<slice>.tsv
    python scripts/spec_dryrun.py --ids fathomnet-coverage oceancv-rovtransect
                                                                # just these ids
    python scripts/spec_dryrun.py --slice w2a --out _queue-w2a.tsv

Reads ``registry/ingest-specs/_manifest-<slice>.tsv`` for label_format/priority
(the two queue columns that don't live in an ``IngestSpec``-loadable yaml) when
run with ``--slice``. ``--ids`` mode has no manifest, so those two columns are
left blank unless the id is present in ``CATALOG_META`` (S57 catalog values
carried over from the w2b slice, used only to compute ``priority`` and to
report the stated format when a dry-run never actually ran).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECDIR = ROOT / "registry" / "ingest-specs"
SPECS = SPECDIR  # alias: SPEC-w3's --glob/--ids-file callers use this name
SUPPORTED_ADAPTERS = {
    "hf",
    "http",
    "zenodo",
    "bucket",
    "github",
    "fathomnet",
    "figshare",
    "pangaea",
    "http-index",
    "gdrive-public",
    "seafile-share",
    "girder",
    "pawsey-portal",
    "frdr-https",
}
TIMEOUT_S = 180
BYTES_PER_HOUR_AT_100MBPS = 100_000_000 * 3600  # planning rate, S57 catalog convention

COLUMNS = [
    "id", "adapter", "pinned_version", "items", "bytes", "licence",
    "label_format", "dry_run", "est_hours_at_100mbps", "priority",
]

# id -> kind, for D-R4 rows already diagnosed by hand (video/rar/json/... payloads
# that a real dry-run resolves but enumerates 0 items for). New callers should add
# to this map as they diagnose more; anything else falls back to "unknown-empty".
ZERO_ITEM_KIND: dict[str, str] = {
    "salmon-cage": "video",  # all files are .mp4/.kva, no decoder for either
    "underwater-images-2542305": "rar",  # payload is .rar archives, no extractor
    "oceaninstruct": "json-captions",  # pure JSON instruction/caption pairs, nothing to stage
    "fathomnet-coverage": "parquet-index-only",  # HF parquet is a coverage index, no image files
    "fathomnet-megalodon": "parquet-index-only",
    "oceancv-rovtransect": "unknown-container",  # HF repo has no STREAMABLE/SPOOLED files
    "deepsea-coral-cornerrise": "video",  # Zenodo record is video, no image extractor
}

# catalog id -> (value 1-5, size_gb-as-stated, label/format) from
# $T/reports/data/2026-09-25-5star-catalog.tsv (S57). Used only in --ids mode when
# a spec id has no manifest row: computes `priority` (value-per-GB) and reports the
# stated format when a dry-run never actually ran (needs_adapter/dead).
CATALOG_META: dict[str, tuple[int | None, float | None, str]] = {
    "fathomnet-coverage": (5, None, "parquet"),
    "fathomnet-megalodon": (3, None, "parquet"),
    "oceancv-rovtransect": (2, None, "unknown"),
    "deepsea-coral-cornerrise": (2, None, "CSV(+imgs?)"),
    "salmon-cage": (2, 5.61, "MP4"),
    "underwater-images-2542305": (2, 33.74, "?"),
    "oceaninstruct": (2, 0.03, "JSON"),
    # carried over from the w2b slice
    "fish-vista": (2, 60.48, "PNG+CSV"),
    "wildfish": (3, 25.83, "JPEG"),
    "fish-length-stereo": (3, 25.18, "images"),
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
    "seathru": (4, 38.44, "RAW+TIFF depth"),
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


def _load_manifest(slice_name: str) -> dict[str, dict[str, str]]:
    path = SPECDIR / f"_manifest-{slice_name}.tsv"
    rows: dict[str, dict[str, str]] = {}
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            rows[row["id"]] = row
    return rows


def _run_dry_run(spec_id: str, adapter: str, spec_path: Path) -> dict[str, str]:
    """Invoke the real CLI dry-run for a supported adapter. Never raises — a
    timeout, non-zero exit or a NEEDS-YOHAN refusal all become a queue status."""
    with tempfile.TemporaryDirectory(prefix=f"specdryrun-{spec_id}-") as tmp:
        cmd = [
            sys.executable,
            "-m",
            "marinedata.cli",
            "ingest-source",
            adapter,
            str(spec_path),
            "--dry-run",
            "--work",
            tmp,
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                capture_output=True,
                text=True,
                timeout=TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return {"dry_run": "dead:timeout", "version": "", "items": "", "bytes": ""}

    if proc.returncode == 3:
        why = proc.stderr.strip().split("\t")[-1] if proc.stderr else "gated"
        return {"dry_run": f"needs_yohan:{why}", "version": "", "items": "", "bytes": ""}
    if proc.returncode == 4:
        # Root cause now fixed in the CLI itself (marinedata.adapters.NoStageableItems):
        # 0 stageable items is a real non-zero exit with a reason, not a hand-diagnosed
        # ZERO_ITEM_KIND lookup. `NEEDS-ADAPTER\t<kind>\t<detail>` on stderr.
        parts = proc.stderr.strip().split("\t")
        kind = parts[1] if len(parts) > 1 else ZERO_ITEM_KIND.get(spec_id, "unknown-empty")
        return {"dry_run": f"needs_adapter:{kind}", "version": "", "items": "0", "bytes": ""}
    if proc.returncode != 0:
        return {
            "dry_run": f"dead:exit{proc.returncode}",
            "version": "",
            "items": "",
            "bytes": "",
        }

    try:
        report = json.loads(proc.stdout)
    except ValueError:
        return {"dry_run": "dead:unparseable", "version": "", "items": "", "bytes": ""}
    # A dry run never uploads, so the plan (not the post-run images/bytes counters,
    # which stay 0) carries the real enumeration numbers.
    plan = report.get("plan") or {}
    items = plan.get("items")
    result = {
        "dry_run": "ok",
        "version": str(report.get("version", "")),
        "items": str(items or 0),
        "bytes": str(plan.get("declared_bytes", "") or ""),
    }
    if not items:
        # D-R4: an "ok" dry-run that enumerates 0 items is a decoder gap, not
        # a real ok — downgrade it here so no false-ok reaches the queue.
        kind = ZERO_ITEM_KIND.get(spec_id, "unknown-empty")
        result["dry_run"] = f"needs_adapter:{kind}"
    return result


def zero_item_kind(sid: str, raw: dict) -> str:
    """D-R4: a dry run that is ok but stages 0 items is ``needs_adapter:<kind>``.

    Checks the hand-diagnosed ``ZERO_ITEM_KIND`` map first (existing ids), then a
    spec's own ``measured.zero_item_kind`` override (new ids, SPEC-w3 convention),
    falling back to ``unsupported-format`` so the gap is still visible.
    """
    if sid in ZERO_ITEM_KIND:
        return ZERO_ITEM_KIND[sid]
    return str((raw.get("measured") or {}).get("zero_item_kind") or "unsupported-format")


def process_one(spec_path: Path, manifest_row: dict[str, str]) -> dict[str, str]:
    doc = yaml.safe_load(spec_path.read_text())
    spec_id = doc["id"]
    adapter = doc["adapter"]
    licence = doc.get("license", "")
    cat_value, cat_size_gb, cat_format = CATALOG_META.get(spec_id, (None, None, ""))
    priority = manifest_row.get("priority") or (str(cat_value) if cat_value else "")
    label_format = manifest_row.get("label_format") or cat_format

    forced_override = manifest_row.get("dry_run_override", "").strip()
    if adapter in SUPPORTED_ADAPTERS and not forced_override:
        result = _run_dry_run(spec_id, adapter, spec_path)
    elif forced_override:
        # Manifest forces a status without invoking the CLI — e.g. a supported
        # adapter whose scope is too broad/unbounded to enumerate live (benthoz15).
        result = {"dry_run": forced_override, "version": "", "items": "", "bytes": ""}
    elif adapter.startswith("needs:"):
        # Documentation-only spec (ozfish, seamapd21, ...): never invoke the CLI,
        # strip the "needs:" marker so the queue reads needs_adapter:<kind>.
        result = {
            "dry_run": f"needs_adapter:{adapter.split(':', 1)[1]}",
            "version": "",
            "items": "",
            "bytes": "",
        }
    else:
        result = {"dry_run": f"needs_adapter:{adapter}", "version": "", "items": "", "bytes": ""}

    bytes_val = result["bytes"]
    est_hours = ""
    if bytes_val.isdigit() and int(bytes_val) > 0:
        est_hours = f"{int(bytes_val) / BYTES_PER_HOUR_AT_100MBPS:.2f}"
    elif cat_size_gb:
        est_hours = f"{cat_size_gb * 1e9 / BYTES_PER_HOUR_AT_100MBPS:.2f}"

    return {
        "id": spec_id,
        "adapter": adapter,
        "pinned_version": result["version"],
        "items": result["items"],
        "bytes": bytes_val,
        "licence": licence,
        "label_format": label_format,
        "dry_run": result["dry_run"],
        "est_hours_at_100mbps": est_hours,
        "priority": priority,
    }


def _write_queue(rows: list[dict[str, str]], queue_path: Path) -> None:
    with queue_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--slice", help="slice name; reads _manifest-<slice>.tsv for every id")
    group.add_argument("--ids", nargs="+", help="explicit spec ids to dry-run")
    group.add_argument(
        "--ids-file", type=Path, help="explicit spec ids, one per line (SPEC-w3 convention)"
    )
    parser.add_argument("--glob", help="unused; kept for SPEC-w3 CLI compatibility")
    parser.add_argument(
        "--out",
        "--queue",
        dest="out",
        help="queue tsv path (default: _queue-<slice>.tsv, or stdout for --ids)",
    )
    args = parser.parse_args(argv[1:])

    if args.slice:
        manifest = _load_manifest(args.slice)
        ids = list(manifest.keys())
        if not ids:
            print(f"no manifest rows for slice {args.slice!r}", file=sys.stderr)
            return 2
    elif args.ids_file:
        manifest = {}
        ids = [ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()]
    else:
        manifest = {}
        ids = args.ids

    rows = []
    for spec_id in ids:
        spec_path = SPECDIR / f"{spec_id}.yaml"
        if not spec_path.exists():
            print(f"missing spec for {spec_id}, skipping", file=sys.stderr)
            continue
        row = process_one(spec_path, manifest.get(spec_id, {}))
        rows.append(row)
        print(f"{spec_id}: {row['dry_run']}")

    default_out = SPECDIR / f"_queue-{args.slice}.tsv" if args.slice else None
    out_path = Path(args.out) if args.out else default_out
    if out_path:
        _write_queue(rows, out_path)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
