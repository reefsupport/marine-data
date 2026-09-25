#!/usr/bin/env python3
"""Dry-run driver for a 5-star ingest-spec queue (see docs/ingest-howto.md).

For every ``registry/ingest-specs/<id>.yaml`` named in this slice's manifest, runs
``marinedata ingest-source <adapter> <spec> --dry-run`` under a bounded subprocess
timeout (network-light: metadata/enumeration only, no bytes move) and appends one
row to ``registry/ingest-specs/_queue-<slice>.tsv``:

    id, adapter, pinned_version, items, bytes, licence, label_format, dry_run,
    est_hours_at_100mbps, priority

Sources whose spec names an adapter the framework does not implement (only hf,
http, zenodo, bucket, github are real) are never invoked — the CLI would reject an
unknown adapter choice outright — their row is filled straight from the spec's own
``adapter``/``notes`` fields, no subprocess call.

Usage: ``python scripts/spec_dryrun.py <slice>`` (reads
``registry/ingest-specs/_manifest-<slice>.tsv`` for label_format/priority, the two
queue columns that don't live in an ``IngestSpec``-loadable yaml).
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECDIR = ROOT / "registry" / "ingest-specs"
SUPPORTED_ADAPTERS = {"hf", "http", "zenodo", "bucket", "github"}
TIMEOUT_S = 180
BYTES_PER_HOUR_AT_100MBPS = 100_000_000 * 3600  # planning rate, S57 catalog convention


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
    if proc.returncode != 0:
        return {
            "dry_run": f"dead:exit{proc.returncode}",
            "version": "",
            "items": "",
            "bytes": "",
        }

    import json

    try:
        report = json.loads(proc.stdout)
    except ValueError:
        return {"dry_run": "dead:unparseable", "version": "", "items": "", "bytes": ""}
    # A dry run never uploads, so the plan (not the post-run images/bytes counters,
    # which stay 0) carries the real enumeration numbers.
    plan = report.get("plan") or {}
    return {
        "dry_run": "ok",
        "version": str(report.get("version", "")),
        "items": str(plan.get("items", "")),
        "bytes": str(plan.get("declared_bytes", "")),
    }


def process_one(spec_path: Path, manifest_row: dict[str, str]) -> dict[str, str]:
    doc = yaml.safe_load(spec_path.read_text())
    spec_id = doc["id"]
    adapter = doc["adapter"]
    licence = doc.get("license", "")
    priority = manifest_row.get("priority", "")
    label_format = manifest_row.get("label_format", "")

    forced_override = manifest_row.get("dry_run_override", "").strip()
    if adapter in SUPPORTED_ADAPTERS and not forced_override:
        result = _run_dry_run(spec_id, adapter, spec_path)
    elif forced_override:
        # Manifest forces a status without invoking the CLI — e.g. a supported
        # adapter whose scope is too broad/unbounded to enumerate live (benthoz15).
        result = {"dry_run": forced_override, "version": "", "items": "", "bytes": ""}
    else:
        result = {"dry_run": f"needs_adapter:{adapter}", "version": "", "items": "", "bytes": ""}

    bytes_val = result["bytes"]
    est_hours = ""
    if bytes_val.isdigit() and int(bytes_val) > 0:
        est_hours = f"{int(bytes_val) / BYTES_PER_HOUR_AT_100MBPS:.2f}"

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


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: spec_dryrun.py <slice>", file=sys.stderr)
        return 2
    slice_name = argv[1]
    manifest = _load_manifest(slice_name)
    ids = list(manifest.keys())
    if not ids:
        print(f"no manifest rows for slice {slice_name!r}", file=sys.stderr)
        return 2

    queue_path = SPECDIR / f"_queue-{slice_name}.tsv"
    fieldnames = [
        "id",
        "adapter",
        "pinned_version",
        "items",
        "bytes",
        "licence",
        "label_format",
        "dry_run",
        "est_hours_at_100mbps",
        "priority",
    ]
    with queue_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for spec_id in ids:
            spec_path = SPECDIR / f"{spec_id}.yaml"
            if not spec_path.exists():
                print(f"missing spec for {spec_id}, skipping", file=sys.stderr)
                continue
            row = process_one(spec_path, manifest[spec_id])
            writer.writerow(row)
            print(f"{spec_id}: {row['dry_run']}")
    print(f"wrote {queue_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
