"""``marinedata ingest-batch <specs_dir> [--only id,…] [--jobs N]`` (WP-6c).

Runs ``ingest-source`` for every spec under ``specs_dir``, resumably: a source whose
``CHECKSUMS.sha256`` marker already exists at its resolved ``sources/<id>/<version>/``
prefix is skipped without re-downloading (re-running the whole batch after a Job restart
just picks up where it left off). Each source emits one JSONL line to stdout as it
finishes; a final summary line collects every ``needs-yohan`` source. The process exits
non-zero only when a source raised something other than the adapters' own
``AccessRefused`` (a real failure — see ``docs/design/ingestion.md``).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def _spec_paths(specs_dir: Path, only: set[str] | None) -> list[Path]:
    paths = sorted({*specs_dir.glob("*.yaml"), *specs_dir.glob("*.yml")})
    if only is None:
        return paths
    kept = [p for p in paths if (yaml.safe_load(p.read_text()) or {}).get("id") in only]
    missing = only - {(yaml.safe_load(p.read_text()) or {}).get("id") for p in kept}
    if missing:
        raise ValueError(f"--only names sources not found under {specs_dir}: {sorted(missing)}")
    return kept


def _resolve_client(remote: str) -> Any:
    """Env-var creds first (the cluster Job path), rclone.conf as the local fallback."""
    import os

    from .s3_upload import client_from_env, client_from_rclone

    if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return client_from_env()
    return client_from_rclone(remote)


def _marker_present(client: Any, spec: Any, version: str) -> bool:
    from .checksums import CHECKSUM_FILE
    from .s3_upload import _head

    key = f"{spec.prefix}/{spec.id}/{version}/{CHECKSUM_FILE}"
    return _head(client, spec.bucket, key) is not None


def run_one(spec_path: Path, work_root: Path, client: Any) -> dict[str, Any]:
    """Run (or skip) a single spec. Never raises for ``AccessRefused``; a real exception
    is caught, reported as ``status: error``, and re-raised by the caller's own check of
    that dict — this function's return value is always what gets JSON-serialised."""
    from .adapters import AccessRefused, make_adapter
    from .ingest_source import IngestSpec, run_ingest

    t0 = time.monotonic()
    spec = IngestSpec.load(spec_path)
    elapsed = lambda: round(time.monotonic() - t0, 1)  # noqa: E731
    try:
        adapter = make_adapter(spec.adapter, spec.params)
        version = spec.version or adapter.resolve_version()
        if _marker_present(client, spec, version):
            return {
                "source_id": spec.id,
                "version": version,
                "status": "skip-done",
                "elapsed_s": elapsed(),
            }
        work = work_root / spec.id
        work.mkdir(parents=True, exist_ok=True)
        report = run_ingest(spec, work, client=client)
        return {
            "source_id": report.source_id,
            "version": report.version,
            "status": "ok",
            "images": report.images,
            "files": report.files,
            "uploaded": report.uploaded,
            "skipped": report.skipped,
            "verified": report.verified,
            "root_digest": report.root_digest,
            "elapsed_s": elapsed(),
        }
    except AccessRefused as exc:
        return {
            "source_id": spec.id,
            "status": "needs-yohan",
            "url": exc.url,
            "needs": exc.needs,
            "elapsed_s": elapsed(),
        }
    except Exception as exc:  # surfaced in the "error" status below, not swallowed
        return {
            "source_id": spec.id,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": elapsed(),
        }


def run_batch(
    specs_dir: Path,
    work_root: Path,
    *,
    only: set[str] | None = None,
    jobs: int = 1,
    remote: str = "rs-hel1",
    out: Any = None,
) -> list[dict[str, Any]]:
    """Run every spec under ``specs_dir``; return the per-source result dicts in spec
    order. One JSONL line per finished source is written to ``out`` as it completes.
    ``out`` defaults to the CURRENT ``sys.stdout`` at call time (not import time — a
    stdlib default-argument trap that would otherwise escape test capture/redirection)."""
    out = sys.stdout if out is None else out
    paths = _spec_paths(specs_dir, only)
    client = _resolve_client(remote)
    results: dict[str, dict[str, Any]] = {}
    if jobs <= 1:
        for p in paths:
            res = run_one(p, work_root, client)
            print(json.dumps(res, sort_keys=True), file=out, flush=True)
            results[str(p)] = res
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(run_one, p, work_root, client): p for p in paths}
            for fut in cf.as_completed(futures):
                res = fut.result()
                print(json.dumps(res, sort_keys=True), file=out, flush=True)
                results[str(futures[fut])] = res
    return [results[str(p)] for p in paths]


def _cmd_ingest_batch(args: argparse.Namespace) -> int:
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    try:
        ordered = run_batch(
            Path(args.specs_dir),
            Path(args.work),
            only=only,
            jobs=args.jobs,
            remote=args.remote,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    needs_yohan = [r for r in ordered if r["status"] == "needs-yohan"]
    errors = [r for r in ordered if r["status"] == "error"]
    summary = {
        "total": len(ordered),
        "ok": sum(1 for r in ordered if r["status"] == "ok"),
        "skip_done": sum(1 for r in ordered if r["status"] == "skip-done"),
        "needs_yohan": needs_yohan,
        "errors": errors,
    }
    print(json.dumps(dataclasses.asdict(_Summary(**summary)), sort_keys=True))
    return 1 if errors else 0


@dataclasses.dataclass
class _Summary:
    total: int
    ok: int
    skip_done: int
    needs_yohan: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def add_ingest_batch_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "ingest-batch",
        help="Run ingest-source over every spec in a directory, resumably (WP-6c server Job)",
    )
    p.add_argument("specs_dir", help="Directory of ingest spec yaml files (one per source)")
    p.add_argument("--work", default="work", help="Work/temp root (one subdir per source)")
    p.add_argument("--only", help="Comma-separated source ids to run (default: all)")
    p.add_argument("--jobs", type=int, default=1, help="Sources to run concurrently")
    p.add_argument("--remote", default="rs-hel1", help="rclone remote name (local fallback)")
    p.set_defaults(func=_cmd_ingest_batch)
