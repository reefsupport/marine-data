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
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# D-AA (2026-09-25 charter): anonymous HF only, no token. On a 429, cool down at
# least this long, then retry the same source (its worker thread just sleeps and
# re-runs it in place — with all sources submitted to the pool up front, a source
# that sleeps 15+ minutes naturally finishes far later than the rest, the same
# outcome as an explicit tail-requeue). After this many cool-downs the source is
# given up on and reported as ``needs-yohan: hf-rate-limit`` instead of ``error``.
HF_COOLDOWN_S = 15 * 60.0
HF_MAX_COOLDOWNS = 3


def _is_hf_rate_limited(result: dict[str, Any]) -> bool:
    if result.get("status") != "error":
        return False
    error = result.get("error", "")
    return "429" in error or "Too Many Requests" in error


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


def _run_one_hf_aware(
    spec_path: Path,
    work_root: Path,
    client: Any,
    *,
    hf_semaphore: threading.Semaphore,
    cooldown_s: float,
    max_cooldowns: int,
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    """``run_one``, plus D-AA: on an HF 429 cool down and retry in place (keeping HF
    concurrency at 1 via ``hf_semaphore``, regardless of ``--jobs``), up to
    ``max_cooldowns`` times before giving up as ``needs-yohan: hf-rate-limit``."""
    from .ingest_source import IngestSpec

    spec = IngestSpec.load(spec_path)
    is_hf = spec.adapter == "hf"
    cooldowns = 0
    while True:
        if is_hf:
            with hf_semaphore:
                res = run_one(spec_path, work_root, client)
        else:
            res = run_one(spec_path, work_root, client)
        if not (is_hf and _is_hf_rate_limited(res)):
            return res
        cooldowns += 1
        if cooldowns >= max_cooldowns:
            return {
                **res,
                "status": "needs-yohan",
                "needs": "hf-rate-limit",
                "cooldowns": cooldowns,
            }
        sleep(cooldown_s)


def run_batch(
    specs_dir: Path,
    work_root: Path,
    *,
    only: set[str] | None = None,
    jobs: int = 1,
    remote: str = "rs-hel1",
    out: Any = None,
    hf_cooldown_s: float = HF_COOLDOWN_S,
    hf_max_cooldowns: int = HF_MAX_COOLDOWNS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Run every spec under ``specs_dir``; return the per-source result dicts in spec
    order. One JSONL line per finished source is written to ``out`` as it completes.
    ``out`` defaults to the CURRENT ``sys.stdout`` at call time (not import time — a
    stdlib default-argument trap that would otherwise escape test capture/redirection).

    D-AA: HF sources never run more than one at a time (``hf_semaphore``, independent
    of ``--jobs``), and an HF 429 is cooled down and retried rather than a fatal error
    (see ``_run_one_hf_aware``)."""
    out = sys.stdout if out is None else out
    paths = _spec_paths(specs_dir, only)
    client = _resolve_client(remote)
    hf_semaphore = threading.Semaphore(1)
    results: dict[str, dict[str, Any]] = {}
    if jobs <= 1:
        for p in paths:
            res = _run_one_hf_aware(
                p,
                work_root,
                client,
                hf_semaphore=hf_semaphore,
                cooldown_s=hf_cooldown_s,
                max_cooldowns=hf_max_cooldowns,
                sleep=sleep,
            )
            print(json.dumps(res, sort_keys=True), file=out, flush=True)
            results[str(p)] = res
    else:
        with cf.ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    _run_one_hf_aware,
                    p,
                    work_root,
                    client,
                    hf_semaphore=hf_semaphore,
                    cooldown_s=hf_cooldown_s,
                    max_cooldowns=hf_max_cooldowns,
                    sleep=sleep,
                ): p
                for p in paths
            }
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
