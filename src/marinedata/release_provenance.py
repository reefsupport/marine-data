"""Public-safe provenance for ``release publish`` (P1.13 delta 1, D1).

``release publish`` ships three files under ``provenance/``. GATES.json and ``release.lock.yaml``
go up as they are (the lock names only the sources that are in the release). RELEASE.json goes up
as :func:`public_release_json`: an allowlist of build facts (counts, config names, ids, digests),
the sources that are in the lock, and the lock's release id, code sha and timestamp. Whatever names
an excluded or skipped source, a bucket, an rclone remote or a local path stays on the build box."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import yaml

from .gates import GATES_FILE
from .release_lock import LOCK_FILE

RELEASE_FILE = "RELEASE.json"
PUBLIC_KEYS = (
    "release", "flavour", "profile", "repo_id", "lock_sha256", "split_map_sha256", "split_ratio",
    "tasks", "configs", "class_counts", "near_dup", "takedown_url",
)  # fmt: skip
LOCK_KEYS = ("release_id", "created", "code_sha")
_PRIVATE = re.compile(r"/Users/|/home/|s3://|rs-hel1|^[/~]|^[A-Za-z]:\\")
_DROP = object()


def _scrub(value):
    """``value`` without any string that looks like a local path, a bucket or the rclone remote."""
    if isinstance(value, dict):
        kept = {k: _scrub(v) for k, v in value.items()}
        return {k: v for k, v in kept.items() if v is not _DROP}
    if isinstance(value, list):
        return [v for v in map(_scrub, value) if v is not _DROP]
    return _DROP if isinstance(value, str) and _PRIVATE.search(value) else value


def public_release_json(release: dict, lock: dict) -> dict:
    """The RELEASE.json that goes on the Hub (see the module docstring); inputs are not changed."""
    ids = {s.get("id") for s in lock.get("sources") or [] if isinstance(s, dict)}
    out = {k: _scrub(release[k]) for k in PUBLIC_KEYS if k in release}
    out = {k: v for k, v in out.items() if v is not _DROP}
    listed = [s for s in release.get("sources") or [] if isinstance(s, dict) and s.get("id") in ids]
    out["sources"] = _scrub(listed)
    return {**out, **{k: lock[k] for k in LOCK_KEYS if k in lock}}


def stage_provenance(target: Path, *, release: Path, gates: Path, lock: Path) -> None:
    """Write ``target`` (``<folder>/provenance/``): the projected RELEASE.json, GATES.json and the
    lock byte for byte. Called at upload time only, so a dry run writes nothing."""
    target.mkdir(exist_ok=True)
    public = public_release_json(
        json.loads(release.read_text()), yaml.safe_load(lock.read_text()) or {}
    )
    (target / RELEASE_FILE).write_text(
        json.dumps(public, indent=2, sort_keys=True, default=str) + "\n"
    )
    shutil.copyfile(gates, target / GATES_FILE)
    shutil.copyfile(lock, target / LOCK_FILE)
