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


# ------------------------------------------------------------------ v1.1.0 root files (E4)

LICENCE_PREAMBLE = (
    "This dataset is not under one licence. Every source keeps the licence its authors chose; "
    "the notices below list each source with the annotation licence and the attribution it "
    "requires, and every `metadata` row repeats them. The Apache-2.0 licence of the marine-data "
    "code does not cover the data. The notices record the licences as Reef Support read them "
    "from the primary sources and are not legal advice."
)
NC_PREAMBLE = "This flavour may be used for non-commercial purposes only."


def _value(row: dict, key: str) -> str:
    return " ".join(str(row.get(key) or "-").split())


def _by_source(sources: list[dict]) -> list[dict]:
    return sorted(sources, key=lambda s: str(s.get("source_id")))


def render_license(sources: list[dict], *, pretty_name: str, flavour: str) -> str:
    """``LICENSE``: the per-source licence notices of the ``sources`` config rows."""
    lines = [f"{pretty_name}: licences", "", LICENCE_PREAMBLE, ""]
    if flavour != "open":
        lines += [NC_PREAMBLE, ""]
    for s in _by_source(sources):
        link = f" ({s['licence_url']})" if s.get("licence_url") else ""
        lines += [
            f"{s['source_id']} (version {_value(s, 'version')})",
            f"  Image licence: {_value(s, 'licence')}{link}",
            f"  Annotation licence: {_value(s, 'ann_licence')}",
            f"  Attribution: {_value(s, 'attribution')}",
            "",
        ]
    return "\n".join(lines).rstrip() + "\n"


def render_notice(sources: list[dict], *, pretty_name: str) -> str:
    """``NOTICE``: the rights holder and the attribution of every source, then the MEOW and WoRMS
    credits."""
    from .card_v11_text import MEOW, WORMS

    lines = [
        pretty_name,
        "Copyright (c) Reef Support and the rights holders named below.",
        "",
        "This dataset redistributes material from the sources below. Keep the attribution each "
        "source requires when you redistribute it.",
        "",
    ]
    for s in _by_source(sources):
        lines += [
            f"- {s['source_id']}: {_value(s, 'attribution')}",
            f"  Rights holder: {_value(s, 'rights_holder')}. Source: {_value(s, 'provenance_url')}",
        ]
    return "\n".join([*lines, "", "Ecology and taxonomy", f"- {MEOW}", f"- {WORMS}"]) + "\n"


def render_citation_cff(cff: dict, *, version: str, released: str | None = None) -> str:
    """``CITATION.cff`` of the release: the repository's file with this release's version (and
    date). No DOI is minted here."""
    doc = {**cff, "version": version}
    if released:
        doc["date-released"] = released
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=10_000)
