"""Release driver — ``marinedata release build`` (WS-D S7k).

Loads every admitted source already resolved to a local ``roots`` path, builds every
task declared under ``registry/tasks/*.yaml`` against one frozen ``SPLIT_MAP.json``, and
writes the resulting per-task split manifests plus a ``RELEASE.json`` under
``<out>/releases/<id>/``. Nothing is uploaded.

Resolving ``roots`` (fetching a pinned staged tree per admitted source, or accepting a
local override for a tree that has not been uploaded yet) is the CLI's job — see
:mod:`marinedata.cli_release` — so this module stays pure and directly testable: hand it
a registry and a ``roots`` mapping and it never touches the network.

Splitting under a frozen map, not allocating one, is the point: ``frozen=True`` raises if
a group an admitted source contributes is missing from the map (see
:mod:`marinedata.splitmap`), so a release can never silently grow the shared map or
disagree with a split some other task, or a previous release, already committed to.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .builder import DatasetBuilder
from .checksums import file_digest
from .gate import evaluate
from .registry import Registry
from .splitmap import load_split_map

DEFAULT_SCHEMA_ID = "rs-benthic-v1"


@dataclass(frozen=True)
class TaskManifest:
    """One task's split manifest — sha256/split pairs, sorted for a stable file."""

    task_id: str
    rows: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ReleaseResult:
    release: str
    out_dir: Path
    sources: tuple[str, ...]
    tasks: tuple[TaskManifest, ...]
    skipped_tasks: tuple[tuple[str, str], ...]
    """``(task_id, reason)`` for a task with no permitted source to build from — not an
    error: a partial ``roots`` (a dry run, or a source temporarily unfetchable) simply
    leaves that task out of this release rather than failing the whole build."""


def _admitted_source_ids(
    registry: Registry, roots: dict[str, str | Path], profile: str
) -> list[str]:
    """Sources with a resolved root that also pass the profile's licence gate.

    ``roots`` already answers "has a pinned staged tree" (the caller's job — see the
    module docstring); this only re-applies the licence gate, so a caller cannot smuggle
    a denied source in by handing it a root directly.
    """
    prof = registry.profile(profile)
    admitted = []
    for source_id in sorted(roots):
        source = registry.source(source_id)
        if evaluate(source, prof).allowed:
            admitted.append(source_id)
    return admitted


def build_release(
    registry: Registry,
    *,
    release: str,
    split_map: str | Path,
    roots: dict[str, str | Path],
    out_dir: str | Path,
    profile: str = "research",
    allow_unmapped: bool = False,
) -> ReleaseResult:
    """Build every registry task against a frozen split map and write the release.

    Raises if ``split_map`` does not exist yet (a release never allocates one — see
    :mod:`marinedata.splitmap`), or if any admitted source contributes a group absent
    from it (``SplitMapError``, propagated from ``Dataset.split(frozen=True)``) — both
    left uncaught deliberately, since either means the release is not reproducible yet.
    """
    split_map_path = Path(split_map)
    if load_split_map(split_map_path) is None:
        raise ValueError(
            f"{split_map_path} does not exist — generate it first with "
            "`marinedata splitmap generate`."
        )

    admitted = _admitted_source_ids(registry, roots, profile)
    if not admitted:
        raise ValueError(f"no admitted source under profile {profile!r} has a resolved root")
    admitted_roots = {source_id: Path(roots[source_id]) for source_id in admitted}

    release_root = Path(out_dir) / "releases" / release
    manifests_dir = release_root / "tasks"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    map_bytes = split_map_path.read_bytes()
    (release_root / "SPLIT_MAP.json").write_bytes(map_bytes)
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()

    tasks: list[TaskManifest] = []
    skipped: list[tuple[str, str]] = []

    for task in sorted(registry.tasks, key=lambda t: t.id):
        builder = DatasetBuilder(
            registry,
            profile=profile,
            roots=admitted_roots,
            schema_id=task.schema_id or DEFAULT_SCHEMA_ID,
            task_id=task.id,
            allow_unmapped=allow_unmapped,
        )
        try:
            dataset = builder.build()
        except ValueError as exc:
            # No admitted source can serve this task (e.g. a partial dry-run roots) —
            # not a defect in the release, just an empty task this time.
            skipped.append((task.id, str(exc)[:200]))
            continue

        dataset.split(by="group", split_map=split_map_path, frozen=True, tolerance=None)

        rows: list[tuple[str, str]] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None:
                    continue
                rows.append((file_digest(Path(sample.image)), split_name))
        rows.sort()

        manifest_path = manifests_dir / f"{task.id}.tsv"
        with manifest_path.open("w", newline="") as fh:
            fh.write("image_sha256\tsplit\n")
            for sha256, split_name in rows:
                fh.write(f"{sha256}\t{split_name}\n")

        tasks.append(TaskManifest(task_id=task.id, rows=tuple(rows)))

    release_json = {
        "release": release,
        "split_map_sha256": map_sha256,
        "sources": [
            {
                "id": source_id,
                "version": registry.source(source_id).version,
                "root_digest": (
                    registry.source(source_id).checksums.root_digest
                    if registry.source(source_id).checksums is not None
                    else ""
                ),
            }
            for source_id in admitted
        ],
        "tasks": [task.task_id for task in tasks],
        "skipped_tasks": [{"task": task_id, "reason": reason} for task_id, reason in skipped],
    }
    release_json_text = json.dumps(release_json, indent=2, sort_keys=True) + "\n"
    (release_root / "RELEASE.json").write_text(release_json_text)

    return ReleaseResult(
        release=release,
        out_dir=release_root,
        sources=tuple(admitted),
        tasks=tuple(tasks),
        skipped_tasks=tuple(skipped),
    )
