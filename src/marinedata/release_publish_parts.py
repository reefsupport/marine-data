"""``marinedata release publish --part``: one part per call, and the tag on the sha they made.

Parts, in upload order:

* ``data:<config>``: ``data/<config>/*.parquet``, one part per directory under ``data/`` that holds
  a parquet shard (``upload_large_folder``: resumable, batched commits);
* ``provenance``: ``provenance/*`` (written at upload time), ``croissant.json``, ``LICENSE``,
  ``NOTICE``, ``CITATION.cff``, ``assets/*`` and ``docs/**``: everything the card links to;
* ``card``: ``README.md`` alone, one commit, uploaded last;
* ``tag``: ``create_tag`` on the commit sha the card upload returned, never on the branch head.

Without ``--part`` every part runs in that order; the build VM runs them one per call
(``--list-parts`` prints the names). The sha of each upload commit is kept in ``PUBLISH_STATE.json``
next to ``RELEASE.json``, so ``--part tag`` in a later process tags the commit the uploads made, and
it refuses when the last upload was not the card: a stale head is never tagged. ``card`` refuses
until every data part and ``provenance`` are uploaded, and a tag that exists stops every part.

For the nc flavour ``repo_info(...).private`` must be ``True`` before any upload. Nothing here
creates a repo or changes visibility (``upload_large_folder`` only ever sees a repo that
``list_repo_refs`` has just found, so its own ``create_repo(exist_ok=True)`` changes nothing).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import release_publish as rp
from .gates import gate_status, read_gates
from .release_publish import PublishPlan, PublishRefused

DATA_PREFIX = "data:"
PART_PROVENANCE = "provenance"
PART_CARD = "card"
PART_TAG = "tag"
STATE_FILE = "PUBLISH_STATE.json"
MAX_LISTED = 200
"""Added paths the plan prints before it says how many more there are."""
GATE_LINES = (("frozen", "G2 frozen v1.0"), ("leak", "G3 leak"))
_CONFIG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_SHA = re.compile(r"[0-9a-f]{40}")
_WORD = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIPPED"}


def files_matching(folder: Path, patterns: Sequence[str]) -> list[tuple[str, int]]:
    """``(relative path, bytes)`` of every file under ``folder`` that an allow-pattern matches."""
    out = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns):
            out.append((rel, path.stat().st_size))
    return out


def data_configs(folder: Path) -> tuple[str, ...]:
    """The config directories under ``<folder>/data`` that hold at least one parquet shard."""
    root = folder / "data"
    if not root.is_dir():
        return ()
    return tuple(
        sorted(d.name for d in root.iterdir() if d.is_dir() and next(d.rglob("*.parquet"), None))
    )


def all_parts(folder: Path) -> tuple[str, ...]:
    return (*(DATA_PREFIX + c for c in data_configs(folder)), PART_PROVENANCE, PART_CARD, PART_TAG)


def select_parts(plan: PublishPlan, part: str | None) -> tuple[str, ...]:
    """Every part in upload order, or the one ``--part`` names (refused when it is not a part)."""
    configs = data_configs(plan.folder)
    if not configs:
        raise PublishRefused([f"no parquet shard under {plan.folder}/data/<config>/"])
    if part is None:
        return all_parts(plan.folder)
    if part in (PART_PROVENANCE, PART_CARD, PART_TAG):
        return (part,)
    config = part[len(DATA_PREFIX) :] if part.startswith(DATA_PREFIX) else ""
    if _CONFIG.fullmatch(config) and config in configs:
        return (part,)
    raise PublishRefused(
        [
            f"--part {part!r} is not a part of this build: use {DATA_PREFIX}<config> "
            f"({', '.join(configs)}), {PART_PROVENANCE}, {PART_CARD} or {PART_TAG}"
        ]
    )


def part_patterns(part: str) -> tuple[str, ...]:
    """The ``allow_patterns`` of one part (the tag uploads nothing)."""
    if part.startswith(DATA_PREFIX):
        return (f"data/{part[len(DATA_PREFIX) :]}/*.parquet",)
    if part == PART_PROVENANCE:
        return rp.PROVENANCE_PATTERNS
    if part == PART_CARD:
        return (rp.CARD_FILE,)
    return ()


def _provenance_sources(plan: PublishPlan) -> dict[str, Path]:
    """The files ``provenance/`` is staged from, by their name in that folder."""
    base = plan.release_dir or plan.folder
    return {
        rp.RELEASE_FILE: base / rp.RELEASE_FILE,
        rp.GATES_FILE: plan.folder / rp.GATES_FILE,
        rp.LOCK_FILE: plan.lock_file or base / rp.LOCK_FILE,
    }


def part_files(plan: PublishPlan, part: str) -> list[tuple[str, int]]:
    """What one part sends. ``provenance/`` does not exist before the upload: it is listed from
    the files it is staged from."""
    found = dict(files_matching(plan.folder, part_patterns(part)))
    if part == PART_PROVENANCE:
        for name, source in _provenance_sources(plan).items():
            size = source.stat().st_size if source.is_file() else 0
            found.setdefault(f"{rp.PROVENANCE_DIR}/{name}", size)
    return sorted(found.items())


# --- the delta against the v1.0 tree -------------------------------------------------------


@dataclass(frozen=True)
class V1Delta:
    tree: Path
    v1_files: int
    added: tuple[str, ...]
    replaced: tuple[str, ...]
    identical: int

    @property
    def shards_replaced(self) -> tuple[str, ...]:
        """v1.0 data shards the upload would replace: frozen, so G2 must have caught them."""
        return tuple(p for p in self.replaced if fnmatch.fnmatchcase(p, rp.DATA_PATTERN))


def _tree(root: Path) -> dict[str, Path]:
    skip = {".git", ".cache"}
    return {
        p.relative_to(root).as_posix(): p
        for p in sorted(root.rglob("*"))
        if p.is_file() and not skip & set(p.relative_to(root).parts)
    }


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def v1_delta(plan: PublishPlan, v1_dir: Path) -> V1Delta:
    """Every path the publish sends, against a checkout of the v1.0 repo: added, replaced
    (same path, different bytes) or identical."""
    v1_dir = Path(v1_dir)
    if not v1_dir.is_dir():
        raise PublishRefused([f"--v1-dir is not a directory: {v1_dir}"])
    old = _tree(v1_dir)
    sends = {rel: plan.folder / rel for rel, _ in files_matching(plan.folder, rp.UPLOAD_PATTERNS)}
    for name, source in _provenance_sources(plan).items():
        sends.setdefault(f"{rp.PROVENANCE_DIR}/{name}", source)
    added, replaced, identical = [], [], 0
    for rel, local in sorted(sends.items()):
        if rel not in old:
            added.append(rel)
        elif not local.is_file() or local.stat().st_size != old[rel].stat().st_size:
            replaced.append(rel)
        elif _digest(local) == _digest(old[rel]):
            identical += 1
        else:
            replaced.append(rel)
    return V1Delta(v1_dir, len(old), tuple(added), tuple(replaced), identical)


# --- the plan ------------------------------------------------------------------------------


def gate_lines(folder: Path, *, only_if_run: bool = False) -> list[str]:
    """G2 (frozen v1.0) and G3 (leak) as ``PASS`` / ``FAIL`` / ``SKIPPED`` / ``MISSING`` lines.
    ``only_if_run`` prints nothing for a build that never ran either gate (a v1.0-style build)."""
    try:
        gates = read_gates(Path(folder))
    except (OSError, ValueError):
        gates = {}
    if only_if_run and not any(key in gates for key, _ in GATE_LINES):
        return []
    lines = []
    for key, label in GATE_LINES:
        entry = gates.get(key)
        word = "MISSING" if key not in gates else _WORD[gate_status(entry)]
        why = ""
        if word == "FAIL" and isinstance(entry, dict) and entry.get("failures"):
            why = f" ({str(entry['failures'][0])[:120]})"
        elif word == "SKIPPED":
            why = f" ({entry.get('skipped')})"
        lines.append(f"  {label:<14}: {word}{why}")
    return lines


def _describe(plan: PublishPlan, part: str) -> str:
    if part == PART_TAG:
        return f"    {part}: {plan.tag} on the sha of the card commit (read from {STATE_FILE})"
    files = part_files(plan, part)
    head = f"    {part}: {len(files)} file{'s' if len(files) != 1 else ''}, "
    head += f"{sum(size for _, size in files):,} bytes"
    if part.startswith(DATA_PREFIX):
        return head
    return head + ": " + ", ".join(rel for rel, _ in files)


def render_parts_plan(plan: PublishPlan, selected: Sequence[str], *, delta: V1Delta | None) -> str:
    """The part list, the visibility rule, the delta against the v1.0 tree and G2/G3."""
    every = all_parts(plan.folder)
    this_run = "every part, in this order" if tuple(selected) == every else ", ".join(selected)
    lines = [f"  parts       : {len(every)} ({', '.join(every)})", f"  this run    : {this_run}"]
    lines += [_describe(plan, part) for part in selected]
    if plan.flavour == "nc":
        lines.append(
            "  visibility  : the repo must be private: checked before every upload; "
            "this command never changes visibility"
        )
    else:
        lines.append("  visibility  : never changed by this command")
    if delta is None:
        lines.append(
            "  prior tree  : not given (--v1-dir DIR, a checkout of the prior repo): no delta"
        )
    else:
        lines.append(
            f"  prior tree  : {delta.tree} ({delta.v1_files} files): adds {len(delta.added)}, "
            f"replaces {len(delta.replaced)}, identical {delta.identical}"
        )
        lines += [f"    + {rel}" for rel in delta.added[:MAX_LISTED]]
        if len(delta.added) > MAX_LISTED:
            lines.append(f"    ... and {len(delta.added) - MAX_LISTED} more added paths")
        for rel in delta.replaced:
            note = "  (replaces a prior-release shard)" if rel in delta.shards_replaced else ""
            lines.append(f"    ~ {rel}{note}")
    return "\n".join([*lines, *gate_lines(plan.folder, only_if_run=True)])


# --- the upload ----------------------------------------------------------------------------


def _identity(plan: PublishPlan) -> dict:
    return {
        "repo_id": plan.repo_id,
        "release": plan.release_id,
        "flavour": plan.flavour,
        "tag": plan.tag,
        "lock_sha256": plan.lock_sha256,
    }


def state_path(plan: PublishPlan) -> Path:
    return (plan.release_dir or plan.folder) / STATE_FILE


def load_state(plan: PublishPlan) -> dict:
    """The uploads recorded so far (``uploads``: part and commit sha, in order)."""
    path, identity = state_path(plan), _identity(plan)
    if not path.is_file():
        return {**identity, "uploads": []}
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        raise PublishRefused([f"{path} is not valid JSON ({exc})"]) from exc
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("uploads"), list)
        or any(data.get(key) != value for key, value in identity.items())
    ):
        raise PublishRefused([f"{path} records another publish (repo, release, flavour or lock)"])
    return data


def save_state(plan: PublishPlan, state: dict) -> None:
    path = state_path(plan)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _record(state: dict, part: str, sha: str | None) -> dict:
    return {**state, "uploads": [*state["uploads"], {"part": part, "sha": sha}]}


def _commit_sha(info: object) -> str:
    sha = getattr(info, "oid", None)
    if not isinstance(sha, str) or not _SHA.fullmatch(sha):
        raise PublishRefused(["the upload returned no commit sha: not recording it, not tagging"])
    return sha


def _require_private(api, plan: PublishPlan) -> None:
    if getattr(api.repo_info(plan.repo_id, repo_type="dataset"), "private", None) is not True:
        raise PublishRefused(
            [
                f"{plan.repo_id} is not private: the nc flavour is never uploaded to a public "
                "repo (this command never changes visibility)"
            ]
        )


def _check_order(plan: PublishPlan, part: str, state: dict) -> None:
    done = [u.get("part") for u in state["uploads"] if isinstance(u, dict)]
    if part == PART_CARD:
        missing = [p for p in all_parts(plan.folder)[:-2] if p not in done]
        if missing:
            raise PublishRefused([f"card is uploaded last: not uploaded yet: {', '.join(missing)}"])
    if part == PART_TAG:
        last = state["uploads"][-1] if state["uploads"] else {}
        if last.get("part") != PART_CARD or not _SHA.fullmatch(str(last.get("sha"))):
            what = last.get("part") or "nothing"
            raise PublishRefused(
                [
                    f"tag needs the card as the last upload (that is {what}): run the missing "
                    f"parts, then --part {PART_CARD}; the tag is never put on the branch head"
                ]
            )


def _upload(api, plan: PublishPlan, part: str) -> str | None:
    kwargs = {
        "repo_id": plan.repo_id,
        "repo_type": "dataset",
        "folder_path": str(plan.folder),
        "allow_patterns": list(part_patterns(part)),
    }
    if part.startswith(DATA_PREFIX):
        api.upload_large_folder(**kwargs)
        return None
    if part == PART_PROVENANCE:
        rp._stage_provenance(plan)
    message = f"marinedata release {plan.release_id} ({plan.flavour}): {part}"
    return _commit_sha(api.upload_folder(**kwargs, commit_message=message))


def _tag(api, plan: PublishPlan, sha: str) -> None:
    try:
        api.create_tag(
            plan.repo_id,
            tag=plan.tag,
            revision=sha,
            tag_message=f"marinedata release {plan.release_id} ({plan.flavour}), "
            f"lock {plan.lock_sha256}",
            repo_type="dataset",
            exist_ok=False,
        )
    except Exception as exc:
        raise PublishRefused(
            [f"uploaded {plan.repo_id} but tagging {plan.tag} on {sha} failed "
             f"({type(exc).__name__}): tag it by hand"]
        ) from exc  # fmt: skip


def publish(plan: PublishPlan, token: str, parts: Sequence[str], *, api=None) -> list[str]:
    """Run ``parts`` in the order given; returns the parts done. Every upload is recorded in
    :data:`STATE_FILE` before the next part starts."""
    api = api if api is not None else rp._hf_api(token)
    if plan.flavour == "nc":
        _require_private(api, plan)
    tags = {ref.name for ref in api.list_repo_refs(plan.repo_id, repo_type="dataset").tags}
    if plan.tag in tags:
        raise PublishRefused(
            [f"tag {plan.tag} already exists on {plan.repo_id}: not published twice"]
        )
    state = load_state(plan)
    for part in parts:
        _check_order(plan, part, state)
        if part == PART_TAG:
            sha = state["uploads"][-1]["sha"]
            _tag(api, plan, sha)
            save_state(plan, {**state, "tagged": sha})
        else:
            state = _record(state, part, _upload(api, plan, part))
            save_state(plan, state)
    return list(parts)
