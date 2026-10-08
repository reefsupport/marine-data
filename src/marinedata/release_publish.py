"""``marinedata release publish``: the last step of a release, behind its guards (Phase A, P1.7).

Without ``--go`` it prints the plan and uploads nothing: the target repo, the configs from
RELEASE.json, the file count and bytes, the card, the lock sha and the tag. It refuses (exit 1,
only the reasons are printed) when RELEASE.json, its lock sha, the card or GATES.json is missing,
when the build is not the flavour asked for, or when a gate required for publishing did not pass.

``--go`` adds three guards before anything leaves the machine: ``HF_TOKEN`` in the environment,
the frozen v1.0 release stays untouched unless ``--allow-frozen``, and the tag must not exist on
the repo yet. It then makes one ``upload_folder`` commit (bookkeeping files left out, see
:data:`NOT_UPLOADED`) and tags that commit. It never creates a repo: both exist already and
their visibility is set by hand.

The gate verdict mirrors ``gates.verdict`` (P1.6): a gate counts as required when
``summary.required_for_publish`` lists it, or when its entry says ``required: true``; with no
summary the default set (:data:`DEFAULT_REQUIRED_GATES`) applies. A required gate that is
missing, skipped or failed blocks the publish.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .flavours import flavour_spec
from .gates import GATES_FILE
from .release_lock import LOCK_FILE, lock_sha256

RELEASE_FILE = "RELEASE.json"
CARD_FILE = "README.md"
TOKEN_ENV = "HF_TOKEN"
SUMMARY_KEY = "summary"
DEFAULT_REQUIRED_GATES = ("licence", "dedup", "splits", "masks", "load", "card")
FROZEN_VERSION = (1,)
"""The v1.0 release (``1``, ``1.0``, ``v1.0``, ``v1.0.0``): live on the Hub and frozen."""
NOT_UPLOADED = (
    RELEASE_FILE,
    GATES_FILE,
    LOCK_FILE,
    "SPLIT_MAP.json",
    "DEDUP_GATE.json",
    "tasks/*",
    "*.tmp",
    ".git/*",
    ".cache/*",
)
"""Release bookkeeping stays off the Hub (it names excluded sources); the plan counts the rest.
The lock, gates and RELEASE.json are mirrored to the bucket instead (plan section 4)."""

_VERSION = re.compile(r"v?(\d+(?:\.\d+)*)")


class PublishRefused(Exception):
    """One or more reasons the publish cannot go ahead; ``problems`` holds one line each."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("; ".join(problems))
        self.problems = list(problems)


@dataclass(frozen=True)
class PublishPlan:
    release_id: str
    flavour: str
    repo_id: str
    tag: str
    folder: Path
    files: tuple[tuple[str, int], ...]
    configs: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    card: Path
    lock_sha256: str
    required_gates: tuple[str, ...]

    @property
    def nbytes(self) -> int:
        return sum(size for _, size in self.files)


def is_frozen(release_id: str) -> bool:
    match = _VERSION.fullmatch(release_id)
    if match is None:
        return False
    parts = [int(p) for p in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts) == FROZEN_VERSION


def tag_for(release_id: str) -> str:
    """``v<release-id>``; an id that already reads ``v1.1`` is not prefixed twice."""
    return release_id if re.match(r"v\d", release_id) else f"v{release_id}"


def _gate_status(entry: object) -> str:
    if not isinstance(entry, dict):
        return "failed"
    if entry.get("skipped") is not None:
        return "skipped"
    return "passed" if entry.get("passed") is True else "failed"


def gate_verdict(directory: Path) -> tuple[tuple[str, ...], list[str]]:
    """``(required gates, problems)`` for ``<directory>/GATES.json``; no problems = publishable."""
    path = directory / GATES_FILE
    if not path.is_file():
        return DEFAULT_REQUIRED_GATES, [
            f"{GATES_FILE} missing in {directory}: run 'marinedata release gate --all'"
        ]
    try:
        gates = json.loads(path.read_text())
    except ValueError as exc:
        return DEFAULT_REQUIRED_GATES, [f"{path} is not valid JSON ({exc})"]
    if not isinstance(gates, dict):
        return DEFAULT_REQUIRED_GATES, [f"{path} is not a JSON object"]
    entries = {k: v for k, v in gates.items() if k != SUMMARY_KEY}
    summary = gates.get(SUMMARY_KEY)
    listed = summary.get("required_for_publish") if isinstance(summary, dict) else None
    base = [str(n) for n in listed] if isinstance(listed, list) else list(DEFAULT_REQUIRED_GATES)
    flagged = [n for n, e in entries.items() if isinstance(e, dict) and e.get("required") is True]
    required = tuple(dict.fromkeys([*base, *flagged]))
    problems = []
    for name in required:
        entry = entries.get(name)
        status = _gate_status(entry) if name in entries else "missing"
        line = f"gate {name}: {status}"
        if status == "skipped":
            line += f" ({entry['skipped']})"
        elif status == "failed" and isinstance(entry, dict) and entry.get("failures"):
            line += f" ({entry['failures'][0]})"
        if status != "passed":
            problems.append(line)
    return required, problems


def _load_json(path: Path, what: str, problems: list[str]) -> dict:
    if not path.is_file():
        problems.append(f"{what} missing: {path}")
        return {}
    try:
        data = json.loads(path.read_text())
    except ValueError as exc:
        problems.append(f"{what} is not valid JSON: {path} ({exc})")
        return {}
    if not isinstance(data, dict):
        problems.append(f"{what} is not a JSON object: {path}")
        return {}
    return data


def _split_rows(value: object) -> dict[str, int]:
    """Rows per split from ``{split: n}`` or hf_export's ``{"splits": {split: {"rows": n}}}``."""
    if not isinstance(value, dict):
        return {}
    splits = value.get("splits") if isinstance(value.get("splits"), dict) else value
    out = {}
    for split, entry in splits.items():
        rows = entry.get("rows") if isinstance(entry, dict) else entry
        if isinstance(rows, int) and not isinstance(rows, bool):
            out[str(split)] = rows
    return out


def release_configs(release: dict) -> dict[str, dict[str, int]]:
    """Config name -> rows per split, from RELEASE.json ``configs`` (a mapping or a name list)."""
    raw = release.get("configs")
    if isinstance(raw, dict):
        return {str(name): _split_rows(value) for name, value in raw.items()}
    if isinstance(raw, list):
        names = [c.get("name") if isinstance(c, dict) else c for c in raw]
        return {str(n): {} for n in names if n}
    return {}


def upload_files(folder: Path) -> list[tuple[str, int]]:
    """What ``upload_folder`` will send: every file under ``folder`` except :data:`NOT_UPLOADED`."""
    out = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if not any(fnmatch.fnmatchcase(rel, pattern) for pattern in NOT_UPLOADED):
            out.append((rel, path.stat().st_size))
    return out


def go_problems(release_id: str, *, allow_frozen: bool, token: str) -> list[str]:
    """Guards that only apply to ``--go`` and need nothing from the disk."""
    problems = []
    if is_frozen(release_id) and not allow_frozen:
        problems.append(
            f"release {release_id} is the frozen v1.0 (live on the Hub): --go needs --allow-frozen"
        )
    if not token:
        problems.append(f"{TOKEN_ENV} is not set: export it on the build VM, never in the repo")
    return problems


def build_plan(
    release_id: str,
    flavour: str,
    folder: Path,
    release_dir: Path | None = None,
    lock: Path | None = None,
) -> PublishPlan:
    """The publish plan, or :class:`PublishRefused` listing every reason it cannot go ahead."""
    release_dir = release_dir or folder
    problems: list[str] = []
    if not folder.is_dir():
        raise PublishRefused([f"not a directory: {folder}"])
    release = _load_json(release_dir / RELEASE_FILE, RELEASE_FILE, problems)
    repo_id = flavour_spec(flavour).repo_id
    if release:
        if release.get("flavour") != flavour:
            problems.append(
                f"{RELEASE_FILE} was built as flavour {release.get('flavour')!r}, not {flavour!r}"
            )
        if release.get("repo_id") not in (None, repo_id):
            problems.append(
                f"{RELEASE_FILE} targets {release.get('repo_id')!r}, flavour {flavour} is {repo_id}"
            )
        if release.get("release") not in (None, release_id):
            problems.append(
                f"{RELEASE_FILE} is release {release.get('release')!r}, not {release_id!r}"
            )
    configs = release_configs(release)
    if release and not configs:
        problems.append(f"{RELEASE_FILE} lists no configs")
    recorded = str(release.get("lock_sha256") or "")
    if release and not recorded:
        problems.append(f"{RELEASE_FILE} has no lock_sha256: the build was not made from a lock")
    lock_file = lock or release_dir / LOCK_FILE
    if recorded and lock_file.is_file() and lock_sha256(lock_file) != recorded:
        problems.append(f"{lock_file} does not hash to the lock_sha256 in {RELEASE_FILE}")
    if recorded and lock is not None and not lock.is_file():
        problems.append(f"lock file missing: {lock}")
    card = folder / CARD_FILE
    if not card.is_file():
        problems.append(f"card missing: {card}")
    required, gate_issues = gate_verdict(folder)
    problems.extend(gate_issues)
    files = upload_files(folder)
    if not files:
        problems.append(f"nothing to upload under {folder}")
    if problems:
        raise PublishRefused(problems)
    return PublishPlan(
        release_id=release_id,
        flavour=flavour,
        repo_id=repo_id,
        tag=tag_for(release_id),
        folder=folder,
        files=tuple(files),
        configs=tuple((n, tuple(sorted(rows.items()))) for n, rows in sorted(configs.items())),
        card=card,
        lock_sha256=recorded,
        required_gates=required,
    )


def _human(nbytes: int) -> str:
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1000 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1000
    return f"{size:.1f} GB"  # unreachable; keeps the type checker honest


def render_plan(plan: PublishPlan, *, go: bool) -> str:
    head = "UPLOAD (--go)" if go else "DRY RUN (nothing is uploaded; add --go to upload)"
    frozen = "  [frozen v1.0]" if is_frozen(plan.release_id) else ""
    lines = [
        f"release publish: {head}",
        f"  release     : {plan.release_id}{frozen}",
        f"  flavour     : {plan.flavour}",
        f"  repo        : {plan.repo_id} (dataset)",
        f"  tag         : {plan.tag}",
        f"  folder      : {plan.folder}",
        f"  files       : {len(plan.files)} ({plan.nbytes:,} bytes, {_human(plan.nbytes)})",
        f"  left out    : {', '.join(NOT_UPLOADED[:5])}, tasks/*",
        f"  configs     : {len(plan.configs)}",
    ]
    for name, rows in plan.configs:
        detail = ", ".join(f"{split} {n}" for split, n in rows)
        total = f" = {sum(n for _, n in rows)} rows" if rows else ""
        lines.append(f"    {name}" + (f": {detail}{total}" if rows else ""))
    lines += [
        f"  card        : {plan.card}",
        f"  lock sha256 : {plan.lock_sha256}",
        f"  gates       : {len(plan.required_gates)} required, all passed "
        f"({', '.join(plan.required_gates)})",
    ]
    return "\n".join(lines)


def _hf_api(token: str):
    from huggingface_hub import HfApi

    return HfApi(token=token)


def upload(plan: PublishPlan, token: str, *, api=None) -> str:
    """One ``upload_folder`` commit, then ``create_tag`` on that commit; returns the tag.

    Never creates a repo (``upload_folder`` fails on a missing one) and never moves a tag:
    an existing tag stops the run before anything is sent."""
    api = api if api is not None else _hf_api(token)
    tags = {ref.name for ref in api.list_repo_refs(plan.repo_id, repo_type="dataset").tags}
    if plan.tag in tags:
        raise PublishRefused(
            [f"tag {plan.tag} already exists on {plan.repo_id}: not published twice"]
        )
    commit = api.upload_folder(
        repo_id=plan.repo_id,
        repo_type="dataset",
        folder_path=str(plan.folder),
        ignore_patterns=list(NOT_UPLOADED),
        commit_message=f"Release {plan.tag} ({plan.flavour})",
    )
    oid = getattr(commit, "oid", None)
    try:
        api.create_tag(
            plan.repo_id,
            tag=plan.tag,
            revision=oid,
            tag_message=f"marinedata release {plan.release_id} ({plan.flavour}), "
            f"lock {plan.lock_sha256}",
            repo_type="dataset",
            exist_ok=False,
        )
    except Exception as exc:
        raise PublishRefused(
            [f"uploaded commit {oid} to {plan.repo_id} but tagging {plan.tag} failed "
             f"({type(exc).__name__}): tag it by hand"]
        ) from exc  # fmt: skip
    return plan.tag


def _refuse(problems: list[str]) -> int:
    for line in problems:
        print(f"release publish: refused: {line}", file=sys.stderr)
    return 1


def cmd_release_publish(args: argparse.Namespace) -> int:
    token = os.environ.get(TOKEN_ENV, "")
    if args.go:
        guards = go_problems(args.release_id, allow_frozen=args.allow_frozen, token=token)
        if guards:
            return _refuse(guards)
    folder = args.folder or args.out / "releases" / args.release_id / args.flavour
    try:
        plan = build_plan(args.release_id, args.flavour, folder, args.release_dir, args.lock)
    except PublishRefused as exc:
        return _refuse(exc.problems)
    print(render_plan(plan, go=args.go))
    if not args.go:
        return 0
    try:
        tag = upload(plan, token)
    except PublishRefused as exc:
        return _refuse(exc.problems)
    except Exception as exc:  # network and Hub errors: say which, never echo the token
        print(f"release publish: upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"release publish: uploaded {plan.repo_id} and tagged {tag}")
    return 0


def add_publish_parser(release_sub: argparse._SubParsersAction) -> None:
    """Wire ``release publish`` onto the ``release`` sub-parser (called from ``cli_release``)."""
    p = release_sub.add_parser(
        "publish",
        help="Upload a gated release to its Hub repo and tag it (dry run unless --go)",
    )
    p.add_argument("--release-id", dest="release_id", required=True)
    p.add_argument("--flavour", choices=("open", "nc"), required=True)
    p.add_argument(
        "--dir", dest="folder", type=Path, default=None, metavar="BUILD_DIR",
        help="the folder to upload (README.md, data/, GATES.json); default "
        "<out>/releases/<release-id>/<flavour>, where 'release build' writes",
    )  # fmt: skip
    p.add_argument("--out", type=Path, default=Path("."), help="'release build' output root")
    p.add_argument(
        "--release-dir", type=Path, default=None,
        help=f"where {RELEASE_FILE} lives (default: --dir)",
    )  # fmt: skip
    p.add_argument(
        "--lock", type=Path, default=None,
        help=f"{LOCK_FILE} to hash against {RELEASE_FILE} (default: <release-dir>/{LOCK_FILE} "
        "when present)",
    )  # fmt: skip
    p.add_argument(
        "--go", action="store_true", help=f"upload for real; needs {TOKEN_ENV} in the environment"
    )
    p.add_argument(
        "--allow-frozen", dest="allow_frozen", action="store_true",
        help="let --go touch the frozen v1.0 release (never while the hackathon runs)",
    )  # fmt: skip
    p.set_defaults(func=cmd_release_publish)
