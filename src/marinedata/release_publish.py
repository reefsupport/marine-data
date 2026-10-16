"""``marinedata release publish``: the last step of a release, behind its guards (Phase A, P1.7).

Without ``--go`` it prints the plan and uploads nothing: the target repo, the configs from
RELEASE.json, the file count and bytes, the card, the lock sha and the tag. It refuses (exit 1,
only the reasons are printed) when RELEASE.json, its lock sha, the card or GATES.json is missing,
when the build is not the flavour asked for, or when a gate required for publishing did not pass.

``--go`` adds three guards before anything leaves the machine: ``HF_TOKEN`` in the environment,
the frozen v1.0 release stays untouched unless ``--allow-frozen``, and the tag must not exist on
the repo yet. It then uploads :data:`UPLOAD_PATTERNS` only, as the parts of
:mod:`marinedata.release_publish_parts` (one per ``data/<config>``, then ``provenance``, then the
card; ``--part`` runs one per call), and tags the commit the card upload made, never the head of
the branch. It never creates a repo or changes its visibility: both exist already and their
visibility is set by hand.

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
from .gates import GATES_FILE, REQUIRED, publish_verdict
from .release_lock import LOCK_FILE, lock_sha256
from .release_provenance import stage_provenance

RELEASE_FILE = "RELEASE.json"
CARD_FILE = "README.md"
TOKEN_ENV = "HF_TOKEN"
SUMMARY_KEY = "summary"
DEFAULT_REQUIRED_GATES = REQUIRED
FROZEN_VERSION = (1,)
"""The v1.0 release (``1``, ``1.0``, ``v1.0``, ``v1.0.0``): live on the Hub and frozen."""
PROVENANCE_DIR = "provenance"
PROVENANCE_FILES = (RELEASE_FILE, GATES_FILE, LOCK_FILE)
DATA_PATTERN = "data/**/*.parquet"
PROVENANCE_PATTERNS = (
    "croissant.json",
    f"{PROVENANCE_DIR}/*",
    "LICENSE",
    "NOTICE",
    "CITATION.cff",
    "assets/*",
    "docs/**",
)
"""What the ``provenance`` part sends: croissant.json, the three provenance files (copied to
``provenance/`` at upload time), the export's LICENSE, NOTICE and CITATION.cff, and the card's
``assets/`` and ``docs/``: everything the card links to."""
UPLOAD_PATTERNS = (DATA_PATTERN, CARD_FILE, *PROVENANCE_PATTERNS)
"""The whole upload set (``allow_patterns``). Nothing else is sent. docs/release-runbook.md lists
it between the ``upload-patterns`` markers and a test keeps the two equal."""
PLAN_PATTERNS = tuple(p for p in UPLOAD_PATTERNS if p != f"{PROVENANCE_DIR}/*")
"""What exists in the release folder before the upload; the provenance copies are made by it."""

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

    release_dir: Path | None = None
    lock_file: Path | None = None


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


def gate_verdict(directory: Path) -> tuple[tuple[str, ...], list[str]]:
    """The gates module's verdict for ``<directory>/GATES.json`` (see ``publish_verdict``)."""
    return publish_verdict(directory)


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
    """What the plan counts: data parquet, the card and croissant.json under ``folder``."""
    out = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(folder).as_posix()
        if any(fnmatch.fnmatchcase(rel, pattern) for pattern in PLAN_PATTERNS):
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
        release_dir=release_dir,
        lock_file=lock_file,
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
        f"  upload set  : {', '.join(PLAN_PATTERNS)}, "
        f"{PROVENANCE_DIR}/ ({', '.join(PROVENANCE_FILES)})",
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


def _stage_provenance(plan: PublishPlan) -> None:
    """Write ``<folder>/provenance/`` (upload time only, so a dry run writes nothing): RELEASE.json
    as its public-safe projection, GATES.json and the lock as they are."""
    base = plan.release_dir or plan.folder
    stage_provenance(
        plan.folder / PROVENANCE_DIR,
        release=base / RELEASE_FILE,
        gates=plan.folder / GATES_FILE,
        lock=plan.lock_file or base / LOCK_FILE,
    )


def _refuse(problems: list[str]) -> int:
    for line in problems:
        print(f"release publish: refused: {line}", file=sys.stderr)
    return 1


def cmd_release_publish(args: argparse.Namespace) -> int:
    from . import release_publish_parts as parts

    token = os.environ.get(TOKEN_ENV, "")
    folder = args.folder or args.out / "releases" / args.release_id / args.flavour
    if args.list_parts:
        if not parts.data_configs(folder):
            return _refuse([f"no parquet shard under {folder}/data/<config>/"])
        print("\n".join(parts.all_parts(folder)))
        return 0
    if args.go:
        guards = go_problems(args.release_id, allow_frozen=args.allow_frozen, token=token)
        if guards:
            return _refuse(guards)
    try:
        plan = build_plan(args.release_id, args.flavour, folder, args.release_dir, args.lock)
        selected = parts.select_parts(plan, args.part)
        delta = parts.v1_delta(plan, args.v1_dir) if args.v1_dir else None
    except PublishRefused as exc:
        code = _refuse(exc.problems)
        for line in parts.gate_lines(folder, only_if_run=True):
            print(line, file=sys.stderr)
        return code
    print(render_plan(plan, go=args.go))
    print(parts.render_parts_plan(plan, selected, delta=delta))
    if not args.go:
        return 0
    try:
        done = parts.publish(plan, token, selected)
    except PublishRefused as exc:
        return _refuse(exc.problems)
    except Exception as exc:  # network and Hub errors: say which, never echo the token
        print(f"release publish: upload failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    tagged = f", tagged {plan.tag}" if parts.PART_TAG in done else ""
    print(f"release publish: {plan.repo_id}: done {', '.join(done)}{tagged}")
    return 0


def add_publish_parser(release_sub: argparse._SubParsersAction) -> None:
    """Wire ``release publish`` onto the ``release`` sub-parser (called from ``cli_release``)."""
    p = release_sub.add_parser(
        "publish",
        help="Upload a gated release to its Hub repo and tag it (dry run unless --go), "
        "all at once or one --part per call. "
        "Fail-closed: refuses unless GATES.json passes all eight gates",
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
        "--part", default=None, metavar="PART",
        help="upload one part only: data:<config> (see --list-parts), provenance (croissant.json, "
        "provenance/, LICENSE, NOTICE, CITATION.cff, assets/, docs/), card (README.md, last) or "
        "tag (the tag, on the sha the card upload made). Default: every part, in that order",
    )  # fmt: skip
    p.add_argument(
        "--list-parts", dest="list_parts", action="store_true",
        help="print the part names of --dir, one per line, and exit (no gates, no network)",
    )  # fmt: skip
    p.add_argument(
        "--v1-dir", dest="v1_dir", type=Path, default=None, metavar="DIR",
        help="a checkout of the v1.0 repo: the plan lists the paths this publish adds and replaces",
    )  # fmt: skip
    p.add_argument(
        "--go", action="store_true", help=f"upload for real; needs {TOKEN_ENV} in the environment"
    )
    p.add_argument(
        "--allow-frozen", dest="allow_frozen", action="store_true",
        help="let --go touch the frozen v1.0 release (never while the hackathon runs)",
    )  # fmt: skip
    p.set_defaults(func=cmd_release_publish)
