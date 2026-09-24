"""Upload a prepared Hub folder in explicit, bounded commits (WS-D S55).

**Dry run by default**: prints the planned commits and exits. A real upload needs BOTH
``--execute`` and ``--confirm-yohan-go`` plus an explicit ``--visibility`` — publishing is
Yohan's call, never a default. (A *private* dataset on the free org plan gets no dataset
viewer: hub-docs ``datasets-adding.md``, "Dataset Viewer".)

Why explicit ``create_commit`` batches rather than ``upload_folder``: the plan is fixed and
printable before anything moves — each commit has at most :data:`MAX_FILES_PER_COMMIT`
files (Hub recommendation: <100 files per commit, hub-docs ``storage-limits.md``) and at
most :data:`MAX_BYTES_PER_COMMIT` bytes, so an interrupted run loses at most one batch.
Resumable: files already on the Hub with the same LFS sha256 are skipped on re-run.
``README.md`` goes in the **last** commit so the card never names shards that are not
there yet.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

MAX_FILES_PER_COMMIT = 99
MAX_BYTES_PER_COMMIT = 4 * 1000**3
LAST = ("README.md",)


@dataclass(frozen=True)
class Commit:
    index: int
    total: int
    files: tuple[str, ...]
    nbytes: int

    @property
    def message(self) -> str:
        return f"v1 export part {self.index + 1}/{self.total}"


def repo_files(folder: Path) -> list[str]:
    """Every file to upload, repo-relative, sorted; hidden files and ``*.tmp`` skipped."""
    out = []
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        if path.is_file() and not rel.startswith(".") and not rel.endswith(".tmp"):
            out.append(rel)
    return out


def plan_commits(
    folder: Path,
    files: list[str] | None = None,
    max_files: int = MAX_FILES_PER_COMMIT,
    max_bytes: int = MAX_BYTES_PER_COMMIT,
) -> list[Commit]:
    if not 0 < max_files < 100:
        raise ValueError("max_files must be 1..99 (Hub: <100 files per commit)")
    files = repo_files(folder) if files is None else files
    ordered = [f for f in files if f not in LAST] + [f for f in files if f in LAST]
    batches: list[list[str]] = []
    sizes: list[int] = []
    for rel in ordered:
        size = (folder / rel).stat().st_size
        if not batches or len(batches[-1]) >= max_files or sizes[-1] + size > max_bytes:
            batches.append([])
            sizes.append(0)
        batches[-1].append(rel)
        sizes[-1] += size
    return [
        Commit(i, len(batches), tuple(b), s)
        for i, (b, s) in enumerate(zip(batches, sizes, strict=True))
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _execute(folder: Path, repo_id: str, commits: list[Commit], private: bool) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    remote = {
        entry.path: getattr(getattr(entry, "lfs", None), "sha256", None)
        for entry in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True)
    }
    for commit in commits:
        todo = [f for f in commit.files if f not in remote or remote[f] != _sha256(folder / f)]
        if not todo:
            print(f"skip {commit.message}: already on the Hub")
            continue
        ops = [CommitOperationAdd(path_in_repo=f, path_or_fileobj=str(folder / f)) for f in todo]
        api.create_commit(
            repo_id, operations=ops, commit_message=commit.message, repo_type="dataset"
        )
        print(f"done {commit.message}: {len(todo)} files")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m marinedata.hf_upload")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--max-files", type=int, default=MAX_FILES_PER_COMMIT)
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES_PER_COMMIT)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-yohan-go", action="store_true")
    parser.add_argument("--visibility", choices=("public", "private"))
    args = parser.parse_args(argv)

    commits = plan_commits(args.folder, max_files=args.max_files, max_bytes=args.max_bytes)
    files = sum(len(c.files) for c in commits)
    total = sum(c.nbytes for c in commits)
    print(f"repo {args.repo_id} (dataset): {files} files, {total} bytes, {len(commits)} commits")
    for c in commits:
        print(
            f"  {c.message}: {len(c.files)} files, {c.nbytes} bytes, "
            f"first {c.files[0]}, last {c.files[-1]}"
        )
    if not args.execute:
        print("DRY RUN — nothing uploaded. Real run: --execute --confirm-yohan-go --visibility …")
        return 0
    if not args.confirm_yohan_go or args.visibility is None:
        print("REFUSED: --execute needs --confirm-yohan-go and an explicit --visibility")
        return 2
    _execute(args.folder, args.repo_id, commits, private=args.visibility == "private")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
