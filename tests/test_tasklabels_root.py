"""INT-core3c: the v2 task-layer configs never depend on the caller's cwd.

``build_release`` used to read ``Path(".")/_tasklabels``; ``_read_parquet`` returns ``[]``
for a missing file, so a build from the repo root silently produced empty configs.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from marinedata.cli import build_parser
from marinedata.registry import Registry
from marinedata.release import build_release
from marinedata.task_layers.configs import build_all_configs, resolve_tasklabels_root

pytest.importorskip("pyarrow")

REPO = Path(__file__).resolve().parents[1]


def _summary(results: dict) -> dict[str, tuple[int, int]]:
    return {cid: (len(r.rows), r.n_images) for cid, r in results.items()}


def test_default_root_is_the_repo_data_dir_from_any_cwd(monkeypatch, tmp_path) -> None:
    registry = Registry.load()
    monkeypatch.chdir(REPO)
    from_repo = resolve_tasklabels_root(registry)
    monkeypatch.chdir(tmp_path)
    assert resolve_tasklabels_root(registry) == from_repo == REPO / "data"


def test_builder_gives_the_same_non_empty_configs_from_any_cwd(monkeypatch, tmp_path) -> None:
    registry = Registry.load()
    monkeypatch.chdir(REPO)
    at_repo = _summary(build_all_configs(registry, resolve_tasklabels_root(registry)))
    # The old behaviour, pinned: Path(".") from the repo root finds nothing.
    old = _summary(build_all_configs(registry, Path(".")))
    assert all(n_rows == 0 for n_rows, _ in old.values())
    monkeypatch.chdir(tmp_path)
    elsewhere = _summary(build_all_configs(registry, resolve_tasklabels_root(registry)))
    assert at_repo == elsewhere
    non_empty = {cid for cid, (n_rows, _) in at_repo.items() if n_rows}
    assert {"points", "semseg", "bleaching"} <= non_empty


def test_explicit_root_wins_and_accepts_the_tasklabels_dir_itself(tmp_path) -> None:
    (tmp_path / "_tasklabels").mkdir()
    assert resolve_tasklabels_root(None, tmp_path) == tmp_path.resolve()
    assert resolve_tasklabels_root(None, tmp_path / "_tasklabels") == tmp_path.resolve()


def test_missing_root_fails_closed(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="--tasklabels-root"):
        resolve_tasklabels_root(None, tmp_path)


def test_cli_flag_reaches_build_release() -> None:
    args = build_parser().parse_args(
        [
            "release",
            "build",
            "--release",
            "v2",
            "--split-map",
            "s.json",
            "--out",
            "o",
            "--tasklabels-root",
            "/x/data",
        ]
    )
    assert args.tasklabels_root == "/x/data"
    assert "tasklabels_root" in inspect.signature(build_release).parameters
    source = inspect.getsource(build_release)
    assert 'Path(".")' not in source
