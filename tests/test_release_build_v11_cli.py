"""``release build --v11``: the split-map refusal and the ``--allow-unsplit`` plumbing (PR-B2)."""

from __future__ import annotations

from pathlib import Path

import pytest

import marinedata.cli_release as cli_release
from marinedata.cli import main
from marinedata.release_lock import unlocked_for_tests
from marinedata.splitmap import SplitMap, save_split_map
from marinedata.task_layers.v11_tables import UnsplitImagesError


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    argv = ["release", "build", "--release", "t1", "--split-map", str(tmp_path / "SPLIT_MAP.json")]
    return [*argv, "--out", str(tmp_path / "out"), "--flavour", "open", "--tasks", "v2", *extra]


def test_v11_refuses_generate_split_map_and_names_the_release_split_map_path(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(_argv(tmp_path, "--v11", "--generate-split-map", "--no-near-dup")) == 1
    err = capsys.readouterr().err
    assert "release split-map --inherit <PRIOR_SPLIT_MAP> --lock <lock>" in err
    assert "--out <SPLIT_MAP_FILE>" in err
    assert not (tmp_path / "SPLIT_MAP.json").exists() and not (tmp_path / "out").exists()


def test_allow_unsplit_needs_v11(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(_argv(tmp_path, "--allow-unsplit")) == 1
    assert "--allow-unsplit needs --v11" in capsys.readouterr().err


@pytest.mark.parametrize("allow", [False, True])
def test_allow_unsplit_reaches_build_release_and_the_refusal_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow: bool,
) -> None:
    ratios = {"train": 0.70, "val": 0.15, "test": 0.15}
    now = "2026-10-15T00:00:00+00:00"
    save_split_map(
        tmp_path / "SPLIT_MAP.json",
        SplitMap(
            by="group", seed=0, ratios=ratios, assignments={"g": "train"}, generated_at=now,
            release="t1", stratify="source",
        ),
    )  # fmt: skip
    seen: list[tuple[bool, bool]] = []

    def fake_build(registry, **kwargs):  # type: ignore[no-untyped-def]
        seen.append((kwargs["v11"], kwargs["allow_unsplit"]))
        raise UnsplitImagesError("2 admitted image(s) have no split in the split map")

    monkeypatch.setattr(cli_release, "build_release", fake_build)
    extra = ["--v11", "--local-only", *(["--allow-unsplit"] if allow else [])]
    with unlocked_for_tests():
        assert main(_argv(tmp_path, *extra)) == 1
    assert seen == [(True, allow)]
    assert "release build: 2 admitted image(s) have no split" in capsys.readouterr().err
