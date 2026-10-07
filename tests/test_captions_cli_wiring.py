"""INT-core3b: `marinedata captions` is wired into the main CLI, and the release `--v2`
preset never turns captions on (captions are a separate step; the VLM is never run here)."""

from __future__ import annotations

import inspect

from marinedata.captions import cli as captions_cli
from marinedata.cli import build_parser
from marinedata.release import build_release


def test_captions_subcommands_reach_the_captions_handlers() -> None:
    parser = build_parser()
    args = parser.parse_args(["captions", "template", "--metadata", "m.parquet", "--out", "o"])
    assert args.func is captions_cli._cmd_template
    args = parser.parse_args(["captions", "check", "--in", "c.parquet", "--out", "o"])
    assert args.func is captions_cli._cmd_check


def test_v2_preset_never_enables_captions() -> None:
    args = build_parser().parse_args(
        ["release", "build", "--release", "v2", "--split-map", "s.json", "--out", "o", "--v2"]
    )
    assert args.v2 is True
    assert not any("caption" in name for name in vars(args))
    assert not any("caption" in name for name in inspect.signature(build_release).parameters)
