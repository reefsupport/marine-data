"""Write docs/cli.md: one row per ``marinedata`` command, from the argparse tree (P1.13).

``make docs`` runs it, ``make docs-check`` fails when the committed file differs. The text of
every row is the command's own ``help=``, so a new command shows up in the docs the day it is
registered."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from marinedata.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def walk(parser: argparse.ArgumentParser, prefix: list[str]):
    """Yield ``(path, help, subparser)`` for every command, groups before their members."""
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        helps = {c.dest: c.help for c in action._choices_actions}
        for name, sub in action.choices.items():
            yield [*prefix, name], helps.get(name) or sub.description or "", sub
            yield from walk(sub, [*prefix, name])


def is_group(parser: argparse.ArgumentParser) -> bool:
    return any(isinstance(a, argparse._SubParsersAction) for a in parser._actions)


def render() -> str:
    rows = []
    for path, text, sub in sorted(walk(build_parser(), []), key=lambda t: t[0]):
        flags = sorted(
            {o for a in sub._actions for o in a.option_strings if o.startswith("--")} - {"--help"}
        )
        kind = "group" if is_group(sub) else "command"
        shown = ", ".join(f"`{f}`" for f in flags) if flags else ""
        text = " ".join(str(text).split()).replace("|", "\\|")
        rows.append(f"| `marinedata {' '.join(path)}` | {kind} | {text} | {shown} |")
    head = (
        "# `marinedata` command reference\n\n"
        "Generated from the argparse tree by `scripts/gen_cli_docs.py` "
        "(run `make docs`); do not edit by "
        "hand. `marinedata <command> --help` is the full reference for one command.\n\n"
        "`release gate --all --require-all` and `release publish` both refuse a build whose "
        "privacy, card or croissant gate is absent, skipped or failed.\n\n"
        "| Command | Kind | What it does | Options |\n|---|---|---|---|\n"
    )
    return head + "\n".join(rows) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "docs", help="directory to write cli.md into"
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "cli.md"
    target.write_text(render(), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
