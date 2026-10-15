"""The CLI and its docs agree (P1.13): every ``marinedata <group> <cmd>`` is named in
README.md or docs/*.md, and no doc names a command that does not exist."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from marinedata.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
DOC_FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
DOCS = "\n".join(p.read_text(encoding="utf-8") for p in DOC_FILES)


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _leaf_commands(parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()):
    subs = _subparsers(parser)
    if not subs:
        yield " ".join(prefix)
        return
    for name, sub in subs.items():
        yield from _leaf_commands(sub, (*prefix, name))


def _flags(command: str) -> set[str]:
    node = build_parser()
    for token in command.split():
        node = _subparsers(node)[token]
    return {o for a in node._actions for o in a.option_strings}


def test_every_command_is_named_in_readme_or_docs():
    missing = [c for c in _leaf_commands(build_parser()) if f"marinedata {c}" not in DOCS]
    assert missing == [], f"undocumented commands (run `make docs`, or describe them): {missing}"


@pytest.mark.parametrize(
    ("command", "flags"),
    [
        ("tasks build-points", ["--metadata-dir"]),
        ("release gate", ["--metadata", "--masks", "--all", "--require-all"]),
        ("release lock", ["--release-id", "--flavour"]),
        ("release publish", ["--go", "--allow-frozen"]),
        ("release card", []),
        ("release croissant", []),
        ("metadata resolve", ["--source", "--out"]),
        ("auth check", []),
    ],
)
def test_phase_a_commands_exist_and_their_flags_are_documented(command, flags):
    assert f"marinedata {command}" in DOCS
    real = _flags(command)
    for flag in flags:
        assert flag in real, f"marinedata {command} has no {flag}"
        assert flag in DOCS, f"{flag} of marinedata {command} is not in README.md or docs/*.md"


def test_the_readme_names_auth_check():
    assert "marinedata auth check" in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script", ["registry_probe", "registry_sync_bucket", "registry_fold_specs"]
)
def test_registry_maintenance_scripts_are_documented(script):
    assert script in DOCS


def _mentions() -> list[tuple[Path, str]]:
    found = []
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"`marinedata ([^`\n]+)`", text):
            found.append((path, match.group(1)))
        for match in re.finditer(r"^\s*(?:\$ )?marinedata ([^\n]+)$", text, re.MULTILINE):
            found.append((path, match.group(1)))
    return found


def _resolves(mention: str) -> bool:
    tokens = mention.split()
    if not tokens or not re.fullmatch(r"[a-z][a-z-]*", tokens[0]):
        return True  # a flag, a placeholder or prose, not a command
    node, top = build_parser(), True
    for token in tokens:
        subs = _subparsers(node)
        if not subs:
            return True
        if token not in subs:
            return not (top or re.fullmatch(r"[a-z][a-z-]*", token))
        node, top = subs[token], False
    return True


def test_no_doc_names_a_command_that_does_not_exist():
    dead = sorted({f"{p.name}: marinedata {m}" for p, m in _mentions() if not _resolves(m)})
    assert dead == [], f"dead command mentions: {dead}"
