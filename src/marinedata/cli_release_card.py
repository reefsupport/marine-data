"""``marinedata release card`` and ``marinedata release croissant`` (offline, build-dir only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .release_card import DEFAULT_REPO


def add_card_commands(release_sub: argparse._SubParsersAction) -> None:
    card = release_sub.add_parser(
        "card",
        help="Generate the Hugging Face dataset card (README.md) of a built release",
        description=(
            "Render README.md from the parquet shards of BUILD_DIR (features, split counts, "
            "bytes), the source registry, and RELEASE.json / GATES.json / release.lock.yaml / "
            "CHANGELOG.md when present. Reads files only: no network, no Hub access."
        ),
    )
    card.add_argument(
        "build_dir", nargs="?", type=Path, help="Folder holding data/<config>/*.parquet"
    )
    card.add_argument("--out", type=Path, help="Output file (default <build-dir>/README.md)")
    card.add_argument(
        "--template", type=Path, help="string.Template card (default docs/card-template.md.tmpl)"
    )
    card.add_argument(
        "--release-dir", type=Path, help="Folder with RELEASE.json (default: found under releases/)"
    )
    card.add_argument(
        "--facts",
        type=Path,
        help="Facts JSON instead of reading BUILD_DIR (tests/data/v1.0-card-facts.json)",
    )
    card.add_argument("--citation", type=Path, default=Path("CITATION.cff"))
    card.add_argument("--repo-id", default=DEFAULT_REPO)
    card.add_argument("--pretty-name")
    card.add_argument("--flavour", help="open | nc (default: from the release folder name)")
    card.set_defaults(func=_cmd_card)

    cro = release_sub.add_parser(
        "croissant",
        help="Write Croissant 1.0 JSON-LD for a built release",
        description=(
            "Distribution = every parquet shard with sha256 and size; one recordSet per config, "
            "typed from the parquet schema. Validated with mlcroissant when it is installed."
        ),
    )
    cro.add_argument("build_dir", type=Path)
    cro.add_argument("--out", type=Path, help="Output file (default <build-dir>/croissant.json)")
    cro.add_argument("--release-dir", type=Path)
    cro.add_argument("--citation", type=Path, default=Path("CITATION.cff"))
    cro.add_argument("--repo-id", default=DEFAULT_REPO)
    cro.add_argument("--name")
    cro.add_argument("--date-published", help="YYYY-MM-DD (default: CITATION.cff date-released)")
    cro.add_argument("--no-validate", action="store_true")
    cro.set_defaults(func=_cmd_croissant)


def _cmd_card(args: argparse.Namespace) -> int:
    from .release_card import run_card

    if not args.build_dir and not args.facts:
        raise SystemExit("release card: give BUILD_DIR or --facts")
    return run_card(args)


def _cmd_croissant(args: argparse.Namespace) -> int:
    from .release_card import load_context
    from .release_croissant import build_croissant, validate

    ctx = load_context(
        args.build_dir, release_dir=args.release_dir, citation=args.citation, repo_id=args.repo_id
    )
    doc = build_croissant(args.build_dir, ctx, date_published=args.date_published, name=args.name)
    out = args.out or Path(args.build_dir) / "croissant.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    status, detail = ("skipped", "--no-validate") if args.no_validate else validate(doc)
    print(f"wrote {out}; validation: {status}" + (f" ({detail})" if detail else ""))
    return 1 if status == "invalid" else 0
