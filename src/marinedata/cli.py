"""Command-line interface.

marinedata list --task benthic-segmentation --profile ship-commercial
marinedata show coralscapes
marinedata check --profile ship-commercial
marinedata lineage --task benthic-segmentation --profile ship-commercial -o LINEAGE.json
"""

from __future__ import annotations

import argparse
import sys

from .gate import evaluate
from .lineage import build_lineage
from .query import find
from .registry import Registry, RegistryError


def _cmd_list(args: argparse.Namespace) -> int:
    result = find(
        profile=args.profile,
        task=args.task,
        region=args.region,
        annotation=args.annotation,
        modality=args.modality,
        legal_opinion_ref=args.legal_opinion_ref,
    )
    print(result.summary())
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    src = Registry.load().source(args.source_id)
    print(f"{src.name}  ({src.id} @ {src.version})")
    print(f"  licence      {src.licence.id}  [{src.licence.tier.value}]")
    print(f"  basis        {src.legal_basis.value}   provenance: {src.provenance.value}")
    print(f"  verified     {src.verification.verified_on} via {src.verification.method}")
    print(f"               {src.verification.verified_by}")
    if src.verification.disputed:
        print(f"  ⚠ DISPUTED   {' '.join((src.verification.dispute_note or '').split())}")
    print(f"  items        {src.items or '—'}  {src.items_note or ''}")
    print(f"  capabilities {', '.join(c.value for c in src.capabilities)}")
    print(f"  regions      {', '.join(r.value for r in src.coverage.regions)}")
    print(f"  access       {src.access.method.value}  {src.access.uri or ''}")
    if src.domain_shift:
        ds = src.domain_shift
        if ds.missing_classes:
            print(f"  ⚠ missing    {', '.join(ds.missing_classes)}")
        for drop in ds.known_drops:
            print(f"  ⚠ drop       {drop.delta:+g} {drop.metric} → {drop.to_region.value}")
        for caveat in ds.caveats:
            print(f"  · {' '.join(caveat.split())}")
    if src.notes:
        print(f"  notes        {' '.join(src.notes.split())}")
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Evaluate every source against a profile. Exit 1 if any would be denied."""
    reg = Registry.load()
    profile = reg.profile(args.profile)
    denied = 0
    for src in reg:
        decision = evaluate(src, profile, legal_opinion_ref=args.legal_opinion_ref)
        mark = "✓" if decision.allowed else "✗"
        print(f"{mark} {src.id}")
        if not decision.allowed:
            denied += 1
            print(f"    {' '.join(decision.reason.split())}")
    print(f"\n{len(reg) - denied} permitted, {denied} denied under '{profile.id}'")
    return 1 if denied and args.strict else 0


def _cmd_lineage(args: argparse.Namespace) -> int:
    reg = Registry.load()
    result = find(
        profile=args.profile,
        task=args.task,
        registry=reg,
        legal_opinion_ref=args.legal_opinion_ref,
    )
    lineage = build_lineage(
        list(result),
        reg.profile(args.profile),
        excluded=list(result.excluded),
        legal_opinion_ref=args.legal_opinion_ref,
    )
    payload = lineage.to_json()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(payload)
        print(f"wrote {args.output}")
        print(lineage.attribution_text())
    else:
        print(payload)
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from .fetch import FetchError, fetch_sample

    reg = Registry.load()
    source = reg.source(args.source_id)
    try:
        result = fetch_sample(source, limit=args.limit, force=args.force)
    except FetchError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1
    print(f"{result.items} items → {result.root}")
    print("Bounded verification sample — NOT the full dataset.")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Fetch a small sample per source and confirm the declared layout reads it."""
    from .verify import summarise, unverified, verify_all

    reg = Registry.load()
    if args.unverified_only:
        pending = unverified(reg)
        for source in pending:
            print(f"· {source.id:<34} layout '{source.loader.layout}' never checked")
        print(f"\n{len(pending)} source(s) with unverified layouts")
        return 0

    results = verify_all(reg, limit=args.limit, only=args.source_id or None)
    for result in results:
        print(result.line())
    print(f"\n{summarise(results)}")
    failed = [r for r in results if r.status == "failed"]
    return 1 if failed and args.strict else 0


def _cmd_mirror(args: argparse.Namespace) -> int:
    """Show what a mirror operation would copy, and what it would refuse."""
    from .mirror import MirrorTarget, attribution_document, plan_mirror

    reg = Registry.load()
    sources = [reg.source(s) for s in args.source_id] if args.source_id else list(reg)
    plan = plan_mirror(sources, MirrorTarget(args.target))
    print(plan.summary())
    if args.attribution:
        with open(args.attribution, "w", encoding="utf-8") as fh:
            fh.write(attribution_document(plan, sources))
        print(f"\nwrote {args.attribution}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="marinedata", description="Licence-aware marine dataset registry."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--profile", required=True, help="Release profile (see profiles.yaml)")
        p.add_argument("--legal-opinion-ref", dest="legal_opinion_ref", default=None)

    p_list = sub.add_parser("list", help="List sources matching facets under a profile")
    add_common(p_list)
    p_list.add_argument("--task")
    p_list.add_argument("--region", action="append")
    p_list.add_argument("--annotation")
    p_list.add_argument("--modality")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show one source in detail")
    p_show.add_argument("source_id")
    p_show.set_defaults(func=_cmd_show)

    p_check = sub.add_parser("check", help="Evaluate every source against a profile")
    add_common(p_check)
    p_check.add_argument("--strict", action="store_true", help="Exit 1 if anything is denied")
    p_check.set_defaults(func=_cmd_check)

    p_lin = sub.add_parser("lineage", help="Emit a LINEAGE.json for a build")
    add_common(p_lin)
    p_lin.add_argument("--task")
    p_lin.add_argument("-o", "--output")
    p_lin.set_defaults(func=_cmd_lineage)

    p_fetch = sub.add_parser("fetch", help="Fetch a bounded verification sample")
    p_fetch.add_argument("source_id")
    p_fetch.add_argument("--limit", type=int, default=100)
    p_fetch.add_argument("--force", action="store_true")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_verify = sub.add_parser("verify", help="Check declared layouts against real fetched samples")
    p_verify.add_argument("source_id", nargs="*")
    p_verify.add_argument("--limit", type=int, default=100)
    p_verify.add_argument("--strict", action="store_true", help="Exit 1 on any failure")
    p_verify.add_argument(
        "--unverified-only",
        action="store_true",
        help="List sources whose layout has never been checked against real data",
    )
    p_verify.set_defaults(func=_cmd_verify)

    p_mirror = sub.add_parser(
        "mirror", help="Plan a copy into our own storage; reports what is refused and why"
    )
    p_mirror.add_argument("source_id", nargs="*")
    p_mirror.add_argument(
        "--target",
        required=True,
        choices=["private-cache", "public-mirror", "training-shard"],
        help="Where the copy would live. This is the whole legal question.",
    )
    p_mirror.add_argument("--attribution", help="Write ATTRIBUTION.md to this path")
    p_mirror.set_defaults(func=_cmd_mirror)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RegistryError as exc:
        print(f"registry error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"invalid argument: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
