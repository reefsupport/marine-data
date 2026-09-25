"""Command-line interface.

marinedata list --task benthic-segmentation --profile ship-commercial
marinedata show coralscapes
marinedata check --profile ship-commercial
marinedata lineage --task benthic-segmentation --profile ship-commercial -o LINEAGE.json
"""

from __future__ import annotations

import argparse
import sys

from .cli_ingest import _cmd_ingest, add_ingest_subparser  # noqa: F401 — re-exported
from .cli_release import add_release_subparser
from .cli_splitmap import add_splitmap_subparser
from .eval.cli import add_eval_subparser
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
        registry_commit=reg.commit,
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


def _cmd_taxa(args: argparse.Namespace) -> int:
    """Re-resolve every anchored node against WoRMS/OBIS and report drift.

    Network-bound and maintainer-run — never part of loading. Exits non-zero on drift so
    a scheduled job can open an issue.
    """
    from .taxa import check_node, resolve_id

    reg = Registry.load()
    drifts, checked, missing = [], 0, []
    for schema in reg.schemas:
        for node in schema.nodes:
            if node.worms_aphia_id is None:
                if args.show_unanchored:
                    missing.append(f"{schema.id}/{node.id}")
                continue
            checked += 1
            record = resolve_id(node.worms_aphia_id)
            found = check_node(node, record)
            drifts.extend(found)
            mark = "✗" if found else "✓"
            print(f"{mark} {node.id:<24} {node.worms_aphia_id:>8}  {node.worms_scientificname}")

    if missing:
        print(f"\nunanchored ({len(missing)}): {', '.join(missing)}")
    print(f"\nchecked {checked} anchored node(s), {len(drifts)} drift(s)")
    for drift in drifts:
        print(drift.line())
    return 1 if drifts else 0


def _cmd_add(args: argparse.Namespace) -> int:
    """Probe a HuggingFace dataset and print a draft registry entry.

    Never writes into registry/ and never asserts a licence — see probe.py's module
    docstring. Review, correct, and verify the licence primarily before committing it.
    """
    from .fetch import FetchError
    from .probe import probe_huggingface

    try:
        result = probe_huggingface(args.hf_id)
    except FetchError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(result.yaml_draft)
        print(f"wrote draft to {args.output} — review every TODO before committing it")
    else:
        print(result.yaml_draft)
    for warning in result.warnings:
        print(f"⚠ {warning}", file=sys.stderr)
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Per-source completeness: layout verified, licence primary, crosswalk, fetchable.

    No network — everything here comes from the registry alone. Run this first when
    picking up unfinished work; run ``verify``/``labels`` for the parts that need real
    data to check.
    """
    from .verify import doctor, doctor_totals

    reg = Registry.load()
    rows = doctor(reg)
    shown = tuple(r for r in rows if not r.complete) if args.incomplete_only else rows
    print("\n".join(row.line() for row in shown))
    print(f"\n{doctor_totals(rows)}")
    return 0


def _cmd_labels(args: argparse.Namespace) -> int:
    """Audit crosswalk labels against the labels the data actually contains."""
    from .fetch import FetchError, fetch_sample
    from .labelcheck import audit_source, summarise

    reg = Registry.load()
    ids = args.source_id or [s.id for s in reg if s.loader and s.loader.layout != "metadata-only"]
    audits = []
    for source_id in ids:
        try:
            root = fetch_sample(reg.source(source_id), limit=args.limit).root
        except FetchError as exc:
            if args.verbose:
                print(f"· {source_id:<30} unfetchable  {str(exc)[:60]}")
            continue
        audit = audit_source(reg, source_id, root, limit=args.limit * 5)
        audits.append(audit)
        print(audit.report() if audit.unmapped or args.verbose else audit.line())

    if audits:
        print(f"\n{summarise(audits)}")
    drops = [a for a in audits if a.unmapped]
    return 1 if drops and args.strict else 0


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

    add_ingest_subparser(sub)
    add_splitmap_subparser(sub)
    add_release_subparser(sub)
    add_eval_subparser(sub)

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

    p_taxa = sub.add_parser(
        "taxa", help="Re-resolve anchored nodes against WoRMS/OBIS and report drift"
    )
    p_taxa.add_argument(
        "--show-unanchored", action="store_true", help="also list nodes with no AphiaID"
    )
    p_taxa.set_defaults(func=_cmd_taxa)

    p_add = sub.add_parser(
        "add",
        help="Probe a HuggingFace dataset and print a draft registry entry for review",
    )
    p_add.add_argument("hf_id", help="owner/name, or a full huggingface.co/datasets/... URL")
    p_add.add_argument("-o", "--output", help="Write the draft to this path instead of stdout")
    p_add.set_defaults(func=_cmd_add)

    p_doctor = sub.add_parser(
        "doctor",
        help="Per-source completeness — layout, licence, crosswalk, fetchable. No network.",
    )
    p_doctor.add_argument(
        "--incomplete-only", action="store_true", help="Only list sources missing something"
    )
    p_doctor.set_defaults(func=_cmd_doctor)

    p_labels = sub.add_parser(
        "labels", help="Audit crosswalk labels against the labels real data contains"
    )
    p_labels.add_argument("source_id", nargs="*")
    p_labels.add_argument("--limit", type=int, default=50)
    p_labels.add_argument("--strict", action="store_true", help="Exit 1 on any silent drop")
    p_labels.add_argument("-v", "--verbose", action="store_true")
    p_labels.set_defaults(func=_cmd_labels)

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
