"""``marinedata dedup`` — run, eval, bench, audit, report and the split-leak gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .dedup.confirm import ConfirmRules


def _items(args: argparse.Namespace) -> list:  # type: ignore[type-arg]
    from .dedup.corpus import iter_hf_images, iter_staged

    items = []
    for root in args.hf_images or []:
        items += list(iter_hf_images(root))
    for spec in args.staged or []:
        label, _, root = spec.partition("=")
        items += list(iter_staged(root, label))
    return items


def _cmd_run(args: argparse.Namespace) -> int:
    from .dedup.corpus import dump_json, main_log, run_corpus, write_outputs

    items = _items(args)
    main_log(f"items {len(items)}")
    rules = ConfirmRules()
    res = run_corpus(
        items,
        workers=args.workers,
        rules=rules,
        weights=args.weights,
        cache_path=args.cache,
        embed_knn=not args.no_embed_knn,
        log=main_log,
    )
    names = ["pixel", "hash"] + (["embed"] if not args.no_embed_knn else [])
    overlap = write_outputs(res, args.out, names)
    ok = res.pairs["confirmed"]
    summary = {
        "records": len(res.items),
        "unique_sha256": len(res.uniq_shas),
        "decode_errors": len(res.errors),
        "errors_sample": res.errors[:10],
        "rules": rules.record(),
        "candidates": len(ok),
        "confirmed": int(ok.sum()),
        "dup_clusters_multi": sum(1 for c in _sizes(res.clusters).values() if c > 1),
        "max_cluster": max(_sizes(res.clusters).values()),
        "overlap": overlap,
        "lowtex_images": int(res.table["lowtex"].sum()),
        "kinds": {
            k: int((res.pairs["kind"][ok] == k).sum())
            for k in ("exact", "pixel", "copy", "agree", "lowtex")
        },
    }
    dump_json(summary, args.out / "summary.json")
    main_log(f"done -> {args.out}")
    return 0


def _sizes(clusters: dict[str, str]) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(clusters.values()))


def _cmd_eval(args: argparse.Namespace) -> int:
    from .dedup.corpus import dump_json, main_log
    from .dedup.evaluate import lowtex_before_after, synthetic_eval

    out = {"lowtex": lowtex_before_after(args.out)}
    main_log(f"lowtex {out['lowtex']}")
    out["synthetic"] = synthetic_eval(
        _items(args),
        args.out,
        args.weights,
        args.cache,
        ConfirmRules(),
        per_family=args.per_family,
        log=main_log,
    )
    dump_json(out, args.out / "eval.json")
    main_log(json.dumps(out["synthetic"], default=str)[:1500])
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from .dedup.evaluate import scaling_bench

    print(json.dumps(scaling_bench(args.n), indent=2))
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .dedup.audit import render, sample_pairs, score

    if args.score:
        print(json.dumps(score(Path(args.score)), indent=2))
        return 0
    tsv = render(sample_pairs(args.out, args.n), _items(args), args.out / "audit")
    print(tsv)
    return 0


def _cmd_gate(args: argparse.Namespace) -> int:
    from .dedup.groups import gate_release, write_gate_report

    result = gate_release(args.release_dir, args.groups, allow_ungrouped=args.allow_ungrouped)
    if args.write:
        write_gate_report(result, Path(args.release_dir) / "DEDUP_GATE.json")
    d = result.as_dict()
    print(
        f"dedup gate: {'PASS' if result.ok else 'FAIL'} rows={d['rows']} images={d['images']} "
        f"groups={d['groups']} spanning={d['spanning_groups']} "
        f"upstream_test_in_train={d['upstream_test_in_train']} ungrouped={d['ungrouped']}",
        file=sys.stdout if result.ok else sys.stderr,
    )
    return 0 if result.ok else 1


def add_dedup_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("dedup", help="Cross-source dedup v2 and the split-leak gate")
    dsub = p.add_subparsers(dest="dedup_command", required=True)

    def inputs(q: argparse.ArgumentParser) -> None:
        q.add_argument("--hf-images", action="append", help="HF images-config dir (repeatable)")
        q.add_argument("--staged", action="append", help="LABEL=staged tree dir (repeatable)")
        q.add_argument("--out", type=Path, required=True, help="Output dir for the parquet files")
        q.add_argument("--weights", type=Path, help="SSCD TorchScript weights")
        q.add_argument("--cache", type=Path, help="Embedding cache (SQLite)")

    q = dsub.add_parser("run", help="Hash, embed, dedup and group a corpus")
    inputs(q)
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--no-embed-knn", action="store_true", help="Embed hash candidates only")
    q.set_defaults(func=_cmd_run)
    q = dsub.add_parser("eval", help="Synthetic recall + v1 low-texture replay")
    inputs(q)
    q.add_argument("--per-family", type=int, default=400)
    q.set_defaults(func=_cmd_eval)
    q = dsub.add_parser("bench", help="MIH scaling benchmark")
    q.add_argument("--n", type=int, default=1_000_000)
    q.set_defaults(func=_cmd_bench)
    q = dsub.add_parser("audit", help="Stratified precision-audit sheets, or --score a filled TSV")
    inputs(q)
    q.add_argument("--n", type=int, default=120)
    q.add_argument("--score", help="Score a filled audit.tsv")
    q.set_defaults(func=_cmd_audit)
    q = dsub.add_parser(
        "gate", help="Fail if a split group spans splits or upstream test is in train"
    )
    q.add_argument("release_dir", type=Path)
    q.add_argument("--groups", type=Path, required=True, help="groups.parquet from 'dedup run'")
    q.add_argument("--allow-ungrouped", action="store_true")
    q.add_argument(
        "--write", action="store_true", help="Write DEDUP_GATE.json into the release dir"
    )
    q.set_defaults(func=_cmd_gate)
