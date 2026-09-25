"""`marinedata captions {template,pilot,check}`.

Not yet wired into the top-level `marinedata` CLI: `cli.py` is shared with WP-8c's
in-flight `configs.py`/`hf_card.py` work, so this brief leaves that one-line
integration (`from .captions.cli import add_captions_subparser` + `add_captions_subparser(sub)`
in `cli.py::main`, mirroring `add_dedup_subparser`) to the integrator. Until then this
module is runnable directly: `python -m marinedata.captions.cli template ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

from .consistency import check_caption
from .facts import build_facts_table
from .schema import load_schema, validate_frame
from .template import build_caption, coverage_report


def _cmd_template(args: argparse.Namespace) -> int:
    metadata = pd.read_parquet(args.metadata)
    benthic = pd.read_parquet(args.benthic) if args.benthic else None
    bleaching = pd.read_parquet(args.bleaching) if args.bleaching else None

    t0 = time.monotonic()
    facts_list = build_facts_table(metadata, benthic_coarse=benthic, bleaching_condition=bleaching)
    rows = [
        {
            "image_sha256": f.image_sha256,
            "caption_template": build_caption(f),
            "caption_facts": f.to_json(),
            "caption_vlm": None,
            "vlm_model": None,
            "vlm_revision": None,
            "prompt_sha256": None,
            "seed": None,
            "label_origin": "derived",
            "consistency_flags": "[]",
            "audit_verdict": None,
        }
        for f in facts_list
    ]
    elapsed = time.monotonic() - t0
    out = pd.DataFrame(rows)
    out["seed"] = out["seed"].astype("Int64")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)

    report = coverage_report(facts_list, min_facts=args.min_facts)
    report["img_per_sec_facts_extraction"] = (
        round(len(facts_list) / elapsed, 1) if elapsed > 0 else None
    )
    report["out"] = str(args.out)
    print(json.dumps(report, indent=2))
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    df = pd.read_parquet(args.in_path)
    flags_col = []
    n_flagged = 0
    for _, row in df.iterrows():
        vlm = row.get("caption_vlm")
        template = row.get("caption_template")
        caption = vlm if pd.notna(vlm) else (template if pd.notna(template) else "")
        facts = json.loads(row["caption_facts"]) if pd.notna(row.get("caption_facts")) else {}
        flags = check_caption(
            caption,
            bleaching_status=facts.get("bleaching_status"),
            benthic_dominant=facts.get("benthic_dominant"),
        )
        if flags:
            n_flagged += 1
        flags_col.append(json.dumps(flags))
    df["consistency_flags"] = flags_col

    schema = load_schema()
    problems = validate_frame(df, schema)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(
        json.dumps(
            {
                "rows": len(df),
                "flagged": n_flagged,
                "pct_flagged": round(100.0 * n_flagged / len(df), 2) if len(df) else 0.0,
                "schema_problems": problems,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 1 if problems else 0


def _cmd_pilot(args: argparse.Namespace) -> int:
    """Stratified pilot runner. Requires a VLM backend (`uv pip install mlx-vlm` or
    `transformers`) — see `vlm.load_backend`. Not exercised by the test suite, which
    injects a fake `VLMBackend` directly against `run_pilot`."""
    from .vlm import load_backend, run_pilot, stratified_sample

    metadata = pd.read_parquet(args.metadata)
    sample = stratified_sample(metadata, n=args.n, seed=args.seed)
    facts_list = build_facts_table(sample)
    facts_by_sha = {f.image_sha256: f for f in facts_list}
    backend = load_backend()

    rows = (
        (sha, str(Path(args.images_root) / f"{sha}.jpg"), facts_by_sha[sha])
        for sha in sample["image_sha256"]
    )
    pilot_rows = run_pilot(rows, backend=backend, seed=args.seed)
    out = pd.DataFrame([r.__dict__ for r in pilot_rows])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(json.dumps({"rows": len(out), "model": backend.model_id, "out": str(args.out)}, indent=2))
    return 0


def add_captions_subparser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("captions", help="WP-13 caption generation and checks")
    sub = p.add_subparsers(dest="captions_cmd", required=True)

    t = sub.add_parser("template", help="Tier A: deterministic captions for all rows")
    t.add_argument("--metadata", required=True)
    t.add_argument("--benthic", default=None)
    t.add_argument("--bleaching", default=None)
    t.add_argument("--out", required=True)
    t.add_argument("--min-facts", dest="min_facts", type=int, default=3)
    t.set_defaults(func=_cmd_template)

    pl = sub.add_parser("pilot", help="Tier B: stratified VLM captioning pilot")
    pl.add_argument("--metadata", required=True)
    pl.add_argument("--images-root", required=True)
    pl.add_argument("--n", type=int, default=500)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--out", required=True)
    pl.set_defaults(func=_cmd_pilot)

    c = sub.add_parser("check", help="Automated consistency check + schema validation")
    c.add_argument("--in", dest="in_path", required=True)
    c.add_argument("--out", required=True)
    c.set_defaults(func=_cmd_check)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="marinedata-captions")
    sub = parser.add_subparsers(dest="cmd", required=True)
    add_captions_subparser(sub)
    # `add_captions_subparser` nests one level ("captions X"); standalone entry skips
    # the outer "captions" hop so `python -m marinedata.captions.cli template` works.
    args = parser.parse_args(["captions", *(argv or sys.argv[1:])])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
