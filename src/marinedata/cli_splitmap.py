"""``marinedata splitmap`` sub-command — generate a group-keyed ``SPLIT_MAP.json``.

Split out of :mod:`marinedata.cli` for the same reason as :mod:`marinedata.cli_ingest`:
keep that module under the line cap. ``cli.py`` imports and wires
``add_splitmap_subparser`` from here.

**Input contract.** ``DatasetBuilder`` enumerates samples from a staged ``sources/``
tree via a registry-driven ``roots`` mapping, but nothing today turns an arbitrary
staged tree into that mapping without a task id and a schema — there is no release
driver yet (see the WS-D design doc, §3). Rather than build one as a side effect of
this CLI, ``generate`` takes a flat, already-deduplicated enumeration instead: a TSV or
Parquet file with exactly two columns, ``image_sha256`` and ``split_group`` — one row
per admitted image, however many sources or tasks it appears in. Producing that file
(e.g. from ``metadata.parquet`` across every staged source) is the caller's job.

**Stratification.** An optional third column, ``stratum`` (normally the source id),
plus ``--stratify <label>`` allocates per stratum (:mod:`marinedata.strata`): one row
per (image, stratum), so an image staged by two sources appears twice with the same
``split_group``. Without ``--stratify`` the column is ignored and one pool is used.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from .builder import SELF_SUPERVISED_DEFAULT_RATIOS, SUPERVISED_DEFAULT_RATIOS, SplitName
from .splitmap import SplitMap, load_split_map, resolve_splits, save_split_map
from .strata import DEFAULT_MIN_GROUPS, achieved_by_stratum, small_strata

Row = tuple[str, str, str | None]


def _read_tsv_rows(path: Path) -> Iterator[Row]:
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fields = set(reader.fieldnames or ())
        required = {"image_sha256", "split_group"}
        if not required <= fields:
            raise ValueError(
                f"{path}: expected TSV columns {sorted(required)}, got {sorted(fields)}"
            )
        has_stratum = "stratum" in fields
        for row in reader:
            yield row["image_sha256"], row["split_group"], (row["stratum"] if has_stratum else None)


def _read_parquet_rows(path: Path) -> Iterator[Row]:
    from .ingest_parquet import _require_pyarrow

    pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    names = ["image_sha256", "split_group"]
    if "stratum" in parquet.schema_arrow.names:
        names.append("stratum")
    for batch in parquet.iter_batches(columns=names):
        columns = batch.to_pydict()
        strata = columns.get("stratum") or [None] * batch.num_rows
        yield from zip(columns["image_sha256"], columns["split_group"], strata, strict=True)


def _group_counts(
    path: Path, *, stratified: bool = False
) -> tuple[dict[str, int], dict[str, dict[str, int]] | None]:
    """One count per unique image: a duplicate across sources must not be counted twice.

    Raises if the same image resolves to two different groups — the registry's
    per-source ``split_group`` rule is supposed to make duplicates converge, and a
    disagreement here means that rule (or the input file) is wrong, not something to
    silently pick a winner for.
    """
    rows = _read_parquet_rows(path) if path.suffix == ".parquet" else _read_tsv_rows(path)
    group_of: dict[str, str] = {}
    members: dict[str, set[str]] = {}
    for sha256, group, stratum in rows:
        if not sha256 or not group:
            raise ValueError(f"{path}: row with an empty image_sha256 or split_group")
        prior = group_of.get(sha256)
        if prior is not None and prior != group:
            raise ValueError(
                f"{path}: image {sha256} maps to two different split_group values "
                f"({prior!r} and {group!r}) — split_group must be immutable per image"
            )
        group_of[sha256] = group
        if stratified:
            if not stratum:
                raise ValueError(f"{path}: --stratify needs a non-empty `stratum` on every row")
            members.setdefault(stratum, set()).add(sha256)
    counts = dict(Counter(group_of.values()))
    if not stratified:
        return counts, None
    strata = {name: dict(Counter(group_of[sha] for sha in shas)) for name, shas in members.items()}
    return counts, strata


def _parse_ratios(spec: str) -> dict[SplitName, float]:
    parts = [float(p) for p in spec.split("/") if p]
    if len(parts) == 3:
        names = list(SUPERVISED_DEFAULT_RATIOS)
    elif len(parts) == 2:
        names = list(SELF_SUPERVISED_DEFAULT_RATIOS)
    else:
        raise ValueError(
            f"--ratios must be 2 or 3 '/'-separated numbers (train/val/test or "
            f"train/probe), got {spec!r}"
        )
    total = sum(parts)
    if total <= 0:
        raise ValueError(f"--ratios values must sum to a positive number, got {spec!r}")
    return {name: value / total for name, value in zip(names, parts, strict=True)}


def _parse_preseed(specs: list[str] | None, counts: dict[str, int]) -> dict[str, SplitName]:
    assignments: dict[str, SplitName] = {}
    for spec in specs or []:
        pattern, sep, split = spec.partition("=")
        if not sep or not split:
            raise ValueError(f"--preseed must be '<glob>=<split>', got {spec!r}")
        matched = [key for key in counts if fnmatch.fnmatchcase(key, pattern)]
        if not matched:
            print(f"splitmap generate: --preseed {spec!r} matched no group", file=sys.stderr)
        for key in matched:
            assignments[key] = split
    return assignments


def _cmd_splitmap_generate(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists():
        print(
            f"splitmap generate: {out} already exists — generate always writes a fresh "
            "map; remove or rename it first",
            file=sys.stderr,
        )
        return 1

    counts, strata = _group_counts(Path(args.input), stratified=bool(args.stratify))
    ratios = _parse_ratios(args.ratios)
    preseed = _parse_preseed(args.preseed, counts)

    if preseed:
        save_split_map(
            out,
            SplitMap(
                by="group",
                seed=args.seed,
                ratios=ratios,
                assignments=preseed,
                generated_at=args.now,
                release=args.release,
                stratify=args.stratify or "",
            ),
        )
    resolve_splits(
        out,
        counts,
        ratios,
        seed=args.seed,
        by="group",
        now=args.now,
        release=args.release,
        strata=strata,
        stratify=args.stratify or "",
        min_groups=args.min_groups,
    )

    split_map = load_split_map(out)
    assert split_map is not None  # just written above
    achieved: dict[str, int] = {}
    for key, count in counts.items():
        split = split_map.assignments[key]
        achieved[split] = achieved.get(split, 0) + count
    total = sum(counts.values())

    print(
        f"splitmap generate: release={args.release} groups={len(counts)} "
        f"images={total} seed={args.seed} -> {out}"
    )
    for name, n in sorted(achieved.items()):
        pct = n / total if total else 0.0
        print(f"  {name}: {n} images ({pct:.1%})")
    if strata is not None:
        small = set(small_strata(strata, args.min_groups))
        for name, per in achieved_by_stratum(strata, split_map.assignments).items():
            n = sum(per.values())
            parts = "  ".join(f"{s}={per.get(s, 0)} ({per.get(s, 0) / n:.1%})" for s in ratios)
            note = f"  [train-only: {len(strata[name])} groups < {args.min_groups}]"
            print(f"  stratum {name}: {parts}{note if name in small else ''}")
    return 0


def add_splitmap_subparser(sub: argparse._SubParsersAction) -> None:
    """Wire the ``splitmap`` sub-command onto ``sub`` (called from ``cli.build_parser``)."""
    p_splitmap = sub.add_parser("splitmap", help="Generate and inspect SPLIT_MAP.json")
    splitmap_sub = p_splitmap.add_subparsers(dest="splitmap_command", required=True)

    p_generate = splitmap_sub.add_parser(
        "generate", help="Write a fresh, deterministic group-keyed SPLIT_MAP.json"
    )
    p_generate.add_argument("--release", required=True, help="Release id, for the summary line")
    p_generate.add_argument(
        "--in",
        dest="input",
        required=True,
        help="TSV or Parquet file of (image_sha256, split_group), one row per admitted image",
    )
    p_generate.add_argument("--now", required=True, help="ISO date/time to record as generated_at")
    p_generate.add_argument("--seed", type=int, default=0)
    p_generate.add_argument("--ratios", default="70/15/15", help="'/'-separated, e.g. 70/15/15")
    p_generate.add_argument(
        "--preseed",
        action="append",
        dest="preseed",
        metavar="GLOB=SPLIT",
        help="Pin every group matching GLOB to SPLIT before allocating the rest; repeatable",
    )
    p_generate.add_argument(
        "--stratify",
        metavar="LABEL",
        help="Allocate per the input's `stratum` column (e.g. LABEL=source); recorded on the map",
    )
    p_generate.add_argument(
        "--min-groups",
        type=int,
        default=DEFAULT_MIN_GROUPS,
        help="A stratum with fewer groups than this is train-only (default 3)",
    )
    p_generate.add_argument("--out", required=True, help="Path to write SPLIT_MAP.json to")
    p_generate.set_defaults(func=_cmd_splitmap_generate)
