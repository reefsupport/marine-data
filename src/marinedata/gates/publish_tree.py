"""G10 publish tree: the tree the Hub holds after this publish is the one the card declares.

``release publish`` uploads the export folder by path and never deletes, so the tree after a
publish is the export's files plus every prior file the export does not overwrite. This gate
builds that tree from ``--v1-dir`` (the prior tree: the Hub repo as it is before the publish) and
the export, offline, and fails on

* an ``images`` export shard that takes the path of a prior shard (``collisions``; an additive
  export names its shards after the release, so there must be none);
* an ``image_sha256`` that a replaced prior shard held and no shard has after the publish
  (``lost``), or that the export and a prior shard the export leaves in place both hold
  (``double``): a loader would read it twice. Only that column is read;
* a shard series that is not ``0..M-1`` of one ``M`` per (config, split, source, release) once the
  prior files are added: a stale ``-of-00002`` next to a new ``-of-00001``;
* a ``data/<config>`` folder of the prior tree that the export's ``README.md`` ``configs:`` does
  not declare (``undeclared_prior_configs``): the Hub keeps its files and no loader finds them.

For every config in both trees the counts hold ``collisions`` (the same path) and ``leftovers``
(prior paths the export does not overwrite). Without ``--v1-dir`` the gate fails: it would have
checked nothing. The output names configs and shard paths, never an image id."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

from . import GateResult
from .shards import config_shards, parse_shard, shard_columns
from .v11_common import IMAGES, Collector, card_configs, column_values

NAME = "publish_tree"
SHA = "image_sha256"
CARD = "README.md"
SHOWN = 3
"""Paths a failure line names."""


def _listed(names: Iterable[str]) -> str:
    names = sorted(names)
    more = f", and {len(names) - SHOWN} more" if len(names) > SHOWN else ""
    return ", ".join(names[:SHOWN]) + more


def _shas(files: Sequence[Path]) -> set[str]:
    return {str(sha) for sha in column_values(files, SHA) if sha}


def _compare(config: str, prior: Sequence[Path], export: Sequence[Path], out: Collector) -> Counter:
    """Collisions, leftovers, lost and double of one config that both trees hold."""
    before, after = {p.name: p for p in prior}, {p.name: p for p in export}
    same = sorted(set(before) & set(after))
    stale = [before[name] for name in sorted(set(before) - set(after))]
    seen = Counter(collisions=len(same), leftovers=len(stale))
    if same and config == IMAGES:
        out.fail(
            "collisions",
            f"{config}: {len(same)} export shard(s) take the path of a prior shard: "
            f"{_listed(same)}",
        )
    if all(SHA in shard_columns(p) for p in (*prior, *export)):
        kept, new = _shas(stale), _shas(export)
        lost = _shas([before[name] for name in same]) - new - kept
        double = new & kept
        seen.update(lost=len(lost), double=len(double))
        if lost:
            out.fail("lost", f"{config}: {len(lost)} image_sha256 in no shard after the publish")
        if double:
            out.fail(
                "double",
                f"{config}: {len(double)} image_sha256 in the export and in "
                f"{len(stale)} prior shard(s) it leaves in place",
            )
    return seen


def _check_series(config: str, names: Iterable[str], out: Collector) -> int:
    """Fail each series of ``config`` that is not ``0..M-1`` of one ``M``; the number of series."""
    series: dict[tuple, list] = defaultdict(list)
    for name in sorted(names):
        shard = parse_shard(name)
        if shard is None:
            out.fail("unparsed_shard", f"data/{config}/{name}: not a shard file name")
        else:
            series[(shard.split, shard.source, shard.release)].append(shard)
    for key, shards in sorted(series.items(), key=lambda item: tuple(map(str, item[0]))):
        label = f"data/{config}/" + "-".join(filter(None, key))
        totals, indexes = {s.total for s in shards}, {s.index for s in shards}
        if len(totals) > 1:
            out.fail("series", f"{label}: one series of {sorted(totals)} parts")
            continue
        total = totals.pop()
        missing = sorted(set(range(total)) - indexes)
        beyond = sorted(i for i in indexes if i >= total)
        if missing or beyond:
            out.fail("series", f"{label}: of {total} parts, missing {missing}, beyond {beyond}")
    return len(series)


def run_publish_tree_gate(build_dir: Path, *, v1_dir: Path | None = None) -> GateResult:
    build, out = Path(build_dir), Collector()
    if v1_dir is None:
        out.fail("no_prior_tree", "no prior tree: pass --v1-dir (the Hub repo before the publish)")
        return out.result(NAME)
    prior, export = config_shards(Path(v1_dir)), config_shards(build)
    if not prior:
        out.fail("no_prior_tree", "no data/<config>/*.parquet under --v1-dir: nothing to compare")
    if not export:
        out.fail("no_export", "the export holds no data/<config>/*.parquet")
    if not prior or not export:
        return out.result(NAME)

    both = sorted(set(prior) & set(export))
    per = {config: _compare(config, prior[config], export[config], out) for config in both}
    total = sum(per.values(), Counter())
    names = {
        c: {p.name for p in (*prior.get(c, ()), *export.get(c, ()))} for c in {*prior, *export}
    }
    series = sum(_check_series(config, names[config], out) for config in sorted(names))

    undeclared: list[str] = []
    if not (build / CARD).is_file():
        out.fail("no_card", f"{CARD} is missing from the export: its `configs:` cannot be read")
    else:
        undeclared = sorted(set(prior) - set(card_configs(build / CARD)))
    for config in undeclared:
        out.fail(
            "undeclared_prior_config",
            f"{config}: a prior config folder ({len(prior[config])} files stay on the Hub) "
            f"that the {CARD} `configs:` does not declare",
        )
    images = per.get(IMAGES, Counter())
    return out.result(
        NAME,
        prior_configs=len(prior), export_configs=len(export), configs_in_both=len(both),
        collisions=total["collisions"], leftovers=total["leftovers"],
        lost=total["lost"], double=total["double"],
        images_collisions=images["collisions"], images_leftovers=images["leftovers"],
        series=series, incomplete_series=out.counts["failed_series"],
        undeclared_prior_configs=len(undeclared),
    )  # fmt: skip
