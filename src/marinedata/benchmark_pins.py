"""Manifest-driven split-map pins (design eval-decontamination-split-v2, section 2.2 rule 1).

``registry/benchmarks/manifests/<id>.parquet`` lists the upstream eval and held-out images of a
benchmark by sha256 with their upstream split. For a ``route-to-our-test`` benchmark the images of
the entry's ``eval_split`` pin to our ``test``; the images of its second held-out split (the
entry's ``heldout_val``, else the upstream ``val`` split when the eval split is another one) pin
to our ``val`` (the release's ``validation``). Every other manifest row reads as an eval image
(fail closed: the manifests list only eval and held-out images), so a benchmark staged under
another label than its registry entry (suim stages ``images`` for ``TEST``) or evaluated on
``val`` (uiis has no test split) is still pinned.

The split map matches a release row by its image sha256, whatever the staged ``upstream_split``
says; the manifest's ``image_sha256`` and ``bytes_sha256`` columns pin too (a release image's
``bytes_sha256`` equals its ``image_sha256``). Gate G7 judges the same two roles through
:func:`split_role`, so the map and the gate cannot disagree about which images are held-out.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .benchmarks import BenchmarkEntry, BenchmarkRegistry

EVAL, HELDOUT = "eval", "heldout"
PIN_TEST, PIN_VAL = "test", "validation"
MAP_NAME = {PIN_TEST: "test", PIN_VAL: "val"}
ACCEPTED = {
    PIN_TEST: frozenset({"test", "test_ood"}),
    PIN_VAL: frozenset({"val", "test", "test_ood"}),
}
OOD_PREFIXES = ("ood-", "ood_")
POLICY = "route-to-our-test"
VAL_NAMES = frozenset({"val", "valid", "validation"})
SHA_COLUMNS = ("sha256", "image_sha256", "bytes_sha256")
SPLIT_COLUMN = "upstream_split"
MAX_CONFLICT_ROWS = 200
PIN_RULE = (
    "route-to-our-test manifests: eval split -> test, second held-out split -> validation; "
    "never-eval train, OOD claims and v1.0 continuity win over a pin; test beats validation"
)
Row = tuple[str, str, str]  # image sha256, split group, source id
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _key(value: object) -> str:
    return str(value or "").strip().lower()


def _clean(value: object) -> str:
    text = _key(value)
    return text if _HEX64.fullmatch(text) else ""


def accepted(pin: str, split: str) -> bool:
    """Whether a map split may hold an image pinned ``pin`` (an OOD split always may)."""
    return split in ACCEPTED[pin] or split.startswith(OOD_PREFIXES)


def split_role(entry: BenchmarkEntry, raw_split: object) -> str:
    """``heldout`` for a row of the entry's second held-out split, else ``eval`` (fail closed).

    The held-out split is the entry's ``heldout_val`` when it declares one, else the upstream
    ``val`` split when the eval split is a different one (design section 1). Every other manifest
    row is an eval image, whatever its label says (suim: ``images``; uiis evaluates on ``val``).
    """
    upstream = entry.upstream_split
    if upstream is None:
        return EVAL
    split, heldout = _key(raw_split), _key(upstream.heldout_val)
    if heldout:
        same = split == heldout or (split in VAL_NAMES and heldout in VAL_NAMES)
        return HELDOUT if same else EVAL
    return HELDOUT if split in VAL_NAMES and _key(upstream.eval_split) not in VAL_NAMES else EVAL


@dataclass(frozen=True)
class BenchmarkPins:
    """``sha256 -> (pin, benchmark label)`` of every route-to-our-test manifest row.

    ``pin`` is ``test`` (an eval image) or ``validation`` (a held-out image); a digest listed in
    both roles pins ``test``. ``manifest_rows`` counts the distinct digests per benchmark and pin,
    whether or not the release holds them."""

    by_sha: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    manifest_rows: Mapping[str, Mapping[str, int]] = field(default_factory=dict)


def read_manifest_rows(path: Path) -> list[tuple[str, str]]:
    """``(sha256, upstream_split)`` for every digest cell of a parquet manifest."""
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema_arrow.names
    columns = [c for c in (SPLIT_COLUMN, *SHA_COLUMNS) if c in names]
    data = pq.read_table(path, columns=columns).to_pydict()
    count = max((len(values) for values in data.values()), default=0)
    labels = data.get(SPLIT_COLUMN) or [None] * count
    rows: list[tuple[str, str]] = []
    for column in SHA_COLUMNS:
        for value, label in zip(data.get(column, ()), labels, strict=False):
            if sha := _clean(value):
                rows.append((sha, str(label or "").strip()))
    return rows


def load_benchmark_pins(
    registry: BenchmarkRegistry | None = None, root: Path | None = None
) -> BenchmarkPins:
    """Pins from every ``route-to-our-test`` entry whose manifest file exists.

    Matching is by digest alone (a manifest sha is the benchmark image wherever it is staged), so
    an entry with no ``registry_id`` (uiis, uiis10k, usis10k) pins too; the label is
    ``registry_id`` else the benchmark id. ``exclude`` entries never pin (rule 2)."""
    if registry is None:
        registry = BenchmarkRegistry.load()
    by_sha: dict[str, tuple[str, str]] = {}
    seen: dict[str, dict[str, set[str]]] = {}
    for entry in sorted(registry.benchmarks, key=lambda e: e.id):
        path = registry.manifest_path(entry.id, root)
        if entry.policy != POLICY or not path.is_file():
            continue
        label = entry.registry_id or entry.id
        digests = seen.setdefault(label, {PIN_TEST: set(), PIN_VAL: set()})
        for sha, raw in read_manifest_rows(path):
            pin = PIN_VAL if split_role(entry, raw) == HELDOUT else PIN_TEST
            digests[pin].add(sha)
            current = by_sha.get(sha)
            if current is None or (pin == PIN_TEST and current[0] == PIN_VAL):
                by_sha[sha] = (pin, label)
    counts = {
        label: {PIN_TEST: len(d[PIN_TEST]), PIN_VAL: len(d[PIN_VAL] - d[PIN_TEST])}
        for label, d in sorted(seen.items())
    }
    return BenchmarkPins(by_sha=by_sha, manifest_rows=counts)


def component_pins(
    rows: Sequence[Row], canonical: Mapping[str, str], pins: BenchmarkPins
) -> dict[str, str]:
    """``component -> test | validation``: a group takes the pin of any member (test wins)."""
    targets: dict[str, str] = {}
    for sha, group, _source in rows:
        hit = pins.by_sha.get(sha)
        if hit is None:
            continue
        canon = canonical.get(group, group)
        if hit[0] == PIN_TEST or canon not in targets:
            targets[canon] = hit[0]
    return targets


def _component_priors(
    rows: Sequence[Row],
    canonical: Mapping[str, str],
    prior: Mapping[str, str],
    targets: Mapping[str, str],
) -> dict[str, set[str]]:
    """The v1.0 splits already held by the groups of each pinned component."""
    priors: dict[str, set[str]] = {}
    for _sha, group, _source in rows:
        canon = canonical.get(group, group)
        if canon in targets and (split := prior.get(group)) is not None:
            priors.setdefault(canon, set()).add(split)
    return priors


def v1_conflicts(
    rows: Sequence[Row],
    canonical: Mapping[str, str],
    pins: BenchmarkPins,
    targets: Mapping[str, str],
    priors: Mapping[str, set[str]],
) -> dict[str, Any]:
    """Pinned images whose component the v1.0 map already placed elsewhere.

    The v1.0 split wins (continuity), so these are counted and reported, never moved.
    ``would_fail_g7`` counts those gate G7 rejects: an eval image outside test/OOD, a held-out
    image outside validation/test/OOD."""
    found: dict[str, dict[str, Any]] = {}
    for sha, group, source in sorted(r for r in rows if r[0] in pins.by_sha):
        canon = canonical.get(group, group)
        have, want = priors.get(canon), MAP_NAME[targets[canon]]
        if not have or have == {want} or sha in found:
            continue
        pin, label = pins.by_sha[sha]
        found[sha] = {
            "sha256": sha,
            "source": source,
            "benchmark": label,
            "v1_split": "+".join(sorted(have)),
            "pin": want,
            "g7_fails": not all(accepted(pin, split) for split in have),
        }
    items = list(found.values())
    by_benchmark: dict[str, Counter[str]] = {}
    for item in items:
        key = f"{item['v1_split']} -> {item['pin']}"
        by_benchmark.setdefault(item["benchmark"], Counter())[key] += 1
    return {
        "count": len(items),
        "would_fail_g7": sum(item["g7_fails"] for item in items),
        "by_benchmark": {k: dict(sorted(v.items())) for k, v in sorted(by_benchmark.items())},
        "rows": items[:MAX_CONFLICT_ROWS],
        "rows_truncated": len(items) > MAX_CONFLICT_ROWS,
    }


def apply_benchmark_pins(
    forced: Mapping[str, str],
    rows: Sequence[Row],
    canonical: Mapping[str, str],
    pins: BenchmarkPins,
    *,
    prior: Mapping[str, str] | None = None,
    ood: Collection[str] = (),
) -> tuple[dict[str, str], dict[str, Any]]:
    """``(forced plus the manifest pins, the SPLIT_MAP ``pins`` header block)``.

    A pinned component yields to an OOD claim, to a v1.0 assignment (``prior``) and to any earlier
    ``forced`` entry (never-eval train, band, upstream test); the header counts each. A v1.0
    conflict is only reported here: ``resolve_inherited_splits`` keeps the prior split itself.
    ``(dict(forced), {})`` when no release row is pinned, so an unpinned map is unchanged."""
    targets = component_pins(rows, canonical, pins)
    if not targets:
        return dict(forced), {}
    priors = _component_priors(rows, canonical, prior, targets) if prior is not None else {}
    merged = dict(forced)
    placed: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    for canon, target in sorted(targets.items()):
        want, have = MAP_NAME[target], priors.get(canon)
        if canon in ood:
            skipped["ood"] += 1
        elif have and have != {want}:
            skipped["v1_continuity"] += 1
        elif merged.get(canon, want) != want:
            skipped["earlier_rule"] += 1
        else:
            merged[canon] = want
            placed[target] += 1
    matched: dict[str, dict[str, set[str]]] = {}
    for sha, _group, _source in rows:
        if hit := pins.by_sha.get(sha):
            matched.setdefault(hit[1], {PIN_TEST: set(), PIN_VAL: set()})[hit[0]].add(sha)
    header: dict[str, Any] = {
        "rule": PIN_RULE,
        "benchmarks": {
            label: {pin: len(shas) for pin, shas in by_pin.items()}
            for label, by_pin in sorted(matched.items())
        },
        "manifest_rows": {k: dict(v) for k, v in sorted(pins.manifest_rows.items())},
        "components": {pin: placed[pin] for pin in (PIN_TEST, PIN_VAL)},
        "skipped_components": {
            "ood": skipped["ood"],
            "v1_continuity": skipped["v1_continuity"],
            "earlier_rule": skipped["earlier_rule"],
        },
    }
    if prior is not None:
        header["v1_continuity_conflicts"] = v1_conflicts(rows, canonical, pins, targets, priors)
    return merged, header


def stamp_pins(out: Path, header: Mapping[str, Any]) -> None:
    """Record ``header`` as the written map's ``pins`` block; the assignments are untouched."""
    from .splitmap import load_split_map, save_split_map

    written = load_split_map(out)
    if written is None:
        raise ValueError(f"{out}: no split map to record the benchmark pins in")
    save_split_map(out, replace(written, pins=dict(header)))
