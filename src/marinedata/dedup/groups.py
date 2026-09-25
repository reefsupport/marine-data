"""Grouping (``dup_cluster_id``, ``split_group_id``) and the split-leak gate.

``dup_cluster_id`` is the connected component of confirmed duplicate pairs over unique
``sha256``. ``split_group_id`` additionally unions every declared group key a record
carries — ``sg:<source>/<split_group>`` (video / sequence / dive / campaign as each
registry rule defines it), ``seq:``, ``stereo:``, ``parent:`` (patch -> parent) — so a
split can be assigned per group without leaking. Upstream split membership is carried,
not unioned (unioning "all upstream test images" into one group would be meaningless):
``upstream_splits`` lists every upstream split seen in the group, and the gate refuses a
group holding an upstream-``test`` member anywhere in our ``train``.

Ids are content-derived and stable across runs: ``dc-``/``sg-`` + the first 16 hex of
the smallest member sha256.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        parent = self.parent
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra


def dup_clusters(shas: Iterable[str], pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """``sha256 -> dup_cluster_id`` for every sha (singletons get their own id)."""
    uf = UnionFind()
    for sha in shas:
        uf.find(sha)
    for a, b in pairs:
        uf.union(a, b)
    members: dict[str, list[str]] = defaultdict(list)
    for sha in list(uf.parent):
        members[uf.find(sha)].append(sha)
    return {s: "dc-" + min(ms)[:16] for ms in members.values() for s in ms}


@dataclass(frozen=True)
class GroupRecord:
    record_id: str
    sha256: str
    keys: Sequence[str] = ()
    upstream_split: str | None = None


def split_groups(
    records: Sequence[GroupRecord], clusters: dict[str, str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """``(sha256 -> split_group_id, split_group_id -> sorted upstream splits)``."""
    uf = UnionFind()
    for rec in records:
        node = "img:" + rec.sha256
        uf.union(node, "dc:" + clusters.get(rec.sha256, rec.sha256))
        for key in rec.keys:
            uf.union(node, "key:" + key)
    members: dict[str, list[str]] = defaultdict(list)
    for rec in records:
        members[uf.find("img:" + rec.sha256)].append(rec.sha256)
    gid_of_root = {root: "sg-" + min(shas)[:16] for root, shas in members.items()}
    sha_to_group = {rec.sha256: gid_of_root[uf.find("img:" + rec.sha256)] for rec in records}
    upstream: dict[str, set[str]] = defaultdict(set)
    for rec in records:
        if rec.upstream_split:
            upstream[sha_to_group[rec.sha256]].add(rec.upstream_split)
    return sha_to_group, {g: sorted(s) for g, s in upstream.items()}


# --------------------------------------------------------------------------- gate


class DedupGateError(RuntimeError):
    """The release leaks: a split group spans splits or upstream test landed in train."""


@dataclass
class GateResult:
    rows: int = 0
    images: int = 0
    groups: int = 0
    ungrouped: list[str] = field(default_factory=list)
    spanning: dict[str, dict[str, int]] = field(default_factory=dict)
    upstream_test_in_train: list[str] = field(default_factory=list)
    allow_ungrouped: bool = False

    @property
    def ok(self) -> bool:
        return (
            not self.spanning
            and not self.upstream_test_in_train
            and (self.allow_ungrouped or not self.ungrouped)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "rows": self.rows,
            "images": self.images,
            "groups": self.groups,
            "ungrouped": len(self.ungrouped),
            "ungrouped_sample": sorted(self.ungrouped)[:20],
            "spanning_groups": len(self.spanning),
            "spanning_sample": dict(sorted(self.spanning.items())[:20]),
            "upstream_test_in_train": len(self.upstream_test_in_train),
            "upstream_test_in_train_sample": sorted(self.upstream_test_in_train)[:20],
        }


def load_groups(path: str | Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["sha256", "split_group_id", "group_upstream_splits"])
    sha_to_group: dict[str, str] = {}
    upstream: dict[str, list[str]] = {}
    for sha, gid, ups in zip(
        table.column("sha256").to_pylist(),
        table.column("split_group_id").to_pylist(),
        table.column("group_upstream_splits").to_pylist(),
        strict=True,
    ):
        sha_to_group[sha] = gid
        if ups:
            upstream[gid] = list(ups)
    return sha_to_group, upstream


def read_release_rows(release_dir: str | Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    manifests = sorted((Path(release_dir) / "manifests").glob("*.tsv"))
    if not manifests:
        raise DedupGateError(f"no manifests/*.tsv under {release_dir}")
    for manifest in manifests:
        with manifest.open() as fh:
            header = fh.readline().rstrip("\n").split("\t")
            if header[:2] != ["image_sha256", "split"]:
                raise DedupGateError(f"{manifest}: unexpected header {header}")
            for line in fh:
                sha, split = line.rstrip("\n").split("\t")[:2]
                rows.append((sha, split))
    return rows


def run_gate(
    rows: Sequence[tuple[str, str]],
    sha_to_group: dict[str, str],
    upstream: dict[str, list[str]],
    *,
    allow_ungrouped: bool = False,
    train: str = "train",
) -> GateResult:
    """Every task manifest row counts: an image in train for one task and test for another
    is leakage for any multi-task model trained on the release."""
    result = GateResult(rows=len(rows), allow_ungrouped=allow_ungrouped)
    splits_of: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    images: set[str] = set()
    for sha, split in rows:
        images.add(sha)
        gid = sha_to_group.get(sha)
        if gid is None:
            result.ungrouped.append(sha)
            gid = "sha:" + sha
        splits_of[gid][split] += 1
    result.images = len(images)
    result.ungrouped = sorted(set(result.ungrouped))
    result.groups = len(splits_of)
    for gid, counts in splits_of.items():
        if len(counts) > 1:
            result.spanning[gid] = dict(counts)
        if train in counts and "test" in upstream.get(gid, ()):
            result.upstream_test_in_train.append(gid)
    return result


def gate_release(
    release_dir: str | Path, groups_path: str | Path, *, allow_ungrouped: bool = False
) -> GateResult:
    sha_to_group, upstream = load_groups(groups_path)
    return run_gate(
        read_release_rows(release_dir), sha_to_group, upstream, allow_ungrouped=allow_ungrouped
    )


def write_gate_report(result: GateResult, path: str | Path) -> None:
    Path(path).write_text(json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n")
