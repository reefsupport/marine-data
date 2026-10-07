"""Corpus runner: enumerate inputs, hash, embed, dedup, group, write the parquet outputs.

Inputs are read-only: an HF ``images`` config directory (``<split>-NNNNN-of-NNNNN.parquet``
with embedded bytes) and staged source trees (``metadata.parquet`` + ``images/<partition>/``).
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .confirm import ConfirmRules, DedupIndex, FeatureTable, confirm, merge_channels, score_pairs
from .features import FeatureError, ImageFeatures, features_from_bytes
from .groups import GroupRecord, dup_clusters, split_groups

_CORALSCAPES_SEQ = re.compile(r"^(site\d+_\d+)_\d+_leftImg8bit$")


@dataclass
class Item:
    record_id: str
    source: str
    split: str | None
    upstream_split: str | None
    split_group: str | None
    keys: tuple[str, ...]
    payload: Path | tuple[str, int, int]  # staged file, or (parquet, row_group, row)
    sha256: str | None = None


def iter_hf_images(root: str | Path, label: str = "v1") -> Iterator[Item]:
    """Items reference ``(parquet, row_group, row)``; bytes are read per row group later."""
    import pyarrow.parquet as pq

    for path in sorted(Path(root).glob("*.parquet")):
        split = path.name.split("-")[0]
        pf = pq.ParquetFile(path)
        for rg in range(pf.num_row_groups):
            cols = ["image_sha256", "source_id", "split_group"]
            for pos, row in enumerate(pf.read_row_group(rg, columns=cols).to_pylist()):
                group = row["split_group"]
                yield Item(
                    record_id=f"{label}:{row['image_sha256']}",
                    source=f"{label}:{row['source_id']}",
                    split=split,
                    upstream_split=None,
                    split_group=group,
                    keys=(f"sg:{group}",) if group else (),
                    payload=(str(path), rg, pos),
                    sha256=row["image_sha256"],
                )


def iter_staged(root: str | Path, label: str) -> Iterator[Item]:
    import pyarrow.parquet as pq

    root = Path(root)
    by_stem = {p.stem: p for p in (root / "images").rglob("*") if p.is_file()}
    for row in pq.read_table(root / "metadata.parquet").to_pylist():
        stem = row["stem"]
        keys: list[str] = []
        match = _CORALSCAPES_SEQ.match(stem)
        if match:
            keys.append(f"seq:{label.split('@')[0]}/{match.group(1)}")
        yield Item(
            record_id=f"{label}:{stem}",
            source=label,
            split=row.get("upstream_split"),
            upstream_split=row.get("upstream_split"),
            split_group=None,
            keys=tuple(keys),
            payload=by_stem[stem],
        )


def _chunk_key(item: Item) -> tuple[str, int]:
    ref = item.payload
    return (ref[0], ref[1]) if isinstance(ref, tuple) else ("", -1)


def chunk_items(items: list[Item], size: int = 32) -> list[list[int]]:
    """Index chunks: one per parquet row group, ``size`` staged files otherwise."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    loose: list[int] = []
    for i, it in enumerate(items):
        key = _chunk_key(it)
        (groups[key] if key[1] >= 0 else loose).append(i)
    chunks = list(groups.values())
    chunks += [loose[lo : lo + size] for lo in range(0, len(loose), size)]
    return chunks


def read_payloads(refs: list[Path | tuple[str, int, int]]) -> list[bytes]:
    """Bytes for refs that share one row group (or are all staged files)."""
    tuples = [r for r in refs if isinstance(r, tuple)]
    if not tuples:
        return [Path(r).read_bytes() for r in refs]  # type: ignore[arg-type]
    import pyarrow.parquet as pq

    path, rg = tuples[0][0], tuples[0][1]
    col = pq.ParquetFile(path).read_row_group(rg, columns=["image"]).column("image").to_pylist()
    return [col[r[2]]["bytes"] for r in refs]  # type: ignore[index]


def _hash_chunk(job: tuple[list[int], list, list]) -> list[tuple[int, ImageFeatures | str]]:  # type: ignore[type-arg]
    idxs, refs, shas = job
    try:
        payloads = read_payloads(refs)
    except OSError as exc:
        return [(i, f"read failed: {exc}") for i in idxs]
    out: list[tuple[int, ImageFeatures | str]] = []
    for i, data, sha in zip(idxs, payloads, shas, strict=True):
        try:
            out.append((i, features_from_bytes(data, sha)))
        except FeatureError as exc:
            out.append((i, str(exc)))
    return out


def hash_items(items: list[Item], workers: int, log=print) -> list[ImageFeatures | str]:  # type: ignore[no-untyped-def]
    out: list[ImageFeatures | str] = [""] * len(items)
    jobs = (
        (c, [items[i].payload for i in c], [items[i].sha256 for i in c]) for c in chunk_items(items)
    )
    t0, done, mark = time.time(), 0, 0
    with get_context("spawn").Pool(workers) as pool:
        for res in pool.imap_unordered(_hash_chunk, jobs):
            for idx, feat in res:
                out[idx] = feat
            done += len(res)
            if done - mark >= 5000:
                mark = done
                log(f"hash {done}/{len(items)} {time.time() - t0:.0f}s")
    return out


def _prepare_bytes(data: bytes) -> np.ndarray:
    from PIL import Image

    from .embed import prepare

    with Image.open(io.BytesIO(data)) as img:
        return prepare(img)


def embed_items(uniq: dict[str, Item], weights: Path, cache_path: Path, log=print) -> np.ndarray:  # type: ignore[no-untyped-def]
    from concurrent.futures import ThreadPoolExecutor

    from .embed import EmbeddingCache, SSCDEmbedder

    cache = EmbeddingCache(cache_path)
    embedder = SSCDEmbedder(weights)
    shas = list(uniq)
    have = cache.get_many(shas)
    todo = [s for s in shas if s not in have]
    todo_items = [uniq[s] for s in todo]
    t0, done, mark = time.time(), 0, 0
    pending_keys: list[str] = []
    pending: list[np.ndarray] = []
    with ThreadPoolExecutor(8) as pool:
        for chunk in chunk_items(todo_items):
            keys = [todo[i] for i in chunk]
            payloads = read_payloads([todo_items[i].payload for i in chunk])
            pending += list(pool.map(_prepare_bytes, payloads))
            pending_keys += keys
            if len(pending) >= 256:
                cache.put_many(zip(pending_keys, embedder.embed_arrays(pending), strict=True))
                done += len(pending)
                pending, pending_keys = [], []
                if done - mark >= 4096:
                    mark = done
                    log(f"embed {done}/{len(todo)} {time.time() - t0:.0f}s")
        if pending:
            cache.put_many(zip(pending_keys, embedder.embed_arrays(pending), strict=True))
    have = cache.get_many(shas)
    return np.stack([have[s].astype(np.float32) for s in shas])


def load_thumbs(shas: list[str], uniq: dict[str, Item]) -> dict[str, object]:
    """``(gray, rgb)`` thumbnail pairs for ``shas``, reading each parquet row group once."""
    from .crop import thumb_pair

    groups: dict[tuple, list[str]] = defaultdict(list)  # type: ignore[type-arg]
    for s in shas:
        ref = uniq[s].payload
        groups[ref[:2] if isinstance(ref, tuple) else ("file", s)].append(s)
    out: dict[str, object] = {}
    for part in groups.values():
        for s, data in zip(part, read_payloads([uniq[s].payload for s in part]), strict=True):
            out[s] = thumb_pair(data)
    return out


def _box_cos_fn(embedder, emb, a, b):  # type: ignore[no-untyped-def]
    """A ``box_cos(k, parent_rgb, box, a_is_patch)`` closure for :func:`crop.crop_refine`."""
    from .embed import prepare

    def fn(k, parent_rgb, box, a_is_patch):  # type: ignore[no-untyped-def]
        region = parent_rgb.crop(box)
        if region.width < 8 or region.height < 8:
            return None
        vec = embedder.embed_arrays([prepare(region)])[0]
        other = emb[a[k]] if a_is_patch else emb[b[k]]
        return float(np.dot(vec, other))

    return fn


def _refine_crops(scores, ok, kind, a, b, table, shas, uniq, rules, log, *, emb=None, weights=None):  # type: ignore[no-untyped-def]
    from .crop import crop_candidates, crop_refine

    area = table["area"]
    todo = crop_candidates(scores, ok, area[a], area[b], rules)
    need = sorted({shas[i] for i in a[todo].tolist()} | {shas[j] for j in b[todo].tolist()})
    log(f"crop check: {len(todo)} pairs, {len(need)} images")
    thumbs = load_thumbs(need, uniq)
    box_cos = None
    if emb is not None and weights is not None and len(todo):
        from .embed import SSCDEmbedder

        box_cos = _box_cos_fn(SSCDEmbedder(weights), emb, a, b)
    ok, kind, ncc = crop_refine(
        scores,
        ok,
        kind,
        area[a],
        area[b],
        lambda k: thumbs[shas[a[k]]],
        lambda k: thumbs[shas[b[k]]],
        rules,
        box_cos=box_cos,
    )
    log(f"crop confirmed {int((kind == 'crop').sum())}")
    return ok, kind, ncc


@dataclass
class RunResult:
    items: list[Item]
    feats: list[ImageFeatures]
    uniq_shas: list[str]
    table: FeatureTable
    emb: np.ndarray | None
    pairs: dict[str, np.ndarray]
    clusters: dict[str, str]
    sha_to_group: dict[str, str]
    upstream: dict[str, list[str]]
    errors: list[tuple[str, str]] = field(default_factory=list)


def run_corpus(
    items: list[Item],
    *,
    workers: int,
    rules: ConfirmRules,
    weights: Path | None,
    cache_path: Path,
    embed_knn: bool = True,
    dedup_crop: bool = False,
    log=print,  # type: ignore[no-untyped-def]
) -> RunResult:
    raw = hash_items(items, workers, log)
    errors = [(it.record_id, f) for it, f in zip(items, raw, strict=True) if isinstance(f, str)]
    keep = [i for i, f in enumerate(raw) if not isinstance(f, str)]
    items = [items[i] for i in keep]
    feats: list[ImageFeatures] = [raw[i] for i in keep]  # type: ignore[misc]
    uniq: dict[str, Item] = {}
    uniq_feats: dict[str, ImageFeatures] = {}
    for it, f in zip(items, feats, strict=True):
        uniq.setdefault(f.sha256, it)
        uniq_feats.setdefault(f.sha256, f)
    shas = list(uniq)
    table = FeatureTable.from_features([uniq_feats[s] for s in shas])
    emb = embed_items(uniq, weights, cache_path, log) if weights is not None else None
    index = DedupIndex(table, emb, rules)
    channels = index.self_candidates(embed_knn=embed_knn)
    log("candidates " + " ".join(f"{k}={len(v[0])}" for k, v in channels.items()))
    a, b, bits = merge_channels(channels, len(shas))
    scores = score_pairs(table, table, a, b, emb, emb)
    ok, kind = confirm(scores, rules)
    if dedup_crop:
        ok, kind, ncc = _refine_crops(
            scores, ok, kind, a, b, table, shas, uniq, rules, log, emb=emb, weights=weights
        )
    else:
        ncc = np.full(len(ok), np.nan, np.float32)
    pairs = {"a": a, "b": b, "channels": bits, "confirmed": ok, "kind": kind, "ncc": ncc, **scores}
    clusters = dup_clusters(shas, ((shas[i], shas[j]) for i, j in zip(a[ok], b[ok], strict=True)))
    records = [
        GroupRecord(it.record_id, f.sha256, it.keys, it.upstream_split)
        for it, f in zip(items, feats, strict=True)
    ]
    sha_to_group, upstream = split_groups(records, clusters)
    log(f"confirmed {int(ok.sum())}/{len(a)} kinds {dict(Counter(kind[ok].tolist()))}")
    return RunResult(
        items, feats, shas, table, emb, pairs, clusters, sha_to_group, upstream, errors
    )


def write_outputs(res: RunResult, out: Path, channel_names: list[str]) -> dict[str, object]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for it, f in zip(res.items, res.feats, strict=True):
        row = f.as_row()
        row.update(
            record_id=it.record_id,
            source=it.source,
            split=it.split,
            upstream_split=it.upstream_split,
            split_group=it.split_group,
        )
        rows.append(row)
    names = ("dhash", "dhash_h", "dhash_v", "phash64", "phash64_h", "phash64_v")
    codes = {n: np.array([r.pop(n) for r in rows], dtype=np.uint64) for n in names}
    table = pa.Table.from_pylist(rows)
    for name in names:  # uint64 codes overflow pyarrow's int64 inference
        table = table.append_column(name, pa.array(codes[name], pa.uint64()))
    pq.write_table(table, out / "hashes.parquet")

    shas = res.uniq_shas
    p = res.pairs
    pq.write_table(
        pa.table(
            {
                "sha_a": [shas[i] for i in p["a"].tolist()],
                "sha_b": [shas[i] for i in p["b"].tolist()],
                "channels": [
                    ",".join(n for pos, n in enumerate(channel_names) if bits >> pos & 1)
                    for bits in p["channels"].tolist()
                ],
                **{
                    k: p[k]
                    for k in (
                        "d_dhash",
                        "orient",
                        "d_phash",
                        "cos",
                        "ncc",
                        "exact",
                        "pixel",
                        "lowtex",
                        "confirmed",
                    )
                },
                "kind": p["kind"].astype(str),
            }
        ),
        out / "pairs.parquet",
    )

    sources_of: dict[str, set[str]] = defaultdict(set)
    for it, f in zip(res.items, res.feats, strict=True):
        sources_of[f.sha256].add(it.source)
    size = Counter(res.clusters.values())
    pq.write_table(
        pa.table(
            {
                "sha256": shas,
                "dup_cluster_id": [res.clusters[s] for s in shas],
                "cluster_size": [size[res.clusters[s]] for s in shas],
                "sources": [sorted(sources_of[s]) for s in shas],
            }
        ),
        out / "clusters.parquet",
    )

    gsize = Counter(res.sha_to_group[s] for s in shas)
    pq.write_table(
        pa.table(
            {
                "record_id": [it.record_id for it in res.items],
                "source": [it.source for it in res.items],
                "split": [it.split for it in res.items],
                "sha256": [f.sha256 for f in res.feats],
                "dup_cluster_id": [res.clusters[f.sha256] for f in res.feats],
                "split_group_id": [res.sha_to_group[f.sha256] for f in res.feats],
                "group_size": [gsize[res.sha_to_group[f.sha256]] for f in res.feats],
                "group_upstream_splits": [
                    res.upstream.get(res.sha_to_group[f.sha256], []) for f in res.feats
                ],
            }
        ),
        out / "groups.parquet",
    )
    return overlap_matrix(res, sources_of)


def overlap_matrix(res: RunResult, sources_of: dict[str, set[str]]) -> dict[str, object]:
    """``matrix[A][B]`` = unique images of A whose dup cluster also holds an image of B.
    The diagonal counts images of A with another A image in their cluster."""
    members: dict[str, list[str]] = defaultdict(list)
    for sha in res.uniq_shas:
        members[res.clusters[sha]].append(sha)
    cluster_sources: dict[str, Counter[str]] = {}
    for cid, shas in members.items():
        c: Counter[str] = Counter()
        for s in shas:
            c.update(sources_of[s])
        cluster_sources[cid] = c
    labels = sorted({s for v in sources_of.values() for s in v})
    matrix = {a: dict.fromkeys(labels, 0) for a in labels}
    for sha in res.uniq_shas:
        cs = cluster_sources[res.clusters[sha]]
        for a in sources_of[sha]:
            for b, n in cs.items():
                if b != a or n > 1:
                    matrix[a][b] += 1
    return {
        "labels": labels,
        "matrix": matrix,
        "unique_images": {a: sum(1 for s in res.uniq_shas if a in sources_of[s]) for a in labels},
    }


def main_log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stdout, flush=True)


def dump_json(obj: object, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
