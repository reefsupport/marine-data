"""WP-10 evaluation: synthetic recall, v1 low-texture false merges, MIH scaling, audit sheets."""

from __future__ import annotations

import io
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .confirm import ConfirmRules, DedupIndex, FeatureTable, confirm, merge_channels, score_pairs
from .corpus import Item, load_thumbs, read_payloads
from .crop import THUMB_SIDE, Thumb, crop_candidates, crop_refine
from .features import features_from_image
from .mih import MultiIndexHamming, unique_pairs

FAMILIES = ("resize", "jpeg", "crop", "jitter", "flip")


def load_corpus(
    out_dir: Path, cache_path: Path | None
) -> tuple[FeatureTable, np.ndarray | None, list[str]]:
    import pyarrow.parquet as pq

    from .embed import EmbeddingCache

    table = pq.read_table(out_dir / "hashes.parquet")
    shas = table.column("sha256").to_pylist()
    first = sorted({s: i for i, s in reversed(list(enumerate(shas)))}.values())
    table = table.take(first)
    uniq = table.column("sha256").to_pylist()
    emb = None
    if cache_path is not None:
        have = EmbeddingCache(cache_path).get_many(uniq)
        emb = np.stack([have[s].astype(np.float32) for s in uniq])
    return FeatureTable.from_arrow(table), emb, uniq


def augment(img, family: str, rng: random.Random):  # type: ignore[no-untyped-def]
    """``(augmented PIL image, jpeg quality, description)``."""
    from PIL import Image, ImageEnhance, ImageOps

    img = img.convert("RGB")
    w, h = img.size
    quality = 90
    if family == "resize":
        s = rng.uniform(0.25, 0.75)
        out, desc = (
            img.resize((max(16, round(w * s)), max(16, round(h * s))), Image.Resampling.BICUBIC),
            f"s={s:.2f}",
        )
    elif family == "jpeg":
        quality = rng.randint(30, 95)
        out, desc = img, f"q={quality}"
    elif family == "crop":
        sx, sy = rng.uniform(0.60, 0.95), rng.uniform(0.60, 0.95)
        cw, ch = round(w * sx), round(h * sy)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        out, desc = img.crop((x0, y0, x0 + cw, y0 + ch)), f"crop={sx:.2f}x{sy:.2f}"
    elif family == "jitter":
        b, c, s, hue = (
            rng.uniform(0.7, 1.3),
            rng.uniform(0.7, 1.3),
            rng.uniform(0.7, 1.3),
            rng.uniform(-0.05, 0.05),
        )
        out = ImageEnhance.Color(
            ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(b)).enhance(c)
        ).enhance(s)
        hsv = np.asarray(out.convert("HSV")).copy()
        hsv[..., 0] = (hsv[..., 0].astype(np.int16) + round(hue * 255)) % 256
        out, desc = (
            Image.fromarray(hsv, "HSV").convert("RGB"),
            f"b={b:.2f} c={c:.2f} s={s:.2f} h={hue:+.3f}",
        )
    elif family == "flip":
        vertical = rng.random() < 0.5
        out, desc = (
            (ImageOps.flip(img) if vertical else ImageOps.mirror(img)),
            ("vflip" if vertical else "hflip"),
        )
    else:
        raise ValueError(family)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=quality)  # every derivative is re-encoded, as real copies are
    return Image.open(io.BytesIO(buf.getvalue())), quality, desc


def synthetic_eval(
    items: list[Item],
    out_dir: Path,
    weights: Path,
    cache_path: Path,
    rules: ConfirmRules,
    per_family: int = 400,
    seed: int = 0,
    dedup_crop: bool = False,
    log=print,  # type: ignore[no-untyped-def]
) -> dict[str, object]:
    from .embed import SSCDEmbedder, prepare

    table, emb, uniq = load_corpus(out_dir, cache_path)
    pos = {s: i for i, s in enumerate(uniq)}
    rng = random.Random(seed)
    by_sha = {it.sha256: it for it in items if it.sha256 in pos}
    picks = rng.sample(sorted(by_sha), per_family * len(FAMILIES))
    by_rg: dict[tuple, list[str]] = defaultdict(list)
    for s in picks:
        by_rg[by_sha[s].payload[:2]].append(s)  # type: ignore[index]
    feats, arrays, meta, qthumbs = [], [], [], []
    for shas in by_rg.values():
        payloads = read_payloads([by_sha[s].payload for s in shas])
        for s, data in zip(shas, payloads, strict=True):
            fam = FAMILIES[picks.index(s) % len(FAMILIES)]
            from PIL import Image

            with Image.open(io.BytesIO(data)) as src:
                aug, _, desc = augment(src, fam, rng)
            feats.append(features_from_image(aug, f"aug:{s}"))
            arrays.append(prepare(aug))
            rgb = aug.convert("RGB")
            rgb.thumbnail((THUMB_SIDE, THUMB_SIDE))
            qthumbs.append(Thumb(gray=rgb.convert("L"), rgb=rgb))
            meta.append((s, fam, desc))
    log(f"synthetic: {len(meta)} derivatives built")
    qt = FeatureTable.from_features(feats)
    embedder = SSCDEmbedder(weights)
    qemb = embedder.embed_arrays(arrays)
    index = DedupIndex(table, emb, rules)
    results: dict[str, dict[str, object]] = {}
    for mode, use_knn in (("hash+embed-knn", True), ("hash-candidates-only", False)):
        chans = index.query_candidates(qt, qemb if use_knn else None)
        a, b, _ = merge_channels(chans, len(uniq))
        sc = score_pairs(qt, table, a, b, qemb, emb)
        ok, kind = confirm(sc, rules)
        if dedup_crop:
            area_q, area_c = qt["area"][a], table["area"][b]
            need = {uniq[j] for j in b[crop_candidates(sc, ok, area_q, area_c, rules)].tolist()}
            thumbs = load_thumbs(sorted(need & set(by_sha)), by_sha)  # type: ignore[arg-type]

            def _own(k, a_is_patch, qemb=qemb, emb=emb, a=a, b=b):  # type: ignore[no-untyped-def]
                return qemb[a[k]] if a_is_patch else emb[b[k]]

            def box_cos(k, parent_rgb, box, a_is_patch, _own=_own, embedder=embedder):  # type: ignore[no-untyped-def]
                region = parent_rgb.crop(box)
                if region.width < 8 or region.height < 8:
                    return None
                vec = embedder.embed_arrays([prepare(region)])[0]
                return float(np.dot(vec, _own(k, a_is_patch)))

            ok, kind, _ = crop_refine(
                sc,
                ok,
                kind,
                area_q,
                area_c,
                lambda k, a=a: qthumbs[a[k]],
                lambda k, b=b, thumbs=thumbs: thumbs.get(uniq[b[k]]),
                rules,
                box_cos=box_cos,
            )
        target = np.array([pos[meta[i][0]] for i in a.tolist()], dtype=np.int64)
        hit_pair = ok & (b == target)
        hit = np.zeros(len(meta), bool)
        hit[a[hit_pair]] = True
        cand = np.zeros(len(meta), bool)
        cand[a[b == target]] = True
        other = Counter(a[ok & (b != target)].tolist())
        fam = np.array([m[1] for m in meta])
        lt = qt["lowtex"]
        results[mode] = {
            "recall": float(hit.mean()),
            "candidate_recall": float(cand.mean()),
            "n": len(meta),
            "by_family": {f: float(hit[fam == f].mean()) for f in FAMILIES},
            "lowtex_recall": float(hit[lt].mean()) if lt.any() else None,
            "lowtex_n": int(lt.sum()),
            "queries_with_other_confirmed": len(other),
            "crop_confirmed_pairs": int((kind == "crop").sum()),
        }
        if use_knn:
            tpos = sc["cos"][b == target]
            results["positive_cos_pct"] = {p: float(np.percentile(tpos, p)) for p in (1, 5, 10, 50)}
            misses = [meta[i] for i in np.nonzero(~hit)[0].tolist()]
            results["misses_sample"] = [f"{m[1]} {m[2]} {m[0][:12]}" for m in misses[:15]]
    return results


def lowtex_before_after(out_dir: Path, v1_prefix: str = "v1:") -> dict[str, object]:
    """Replay v1 rule A (dHash <= 8) / rule B (<= 4) over the v1 images and ask what v2
    confirms. Before = every v1 pair counts; after = the pair shares a v2 dup cluster."""
    import pyarrow.parquet as pq

    h = pq.read_table(
        out_dir / "hashes.parquet", columns=["sha256", "source", "dhash", "lowtex", "split_group"]
    )
    src, sha = h.column("source").to_pylist(), h.column("sha256").to_pylist()
    keep = sorted(
        {
            s: i for i, (s, so) in enumerate(zip(sha, src, strict=True)) if so.startswith(v1_prefix)
        }.values()
    )
    shas = [sha[i] for i in keep]
    codes = h.column("dhash").to_numpy()[keep].astype(np.uint64)
    lowtex = h.column("lowtex").to_numpy(zero_copy_only=False)[keep]
    group = [h.column("split_group")[i].as_py() for i in keep]
    index = MultiIndexHamming.build(codes)
    a, b, d = unique_pairs(index.search(codes, 8, self_join=True), len(codes))
    cl = pq.read_table(out_dir / "clusters.parquet", columns=["sha256", "dup_cluster_id"])
    cid = dict(
        zip(cl.column("sha256").to_pylist(), cl.column("dup_cluster_id").to_pylist(), strict=True)
    )
    same = np.array(
        [cid[shas[i]] == cid[shas[j]] for i, j in zip(a.tolist(), b.tolist(), strict=True)], bool
    )
    lt = lowtex[a] | lowtex[b]
    cross = np.array(
        [group[i] != group[j] for i, j in zip(a.tolist(), b.tolist(), strict=True)], bool
    )
    out: dict[str, object] = {"v1_images": len(shas), "lowtex_images": int(lowtex.sum())}
    for rule, lim in (("ruleA_le8", 8), ("ruleB_le4", 4)):
        m = d <= lim
        out[rule] = {
            "pairs": int(m.sum()),
            "pairs_kept_v2": int((m & same).sum()),
            "lowtex_pairs": int((m & lt).sum()),
            "lowtex_kept_v2": int((m & lt & same).sum()),
            "cross_group_pairs": int((m & cross).sum()),
            "cross_group_kept_v2": int((m & cross & same).sum()),
            "lowtex_cross_group_pairs": int((m & lt & cross).sum()),
            "lowtex_cross_group_kept_v2": int((m & lt & cross & same).sum()),
        }
    return out


def scaling_bench(
    n: int = 1_000_000, radii: tuple[int, ...] = (3, 7), seed: int = 0
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 2**63, size=n, dtype=np.int64).astype(np.uint64) ^ (
        rng.integers(0, 2, size=n, dtype=np.int64).astype(np.uint64) << np.uint64(63)
    )
    planted = n // 100
    src = rng.choice(n - planted, planted, replace=False)
    flips = np.zeros(planted, np.uint64)
    for _ in range(3):
        flips |= np.uint64(1) << rng.integers(0, 64, planted).astype(np.uint64)
    codes[n - planted :] = codes[src] ^ flips
    t0 = time.time()
    index = MultiIndexHamming.build(codes)
    out: dict[str, object] = {
        "n": n,
        "build_s": round(time.time() - t0, 2),
        "index_mb": round(index.nbytes() / 2**20, 1),
    }
    for r in radii:
        t0 = time.time()
        a, b, _ = unique_pairs(index.search(codes, r, self_join=True), n)
        found = set(zip(a.tolist(), b.tolist(), strict=True))
        truth = {
            (int(min(s, n - planted + k)), int(max(s, n - planted + k)))
            for k, s in enumerate(src.tolist())
        }
        out[f"r{r}"] = {
            "query_s": round(time.time() - t0, 2),
            "pairs": len(found),
            "planted_recall": round(len(truth & found) / len(truth), 4),
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out["peak_rss_gb"] = round(rss / (2**30 if sys.platform == "darwin" else 2**20), 3)
    return out
