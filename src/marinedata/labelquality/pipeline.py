"""WP-9 runners: load an HF build + WP-10 groups + a feature cache, write the outputs.

``run_agreement`` → ``agreement.json`` + ``conflicts.tsv``; ``run_confident`` →
``label_issues.parquet`` + ``noise.json``; ``audit_sample`` → a stratified sample of flags.
The release is only read, never modified.
"""

from __future__ import annotations

import glob
import json
import sqlite3
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import agreement as A
from . import confident as CL
from .features import load_features

TASKS = ("coral-health-binary", "bleaching-condition")
NEVER_EVAL = frozenset({"coralscop-masks-rs"})
C_GRID = (0.001, 0.01, 0.1, 1.0)

# WP-9 D-U2: a confident-learning flag becomes ``flagged_hard`` only in a direction
# whose audited precision (``docs/label-quality/model-audit.tsv``) clears this floor.
# Today only toward-HEALTHY clears it (87.1%); toward-BLEACHED/UNHEALTHY (36.5%) does
# not — excluding those from val/test would make the eval easier than reality. See
# ``docs/label-quality.md`` §5c and ``run_label_status``.
HARD_FLAG_MIN_PRECISION = 0.60


def load_labels(hf_root: Path, task: str):  # type: ignore[no-untyped-def]
    import pandas as pd

    cols = ["image_sha256", "source_id", "sample_key", "label", "native_label", "label_reason"]
    parts = [
        pd.read_parquet(f, columns=cols).assign(split=Path(f).name.split("-")[0])
        for f in sorted(glob.glob(str(hf_root / "data" / task / "*.parquet")))
    ]
    df = pd.concat(parts, ignore_index=True)
    return df[df.label.notna()].reset_index(drop=True)


def load_sources(hf_root: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    out: dict[str, str] = {}
    for f in sorted(glob.glob(str(hf_root / "data" / "images" / "*.parquet"))):
        t = pq.read_table(f, columns=["image_sha256", "source_ids"])
        out.update(zip(t.column(0).to_pylist(), t.column(1).to_pylist(), strict=True))
    return out


def d0_clusters(dhash_db: Path, sources: dict[str, str]) -> dict[str, list[str]]:
    """dHash-0 clusters over eval-capable v1 sha (the S47 ``near_dup_pairs`` population)."""
    con = sqlite3.connect(f"file:{dhash_db}?mode=ro", uri=True)
    rows = con.execute("SELECT sha256, dhash FROM dhash").fetchall()
    con.close()
    keep = {s for s, v in sources.items() if set(v.split(",")) - NEVER_EVAL}
    return A.dhash_clusters({s: h for s, h in rows if s in keep})


def run_agreement(hf_root: Path, dhash_db: Path, out_dir: Path) -> dict[str, object]:
    sources = load_sources(hf_root)
    d0 = d0_clusters(dhash_db, sources)
    n_d0 = sum(len(v) * (len(v) - 1) // 2 for v in d0.values())
    result: dict[str, object] = {"d0_sha_pairs": n_d0, "d0_clusters": len(d0), "tasks": {}}
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        lab = load_labels(hf_root, task)
        units: dict[str, list[A.Unit]] = {}
        for r in lab.itertuples():
            units.setdefault(r.image_sha256, []).append(
                A.Unit(r.image_sha256, r.source_id, r.label)
            )
        same = {f"sha:{s}": [s] for s, us in units.items() if len(us) > 1}
        per: dict[str, object] = {}
        for name, clusters in (("dhash0", d0), ("identical_sha", same)):
            pairs = A.unit_pairs(clusters, units)
            for scope, sel in (
                ("cross_source", [p for p in pairs if p.cross_source]),
                ("within_source", [p for p in pairs if not p.cross_source]),
            ):
                per[f"{name}/{scope}"] = asdict(A.summarise(sel))
        conf = A.conflicts(u for us in units.values() for u in us)
        lab[lab.image_sha256.isin(conf)].sort_values(["image_sha256", "source_id"]).to_csv(
            out_dir / f"conflicts-{task}.tsv", sep="\t", index=False
        )
        per["conflicting_identical_sha"] = len(conf)
        result["tasks"][task] = per  # type: ignore[index]
    (out_dir / "agreement.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    return result


def _groups(dedup_groups: Path) -> dict[str, str]:
    import pyarrow.parquet as pq

    t = pq.read_table(dedup_groups, columns=["sha256", "split_group_id"])
    return dict(zip(t.column(0).to_pylist(), t.column(1).to_pylist(), strict=True))


def run_confident(
    hf_root: Path, dedup_groups: Path, features: Path, out_dir: Path, *, k: int = 5
) -> dict[str, object]:
    import pandas as pd

    shas, mat = load_features(features)
    row_of = {s: i for i, s in enumerate(shas)}
    group_of = _groups(dedup_groups)
    issues = []
    summary: dict[str, object] = {"k": k, "c_grid": list(C_GRID), "tasks": {}}
    for task in TASKS:
        lab = load_labels(hf_root, task)
        lab = lab[lab.image_sha256.isin(row_of)].reset_index(drop=True)
        classes = sorted(lab.label.unique())
        y = lab.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
        x = mat[lab.image_sha256.map(row_of).to_numpy()]
        groups = [group_of.get(s, f"sha:{s}") for s in lab.image_sha256]
        folds = CL.group_folds(groups, k)
        best = None
        for c in C_GRID:
            p = CL.oof_probs(x, y, folds, len(classes), c)
            ll = float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1))))
            acc = float(np.mean(p.argmax(1) == y))
            if best is None or ll < best[1]:
                best = (c, ll, acc, p)
        assert best is not None
        c, ll, acc, probs = best
        thr = CL.per_class_thresholds(probs, y)
        assigned = CL.confident_assign(probs, thr)
        flag = CL.label_issues(y, assigned)
        per_source = {}
        for src, idx in lab.groupby("source_id").indices.items():
            g = [groups[i] for i in idx]
            per_source[src] = {
                "n": len(idx),
                "flagged": int(flag[idx].sum()),
                "noise": CL.noise_rate(y[idx], assigned[idx], len(classes)),
                "noise_ci95": CL.group_bootstrap_noise(y[idx], assigned[idx], g, len(classes)),
            }
        summary["tasks"][task] = {  # type: ignore[index]
            "classes": classes,
            "n": len(y),
            "c": c,
            "oof_logloss": ll,
            "oof_accuracy": acc,
            "thresholds": thr.tolist(),
            "flagged": int(flag.sum()),
            "noise": CL.noise_rate(y, assigned, len(classes)),
            "noise_ci95": CL.group_bootstrap_noise(y, assigned, groups, len(classes)),
            "per_source": per_source,
        }
        sel = np.flatnonzero(flag)
        issues.append(
            pd.DataFrame(
                {
                    "image_sha256": lab.image_sha256.to_numpy()[sel],
                    "task": task,
                    "source_id": lab.source_id.to_numpy()[sel],
                    "sample_key": lab.sample_key.to_numpy()[sel],
                    "split": lab.split.to_numpy()[sel],
                    "given": [classes[i] for i in y[sel]],
                    "suggested": [classes[i] for i in assigned[sel]],
                    "self_confidence": probs[sel, y[sel]],
                    "suggested_prob": probs[sel, assigned[sel]],
                    "fold": folds[sel],
                }
            )
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(issues, ignore_index=True).sort_values(["task", "self_confidence"])
    table.to_parquet(out_dir / "label_issues.parquet", index=False)
    (out_dir / "noise.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
    return summary


# The bleaching-family lineage (label-origin.yaml, WP-9 D-U (4)); v3i is deliberately not
# in this set — its Unhealthy label is a different concept, not a rater of the same call.
V3I = "roboflow-coral-reef-classification-v3i"
V13I = "roboflow-coral-classification-copy-changed-v13i"
V6I = "roboflow-coral-bleaching-final-v6i"
V1_YOLOV8S = "roboflow-coral-bleaching-general-v1-yolov8s"

# label_status precedence when a sha256 hits more than one rule (WP-9 D-U).
_STATUS_RANK = {"conflict": 3, "ambiguous": 2, "flagged_hard": 1, "ok": 0}


def classify_conflict(rows: list[tuple[str, str]]) -> str:
    """One of the 14 conflicting-identical-image sha256 → ``conflict`` | ``ambiguous`` | ``ok``.

    ``rows`` is ``(source_id, label)`` for one sha256 in the ``coral-health-binary``
    identical-sha conflict set (WP-9 D-U (2)); rule, not a lookup table, so it re-derives
    the same categories tabled in ``docs/label-quality.md`` §2:

    * a source appears twice with different labels (within-source duplicate) → ``conflict``;
    * v3i disagrees with any other source → ``ok`` (a concept mismatch, not an error: v3i's
      "Unhealthy" is a different axis, per D-U (1), never a rater of the bleaching family);
    * v13i disagrees with v1-yolov8s alone (v6i absent) → ``conflict`` (the "Copy (Changed)"
      fork relabelling pigmented brain coral as Bleached);
    * v1-yolov8s disagrees with v6i and v13i together → ``ambiguous`` (pale soft coral, an
      expert call — D-I).
    """
    sources = [s for s, _ in rows]
    if len(sources) != len(set(sources)):
        return "conflict"
    by_source = dict(rows)
    if V3I in by_source and any(s != V3I for s in sources):
        return "ok"
    if V13I in by_source and V1_YOLOV8S in by_source and V6I not in by_source:
        return "conflict"
    if V13I in by_source and V6I in by_source and V1_YOLOV8S in by_source:
        return "ambiguous"
    return "conflict"  # pragma: no cover - no such case in the audited population


def _cl_direction(suggested: str) -> str:
    """A confident-learning flag's direction (WP-9 D-U2), keyed off ``suggested`` alone.

    Task-agnostic: ``bleaching-condition`` suggests BLEACHED, ``coral-health-binary``
    suggests UNHEALTHY — both are the same "toward-unhealthy" direction, the audited
    36.5%-precision one. Anything suggesting HEALTHY is the 87.1%-precision direction.
    """
    return "toward_healthy" if suggested == "HEALTHY" else "toward_bleached_unhealthy"


def audited_direction_precision(model_audit_tsv: Path) -> dict[str, float]:
    """Per-direction flag precision from the blind model audit (WP-9 D-U2).

    Over ``decided`` rows only (``verdict_class`` not null — the model could tell H/U/N
    from the thumbnail; matches ``docs/label-quality.md`` §5's 147/235 decided count).
    Reproduces the 87.1% (toward-healthy) / 36.5% (toward-bleached/unhealthy) split
    reported there.
    """
    import pandas as pd

    audit = pd.read_csv(model_audit_tsv, sep="\t")
    decided = audit[audit.verdict_class.notna()]
    direction = decided.suggested.map(_cl_direction)
    out: dict[str, float] = {}
    for d in ("toward_healthy", "toward_bleached_unhealthy"):
        sub = decided[direction == d]
        out[d] = float(sub.flag_correct.mean()) if len(sub) else 0.0
    return out


def run_label_status(
    hf_root: Path,
    labelquality_dir: Path,
    model_audit_tsv: Path,
    out_path: Path,
    *,
    hard_flag_min_precision: float = HARD_FLAG_MIN_PRECISION,
) -> dict[str, int]:
    """``label_status`` per sha256 (WP-9 D-U/D-U2): ``ok|conflict|ambiguous|flagged_hard``.

    Population: every sha256 labelled on either task. Rules, in precedence order:
    the 14 identical-sha conflicts (:func:`classify_conflict`, over
    ``conflicts-coral-health-binary.tsv`` — the superset of both tasks' conflicts) beat a
    confident-learning flag (``label_issues.parquet``), which beats ``ok``.

    D-U2: a CL flag only becomes ``flagged_hard`` when its direction's audited precision
    (:func:`audited_direction_precision` over ``model_audit_tsv`) is at least
    ``hard_flag_min_precision``. Every other CL flag — today, every toward-BLEACHED/
    UNHEALTHY one — stays whatever it would otherwise be (``ok`` unless it is also a
    conflict/ambiguous row). Every row gets ``cl_flag``/``cl_flag_direction`` columns:
    true/direction for any sha256 that is a raw CL flag (regardless of final status),
    null for the rest. The per-direction precisions are written into the parquet's
    schema metadata (``label_quality.direction_precision``). Non-``ok`` rows are
    excluded from val/test scoring (kept in train/pretrain); the eval harness applies
    that filter, this module only labels it.
    """
    import json

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    status: dict[str, str] = {}
    for task in TASKS:
        for sha in load_labels(hf_root, task).image_sha256.unique():
            status.setdefault(sha, "ok")

    conflicts = pd.read_csv(labelquality_dir / "conflicts-coral-health-binary.tsv", sep="\t")
    for sha, grp in conflicts.groupby("image_sha256"):
        cls = classify_conflict(list(zip(grp.source_id, grp.label, strict=True)))
        status[sha] = cls

    precision = audited_direction_precision(model_audit_tsv)
    issues = pd.read_parquet(
        labelquality_dir / "label_issues.parquet", columns=["image_sha256", "suggested"]
    )
    direction_of: dict[str, str] = {
        sha: _cl_direction(suggested)
        for sha, suggested in zip(issues.image_sha256, issues.suggested, strict=True)
    }
    for sha, direction in direction_of.items():
        if precision[direction] < hard_flag_min_precision:
            continue
        if _STATUS_RANK["flagged_hard"] > _STATUS_RANK.get(status.get(sha, "ok"), 0):
            status[sha] = "flagged_hard"

    table = pd.DataFrame({"sha256": list(status), "label_status": list(status.values())})
    table["cl_flag"] = pd.array(
        [True if s in direction_of else None for s in table.sha256], dtype="boolean"
    )
    table["cl_flag_direction"] = [direction_of.get(s) for s in table.sha256]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pandas(table, preserve_index=False)
    meta = {
        b"label_quality.direction_precision": json.dumps(precision).encode(),
        b"label_quality.hard_flag_min_precision": repr(hard_flag_min_precision).encode(),
    }
    pq.write_table(arrow_table.replace_schema_metadata(meta), out_path)
    return dict(table.label_status.value_counts())


def audit_sample(issues, n: int, *, per_source_min: int = 10, seed: int = 20260925):  # type: ignore[no-untyped-def]
    """Stratified sample of flagged rows (task × source), distinct images, seeded."""
    rng = np.random.default_rng(seed)
    uniq = issues.drop_duplicates("image_sha256")
    strata = uniq.groupby(["task", "source_id"]).indices
    total = len(uniq)
    picks = []
    for key in sorted(strata):
        idx = strata[key]
        want = max(min(per_source_min, len(idx)), round(n * len(idx) / total))
        picks.extend(rng.choice(idx, size=min(want, len(idx)), replace=False).tolist())
    return uniq.iloc[sorted(picks)].reset_index(drop=True)
