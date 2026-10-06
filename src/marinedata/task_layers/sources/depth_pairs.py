"""Depth + pair producers for the staged depth/pair sources (WP-U11).

Every producer reads only the bucket *listing* of ``images/`` and the tree's ``CHECKSUMS.sha256``
(sha256 per staged file): no image is ever fetched, no object is moved. Depth, GT, sonar and
reference files that live inside ``images/`` are paired to their RGB anchor by stem; a file is
marked ``non_image`` in the row ``attrs``. Without a checksum for the anchor or the reference the
row is *pending* (``image_key`` / ``ref_image_key``). ``ann_id`` ordinals come from the sorted
full set, so they are stable under ``limit``; with ``limit`` evenly spaced groups are kept.
"""

# ruff: noqa: E501

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import PurePosixPath

from ...checksums import parse_checksums
from ..boxes_table import bucket_lister
from ..depth_table import DepthPairResult, DepthPairSource, depth_row
from ..pairs_table import pair_row
from ..s3_keyed import FetchFailed, fetch_small

_SPLITS = {"TR": "train", "VAL": "val", "TE": "test"}


class Tree:
    """The staged ``images/`` listing of one source plus its sha256 index."""

    def __init__(self, spec: DepthPairSource, fetch, lister) -> None:
        self.base = f"sources/{spec.source_id}/{spec.version}/"
        keys = (lister or bucket_lister())(self.base + "images/")
        self.names = {PurePosixPath(k).stem: PurePosixPath(k).name for k in keys}
        try:
            self.sums = parse_checksums(
                (fetch or fetch_small)(self.base + "CHECKSUMS.sha256").decode()
            )
        except (FetchFailed, OSError):
            self.sums = {}  # no CHECKSUMS on the tree: every row is pending

    def sha(self, stem: str) -> str | None:
        return self.sums.get(f"images/{self.names[stem]}")

    def key(self, stem: str) -> str:
        return f"{self.base}images/{self.names[stem]}"


def _pick(items: list, limit: int | None) -> list[tuple[int, object]]:
    """``(ordinal, item)``: evenly spaced ``limit`` of the sorted full list, ordinals from the full set."""
    indexed = list(enumerate(items))
    if limit is None or len(indexed) <= limit:
        return indexed
    step = len(indexed) / limit
    return [indexed[int(i * step)] for i in range(limit)]


def _add_depth(
    res: DepthPairResult, tree: Tree, spec, ordinal: int, anchor: str, depth: str, **kw
) -> None:
    attrs = {**kw.pop("attrs"), "non_image": True, "depth_in_images_dir": True, "depth_stem": depth}
    if (dsha := tree.sha(depth)) is not None:
        attrs["depth_sha256"] = dsha
    sha = tree.sha(anchor)
    row = depth_row(
        spec=spec, ordinal=ordinal, sha=sha, depth_ref=tree.key(depth), attrs=attrs, **kw
    )
    if sha is None:
        res.depth.pending.append(
            {k: v for k, v in row.items() if k != "image_sha256"} | {"image_key": anchor}
        )
    else:
        res.depth.rows.append(row)


def _add_pair(
    res: DepthPairResult, tree: Tree, spec, ordinal: int, anchor: str, ref: str, **kw
) -> None:
    sha, rsha = tree.sha(anchor), tree.sha(ref)
    attrs = {**kw.pop("attrs", {}), "ref_key": tree.key(ref)}
    row = pair_row(spec=spec, ordinal=ordinal, sha=sha, ref_sha=rsha, attrs=attrs, **kw)
    if sha is None or rsha is None:
        res.pairs.pending.append(
            {k: v for k, v in row.items() if k not in ("image_sha256", "ref_image_sha256")}
            | {"image_key": anchor, "ref_image_key": ref}
        )
    else:
        res.pairs.rows.append(row)


def usod10k(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """``USOD10k_USOD10k_<TR|VAL|TE>_{RGB,GT,depth,Boundary}_<n>[_edge]``: RGB <-> depth by (split, n)."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    rx = re.compile(
        r"^USOD10k_USOD10k_(?P<s>TR|VAL|TE)_(?P<k>RGB|GT|depth|Boundary)_(?P<n>\d+)(?:_edge)?$"
    )
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for stem in tree.names:
        if m := rx.match(stem):
            groups[(m["s"], m["n"])][m["k"]] = stem
    anchors = sorted(k for k, g in groups.items() if "RGB" in g)
    res.depth.seen = len(anchors)
    for ordinal, key in _pick(anchors, limit):
        g = groups[key]
        if "depth" not in g:
            res.depth.skipped["no depth file for the RGB frame"] += 1
            continue
        siblings = {k: g[k] for k in ("GT", "Boundary") if k in g}
        _add_depth(res, tree, spec, ordinal, g["RGB"], g["depth"], units="relative", gt_type="estimated",
                   depth_kind="pseudo", split=_SPLITS[key[0]],
                   attrs={"units_basis": "no metric scale stated upstream", "sibling_non_image": siblings})  # fmt: skip
    return res


_SONAR_SUFFIXES = tuple(sorted((
    "enhanced_gray_depth_right_visualize", "enhanced_gray_depth_left_visualize",
    "enhanced_gray_cam_right", "enhanced_gray_cam_left", "cropped_depth_right_visualize",
    "cropped_depth_left_visualize", "cropped_cam_right", "cropped_cam_left", "depth_right_visualize",
    "depth_left_visualize", "cam_right", "cam_left", "sonar_rect_denoise", "sonar_denoise",
    "sonar_rect", "sonar"), key=len, reverse=True))  # fmt: skip
_SONAR_PAIRS = (  # (pair_role, partner suffix, pair_kind)
    ("stereo_right", "cam_right", "stereo"),
    ("enhanced", "enhanced_gray_cam_left", "enhancement"),
    ("sonar", "sonar", "cross-modal"),
)


def sonarsweep(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """``<scene>_<frame>_{cam_left,cam_right,enhanced_gray_cam_left,depth_left_visualize,sonar,...}``:
    every render of one simulated frame shares the prefix; the left camera is the anchor."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    frames: dict[str, dict[str, str]] = defaultdict(dict)
    for stem in tree.names:
        for suffix in _SONAR_SUFFIXES:
            if stem.endswith("_" + suffix):
                frames[stem[: -len(suffix) - 1]][suffix] = stem
                break
    anchors = sorted(f for f, g in frames.items() if "cam_left" in g)
    res.depth.seen = res.pairs.seen = len(anchors)
    per = 1 + len(_SONAR_PAIRS)
    for ordinal, frame in _pick(anchors, None if limit is None else limit // len(_SONAR_PAIRS)):
        g = frames[frame]
        base = ordinal * per
        if "depth_left_visualize" in g:
            _add_depth(res, tree, spec, base, g["cam_left"], g["depth_left_visualize"], units="relative",
                       gt_type="synthetic", depth_kind="gt",
                       attrs={"units_basis": "simulator depth rendered to png (visualisation, not metric)"})  # fmt: skip
        else:
            res.depth.skipped["no depth render for the frame"] += 1
        for k, (role, suffix, kind) in enumerate(_SONAR_PAIRS, start=1):
            if suffix in g:
                _add_pair(res, tree, spec, base + k, g["cam_left"], g[suffix], pair_role=role, pair_kind=kind,
                          attrs={"non_image": suffix == "sonar"})  # fmt: skip
            else:
                res.pairs.skipped[f"no {suffix} for the frame"] += 1
    return res


def auv_sunboat(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """``<campaign>_<timestamp>_{camera,sonar}_<n>``: forward-looking camera <-> FLS frame."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    rx = re.compile(r"^(?P<p>.+)_(?P<k>camera|sonar)_(?P<n>\d+)$")
    groups: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for stem in tree.names:
        if m := rx.match(stem):
            groups[(m["p"], m["n"])][m["k"]] = stem
    anchors = sorted(k for k, g in groups.items() if "camera" in g)
    res.pairs.seen = len(anchors)
    for ordinal, key in _pick(anchors, limit):
        g = groups[key]
        if "sonar" not in g:
            res.pairs.skipped["no sonar frame for the camera frame"] += 1
            continue
        _add_pair(res, tree, spec, ordinal, g["camera"], g["sonar"], pair_role="sonar",
                  pair_kind="cross-modal", attrs={"non_image": False, "ref_modality": "forward-looking sonar"})  # fmt: skip
    return res


def lsui(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """``LSUI_input_<n>`` <-> ``LSUI_GT_<n>``: the input is the anchor, the GT its enhanced reference."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    rx = re.compile(r"^LSUI_(?P<k>GT|input)_(?P<n>\d+)$")
    groups: dict[int, dict[str, str]] = defaultdict(dict)
    for stem in tree.names:
        if m := rx.match(stem):
            groups[int(m["n"])][m["k"]] = stem
    anchors = sorted(n for n, g in groups.items() if "input" in g)
    res.pairs.seen = len(anchors)
    for ordinal, n in _pick(anchors, limit):
        g = groups[n]
        if "GT" not in g:
            res.pairs.skipped["no GT reference for the input"] += 1
            continue
        _add_pair(
            res,
            tree,
            spec,
            ordinal,
            g["input"],
            g["GT"],
            pair_role="enhanced",
            pair_kind="enhancement",
        )
    return res


def viame_habcam(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """HabCam2019 ``3d_models`` samples: ``<id>-{left,right,disp,rectified}``. No CHECKSUMS on the
    tree today, so every row is pending (``image_key`` / ``ref_image_key``)."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    rx = re.compile(
        r"^(?P<id>3d_models_HabCam\d+_dataset\d+_samples_.+?)-(?P<k>left|right|disp|rectified)$"
    )
    groups: dict[str, dict[str, str]] = defaultdict(dict)
    for stem in tree.names:
        if m := rx.match(stem):
            groups[m["id"]][m["k"]] = stem
    anchors = sorted(i for i, g in groups.items() if "left" in g)
    res.depth.seen = res.pairs.seen = len(anchors)
    for ordinal, gid in _pick(anchors, limit):
        g = groups[gid]
        if "disp" in g:
            _add_depth(res, tree, spec, ordinal * 2, g["left"], g["disp"], units="disparity_px", gt_type="stereo",
                       depth_kind="disparity", attrs={"units_basis": "stereo disparity map (HabCam2019 3d_models)"})  # fmt: skip
        else:
            res.depth.skipped["no disparity map for the left frame"] += 1
        if "right" in g:
            _add_pair(res, tree, spec, ordinal * 2 + 1, g["left"], g["right"], pair_role="stereo_right",
                      pair_kind="stereo", attrs={"rectified_stem": g.get("rectified")})  # fmt: skip
        else:
            res.pairs.skipped["no right view for the left frame"] += 1
    return res


def euvp(spec, *, limit=None, fetch=None, lister=None) -> DepthPairResult:
    """One image per parquet row (``data_train-..._parquet_<N>``), the reference column was lost in
    staging: stems allow no pair, so no row is built and the gap is logged (U7)."""
    tree, res = Tree(spec, fetch, lister), DepthPairResult()
    rx = re.compile(r"^data_train-\d+-of-\d+_parquet_(?P<n>\d+)$")
    idx = sorted(int(m["n"]) for stem in tree.names if (m := rx.match(stem)))
    res.pairs.seen = len(idx)
    res.pairs.skipped["reference image not staged"] += len(idx)
    contiguous = bool(idx) and idx == list(range(idx[0], idx[0] + len(idx)))
    res.gaps.append(
        f"euvp: {len(idx)} input images (parquet row index {'contiguous' if contiguous else 'NOT contiguous'}), "
        "the reference column is not staged: 0 pair rows, ref_image_sha256 cannot be resolved (U7 gap)"
    )
    return res


PRODUCERS: dict[str, Callable[..., DepthPairResult]] = {
    "usod10k": usod10k,
    "sonarsweep": sonarsweep,
    "auv_sunboat": auv_sunboat,
    "lsui": lsui,
    "viame_habcam": viame_habcam,
    "euvp": euvp,
}
