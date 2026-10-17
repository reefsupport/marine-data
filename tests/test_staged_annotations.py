"""``--v11`` builds the task tables from the staged trees: overlay, local fetch, build wiring."""

from __future__ import annotations

import contextlib
import hashlib
import io
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from test_release import _two_source_fixture, _write_map

from marinedata.checksums import CHECKSUM_FILE, file_digest, parse_checksums
from marinedata.registry import Registry
from marinedata.release import build_release
from marinedata.release.build_helpers import _memoized
from marinedata.task_layers import seaview_points as S
from marinedata.task_layers import staged_annotations as SA
from marinedata.task_layers.configs import build_all_configs, resolve_tasklabels_root
from marinedata.task_layers.s3_keyed import FetchFailed, ImageFetchRefused

SEAVIEW = SA.SEAVIEW
DATA = Path(__file__).resolve().parents[1] / "data"

# The tree of tests/test_seaview_points.py (copied, not imported: that module imports the CLI,
# which needs the ``ingest`` extra): three quadrats with points, one image with none, one id with
# no image, and a ``labelled_data`` twin of Q1 that never keys a point.
REGION = "TESTREG"
Q1, Q2, Q3, Q4, ORPHAN = "11111111111", "22222222222", "33333333333", "44444444444", "55555555555"
SIZES = {Q1: (40, 30), Q2: (64, 48), Q3: (32, 32), Q4: (16, 16)}
HEADER = "quadratid,y,x,label_name,label,func_group,method,data_set\n"
POINTS = [  # (quadrat, y, x, label_name, label, func_group, data_set)
    (Q1, 10, 20, "Hard coral", "EAM_DHC", "Hard coral", "train"),
    (Q1, 15, 5, "Sand", "LSUB_SAND", "Substrate", "train"),
    (Q1, 500, -3, "Mystery", "ZZZ_not_a_label", "Other", "train"),  # outside the frame, unmapped
    (Q2, 3, 4, "Dead substrate", "DSUB", "Substrate", "test"),
    (Q2, 7, 8, "Algal mat", "AMAT", "Algae", "test"),
    (Q3, 9, 9, "Recently bleached", "EAM_RB", "Hard coral", "train"),
    (ORPHAN, 1, 1, "Sand", "LSUB_SAND", "Substrate", "train"),
    (ORPHAN, 2, 2, "Sand", "LSUB_SAND", "Substrate", "train"),
]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _jpeg(quadrat: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", SIZES[quadrat], (int(quadrat[0]) * 20, 90, 140)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def registry() -> Registry:
    return Registry.load()


def _write_tree(root: Path, points=POINTS) -> dict[str, str]:
    """A seaview staged tree as the VM caches it: no CHECKSUMS.sha256, ``{quadrat: sha256}``."""
    jpgs = {q: _jpeg(q) for q in SIZES}
    shas = {q: _sha(b) for q, b in jpgs.items()}
    twin = jpgs[Q1] + b"\0"  # the labelled_data copy of Q1: same stem, other bytes
    files = {f"images/{REGION}/{q}.jpg": b for q, b in jpgs.items()}
    files[f"images/{S.LABELLED_PARTITION}/{Q1}.jpg"] = twin
    meta = [
        {"partition": REGION, "stem": q, "image_path": f"images/{REGION}/{q}.jpg",
         "image_sha256": shas[q], "width": SIZES[q][0], "height": SIZES[q][1]}
        for q in SIZES
    ]  # fmt: skip
    meta.append(
        {"partition": S.LABELLED_PARTITION, "stem": Q1, "width": 40, "height": 30,
         "image_path": f"images/{S.LABELLED_PARTITION}/{Q1}.jpg", "image_sha256": _sha(twin)}
    )  # fmt: skip
    csv = HEADER + "".join(
        f"{q},{y},{x},{name},{code},{group},random,{split}\n"
        for q, y, x, name, code, group, split in points
    )
    files[f"{S.TABLES}annotations_{REGION}.csv"] = csv.encode()
    files["INGEST.json"], files["LICENSE"] = b"{}", b"CC BY 3.0\n"
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    pq.write_table(pa.Table.from_pylist(meta), root / "metadata.parquet")
    return shas


@pytest.fixture
def seaview(tmp_path):
    root = tmp_path / "cache" / SEAVIEW
    return SimpleNamespace(root=root, shas=_write_tree(root))


def test_overlay_adds_seaview_points_and_keeps_the_tasklabels_rows(registry, seaview):
    base = resolve_tasklabels_root(registry)  # <repo>/data: bleaching, semseg, vqa, points
    plain = build_all_configs(registry, base)
    assert plain["bleaching"].rows and plain["semseg"].rows  # the guard: there is something to keep
    with SA.staged_overlay(registry, base, {SEAVIEW: seaview.root}) as overlay:
        staged = build_all_configs(registry, overlay.root)
        (layer,) = overlay.layers
    assert not overlay.root.exists()  # the temporary overlay is gone
    # (a) every config but points is byte-for-byte what it is without the overlay
    assert {cid: r.rows for cid, r in staged.items() if cid != "points"} == {
        cid: r.rows for cid, r in plain.items() if cid != "points"
    }
    # (b) points = the old rows + the seaview rows, keyed by image_sha256
    added = [r for r in staged["points"].rows if r["source_id"] == SEAVIEW]
    assert len(added) == 6 and {r["image_sha256"] for r in added} == {
        seaview.shas[q] for q in (Q1, Q2, Q3)
    }
    assert {r["source_id"] for r in plain["points"].rows}.isdisjoint({SEAVIEW})
    assert tuple(r for r in staged["points"].rows if r["source_id"] != SEAVIEW) == (
        plain["points"].rows
    )
    assert (layer.table, layer.source_id, layer.rows, layer.images, layer.reused) == (
        "points", SEAVIEW, 6, 3, False,
    )  # fmt: skip


def test_fetch_maps_pinned_keys_serves_disk_checksums_and_hashes_each_file_once(seaview):
    calls: Counter[Path] = Counter()

    def counting(path: Path) -> str:
        calls[path] += 1
        return file_digest(path)

    memo = _memoized(counting)  # the build's digest
    for image in sorted((seaview.root / "images").rglob("*.jpg")):
        memo(image)  # the metadata spine's pass over the images
    fetch = SA.StagedFetch({SEAVIEW: seaview.root}, memo)
    key = f"sources/{SEAVIEW}/2026-10-06/"
    sums = parse_checksums(fetch(key + CHECKSUM_FILE).decode())
    fetch(key + CHECKSUM_FILE)
    on_disk = {
        p.relative_to(seaview.root).as_posix() for p in seaview.root.rglob("*") if p.is_file()
    }
    assert set(sums) == on_disk and CHECKSUM_FILE not in sums
    assert all(sums[rel] == _sha((seaview.root / rel).read_bytes()) for rel in sums)
    assert set(calls.values()) == {1}  # the images the spine hashed are not hashed again
    assert fetch(key + "metadata.parquet") == (seaview.root / "metadata.parquet").read_bytes()
    with pytest.raises(ImageFetchRefused):
        fetch(key + f"images/{REGION}/{Q1}.jpg")
    for missing in (key + "labels/none.csv", key + "../x", f"sources/other/v/{CHECKSUM_FILE}"):
        with pytest.raises(FetchFailed):
            fetch(missing)


def test_a_tree_without_a_producer_is_ignored(registry, tmp_path):
    other = tmp_path / "coralscapes"
    other.mkdir()

    def never(path: Path) -> str:
        raise AssertionError(f"hashed {path}")

    base = resolve_tasklabels_root(registry)
    with SA.staged_overlay(registry, base, {"coralscapes": other}, digest=never) as overlay:
        assert overlay.layers == ()
        assert not (overlay.root / SA.ANNOTATIONS_DIR / "points").exists()
        assert (overlay.root / "_tasklabels").resolve() == (base / "_tasklabels").resolve()


def test_existing_annotations_win_and_the_base_root_is_never_written(registry, seaview, tmp_path):
    base = tmp_path / "base"
    (base / "_tasklabels").mkdir(parents=True)
    with SA.staged_overlay(registry, base, {SEAVIEW: seaview.root}) as first:
        (made,) = first.layers
        kept = base / "_annotations" / "points" / SEAVIEW / "hand-staged.parquet"
        kept.parent.mkdir(parents=True)
        shutil.copyfile(
            first.root / "_annotations" / "points" / SEAVIEW / f"{made.version}.parquet", kept
        )
    before = sorted(p.relative_to(base) for p in base.rglob("*"))
    with SA.staged_overlay(registry, base, {SEAVIEW: seaview.root}) as second:
        (layer,) = second.layers
        files = sorted(
            p.name for p in (second.root / "_annotations" / "points" / SEAVIEW).iterdir()
        )
    assert layer.reused and layer.rows == made.rows and files == ["hand-staged.parquet"]
    assert sorted(p.relative_to(base) for p in base.rglob("*")) == before


def test_a_producer_that_yields_no_rows_fails_closed(registry, tmp_path):
    root = tmp_path / SEAVIEW
    _write_tree(root, points=[p for p in POINTS if p[0] not in (Q1, Q2, Q3)])  # ids with no image
    with pytest.raises(SA.StagedLayerError, match=f"points/{SEAVIEW}"):  # noqa: SIM117
        with SA.staged_overlay(registry, DATA, {SEAVIEW: root}):
            pytest.fail("the overlay must not be handed out empty")


def test_build_release_builds_the_overlay_only_with_v11(monkeypatch, tmp_path):
    registry, roots = _two_source_fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    _write_map(split_map, {"shared/dup": "val", "src-a/only": "train", "src-b/only": "test"})
    seen: dict[str, object] = {}

    @contextlib.contextmanager
    def fake_overlay(reg, base_dir, admitted, *, digest):
        seen["overlay"] = (Path(base_dir), dict(admitted), digest)
        yield SimpleNamespace(root=tmp_path / "the-overlay", layers=())

    class Stop(Exception):
        pass

    def fake_configs(reg, base_dir, flavour=None, *, v11=None):
        seen["configs"] = (Path(base_dir), v11 is not None)
        if v11 is not None:
            raise Stop
        return {}

    monkeypatch.setattr(SA, "staged_overlay", fake_overlay)
    monkeypatch.setattr("marinedata.task_layers.configs.build_all_configs", fake_configs)
    args = dict(release="r7k", split_map=split_map, roots=roots, tasks="v2", tasklabels_root=DATA)

    build_release(registry, out_dir=tmp_path / "plain", **args)
    assert "overlay" not in seen and seen["configs"] == (DATA, False)  # (c) nothing changes

    with pytest.raises(Stop):
        build_release(registry, out_dir=tmp_path / "v11", v11=True, allow_unsplit=True, **args)
    base, admitted, digest = seen["overlay"]
    assert base == DATA and admitted == {k: Path(v) for k, v in roots.items()} and callable(digest)
    assert seen["configs"] == (tmp_path / "the-overlay", True)  # the configs read the overlay
