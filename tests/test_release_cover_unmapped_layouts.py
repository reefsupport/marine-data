"""WP-R2e/R2f: no releasable source is silently skipped, and the map never depends on a layout.

flat-images and image-mask-pairs sources are split-mapped (a pair shares its primary image's
group) and ship in the flavour build; so do image-folder and coco-json sources (WP-R2f: the map
assigns IMAGES, so no loader layout is a skip reason). A source the map still skips with rows > 0
(a registry ``release_skip_reason`` or an uncited ``needs-attribution``) fails the build unless it
is named in ``allow_skip`` or carries the registry reason; ``marinedata release split-map`` runs
end to end.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import pytest
from PIL import Image
from test_global_split_map_flavours import NOW, _manifest, _registry, _src
from test_upstream_test_split import _stage

from marinedata.cli import main
from marinedata.loaders.generic import IMAGE_SUFFIXES
from marinedata.models import LoaderSpec
from marinedata.release import (
    ReleaseSkipError,
    build_release,
    generate_split_map,
    release_skip_reason,
    source_release_entries,
)
from marinedata.splitmap import load_split_map

N = 40


def _png(seed: str) -> bytes:
    """A distinct noise PNG (the CLI's near-dup pass dHashes every image)."""
    rng = random.Random(seed)
    image = Image.frombytes("L", (32, 32), bytes(rng.randrange(256) for _ in range(32 * 32)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _flat(root: Path, prefix: str) -> Path:
    for i in range(N):
        path = root / "frames" / f"clip{i % 14}" / f"{prefix}{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(f"{prefix}-image-{i}"))
    return root


def _pairs(root: Path, prefix: str) -> Path:
    for i in range(N):
        for sub, tag in (("images", "img"), ("masks", "mask")):
            path = root / sub / f"{prefix}{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_png(f"{prefix}-{tag}-{i}"))
    return root


def _staged_rows(prefix: str) -> list:  # type: ignore[no-untyped-def]
    return [
        (f"{prefix}{g}{k}", f"{prefix}/g{g}", "train", f"{prefix}-{g}{k}".encode())
        for g in range(12)
        for k in "ab"
    ]


def _fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    registry = _registry(
        [
            _src("src-open", "open"),
            _src("src-flat", "restricted-nc", layout="flat-images"),
            _src("src-pairs", "restricted-nc", layout="image-mask-pairs"),
        ]
    )
    roots = {
        "src-open": _stage(tmp_path / "open", _staged_rows("o")),
        "src-flat": _flat(tmp_path / "flat", "f"),
        "src-pairs": _pairs(tmp_path / "pairs", "p"),
    }
    return registry, roots


def _gen(registry, roots, out: Path):  # type: ignore[no-untyped-def]
    skipped: dict[str, str] = {}
    generate_split_map(
        registry,
        out=out,
        roots=roots,
        profile="ship-noncommercial",
        now=NOW,
        seed=0,
        min_groups=3,
        skipped=skipped,
    )
    return skipped


def test_flat_and_pair_sources_are_split_mapped_and_ship_with_pairs_whole(tmp_path: Path) -> None:
    registry, roots = _fixture(tmp_path)
    split_map = tmp_path / "SPLIT_MAP.json"
    assert _gen(registry, roots, split_map) == {}  # nothing skipped any more
    groups = load_split_map(split_map).assignments  # type: ignore[union-attr]
    pairs = source_release_entries(registry.source("src-pairs"), roots["src-pairs"])
    flat = source_release_entries(registry.source("src-flat"), roots["src-flat"])
    assert len(pairs) == len(flat) == N
    assert all(e.group in groups for e in (*pairs, *flat))
    assert all(e.path.parent.name == "images" for e in pairs)  # a mask is never its own row

    out = tmp_path / "rel"
    result = build_release(
        registry,
        release="r1",
        split_map=split_map,
        roots=roots,
        out_dir=out,
        profile="ship-noncommercial",
        flavour="nc",
        allow_unmapped=True,
    )
    assert set(result.sources) == {"src-flat", "src-pairs"}  # the nc delta
    manifest = _manifest(out, "nc")
    pretrain = {"train": "train", "val": "probe"}  # the self-supervised task drops `test`
    for entry in (*pairs, *flat):  # an image and its mask ship together or not at all
        assert manifest.get(entry.sha256) == pretrain.get(groups[entry.group])
    mask_shas = {
        hashlib.sha256(p.read_bytes()).hexdigest() for p in (roots["src-pairs"] / "masks").glob("*")
    }
    assert len(mask_shas) == N and not mask_shas & set(manifest)
    assert {groups[e.group] for e in (*pairs, *flat)} >= {"train"}


def _image_folder(root: Path, prefix: str) -> Path:
    for i in range(N):
        path = root / ("coral" if i % 2 else "sand") / f"{prefix}{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(f"{prefix}-folder-{i}"))
    return root


def _coco(root: Path, prefix: str) -> Path:
    """``annotations.json`` + ``images/<n>.png``; ``file_name`` is relative to ``images``, so
    ``Sample.key`` (``file_name``) is NOT the path relative to the root."""
    images = []
    for i in range(N):
        path = root / "images" / f"{prefix}{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_png(f"{prefix}-coco-{i}"))
        images.append({"id": i, "file_name": path.name, "width": 32, "height": 32})
    doc = {"images": images, "annotations": [], "categories": [{"id": 1, "name": "fish"}]}
    (root / "annotations.json").write_text(json.dumps(doc))
    return root


def test_image_folder_and_coco_json_sources_are_split_mapped_and_ship(tmp_path: Path) -> None:
    """The split map is layout-agnostic: an image-folder and a coco-json source are enumerated
    from their image tree, appear in the map, and ship in the flavour build with the group the
    map was generated from (the loader's own key is not the root-relative path)."""
    registry = _registry(
        [
            _src("src-open", "open"),
            _src("src-folder", "restricted-nc", layout="image-folder"),
            _src("src-coco", "restricted-nc", layout="coco-json"),
        ]
    )
    roots = {
        "src-open": _stage(tmp_path / "open", _staged_rows("o")),
        "src-folder": _image_folder(tmp_path / "folder", "d"),
        "src-coco": _coco(tmp_path / "coco", "c"),
    }
    split_map = tmp_path / "SPLIT_MAP.json"
    assert _gen(registry, roots, split_map) == {}  # nothing skipped on layout
    groups = load_split_map(split_map).assignments  # type: ignore[union-attr]
    entries = {
        sid: source_release_entries(registry.source(sid), roots[sid])
        for sid in ("src-folder", "src-coco")
    }
    assert all(len(e) == N for e in entries.values())
    assert all(x.group in groups for e in entries.values() for x in e)

    out = tmp_path / "rel"
    result = build_release(
        registry,
        release="r1",
        split_map=split_map,
        roots=roots,
        out_dir=out,
        profile="ship-noncommercial",
        flavour="nc",
        allow_unmapped=True,
    )
    assert set(result.sources) == {"src-folder", "src-coco"}  # the nc delta
    manifest = _manifest(out, "nc")
    pretrain = {"train": "train", "val": "probe"}
    for entry in (*entries["src-folder"], *entries["src-coco"]):
        assert manifest.get(entry.sha256) == pretrain.get(groups[entry.group])
    assert {manifest.get(x.sha256) for e in entries.values() for x in e} >= {"train"}


def test_skipped_source_with_rows_fails_unless_allowed(tmp_path: Path) -> None:
    registry, roots = _fixture(tmp_path)
    nc = _src("src-skip", "restricted-nc").model_copy(update={"tags": ("needs-attribution",)})
    registry = _registry([*registry.sources, nc])
    roots["src-skip"] = _stage(tmp_path / "skip", _staged_rows("s"))  # rows > 0, uncited
    split_map = tmp_path / "SPLIT_MAP.json"
    assert list(_gen(registry, roots, split_map)) == ["src-skip"]
    kwargs = dict(
        release="r1",
        split_map=split_map,
        roots=roots,
        out_dir=tmp_path / "rel",
        profile="ship-noncommercial",
        flavour="nc",
        allow_unmapped=True,
    )
    with pytest.raises(ReleaseSkipError, match="src-skip"):
        build_release(registry, **kwargs)  # type: ignore[arg-type]
    build_release(registry, allow_skip=["src-skip"], **kwargs)  # type: ignore[arg-type]
    release = json.loads((tmp_path / "rel/releases/r1/nc/RELEASE.json").read_text())
    assert release["skipped_sources"] == [
        {
            "id": "src-skip",
            "reason": release_skip_reason(registry.source("src-skip")),
            "allowed_by": "allow-skip",
        }
    ]
    # a registry release_skip_reason also allows it, and the reason is kept
    why = registry.source("src-skip").model_copy(update={"release_skip_reason": "labels TBD"})
    registry = _registry([*[s for s in registry.sources if s.id != "src-skip"], why])
    build_release(registry, **kwargs)  # type: ignore[arg-type]
    release = json.loads((tmp_path / "rel/releases/r1/nc/RELEASE.json").read_text())
    assert release["skipped_sources"][0]["allowed_by"] == "registry"
    assert "labels TBD" in release["skipped_sources"][0]["reason"]


def test_needs_attribution_takes_registry_citation_else_stays_skipped() -> None:
    tagged = _src("src-attr", "open").model_copy(update={"tags": ("needs-attribution",)})
    assert "needs-attribution" in (release_skip_reason(tagged) or "")
    cited = tagged.model_copy(update={"citation": "Doe 2020", "homepage": "https://x.org"})
    assert release_skip_reason(cited) is None
    for layout in ("coco-json", "image-folder", "yolo-txt", "metadata-only"):  # never a skip
        loader = LoaderSpec(layout=layout, params={})
        assert release_skip_reason(cited.model_copy(update={"loader": loader})) is None


def test_release_split_map_cli_writes_once_and_never_overwrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    registry, roots = _fixture(tmp_path)
    roots = {sid: r for sid, r in roots.items() if sid != "src-open"}  # fake-bytes tree
    monkeypatch.setattr("marinedata.cli_release.Registry.load", lambda *a, **k: registry)
    monkeypatch.setattr("marinedata.cli_release.SPLIT_MAP_PROFILE", "ship-noncommercial")
    argv = [
        "release",
        "split-map",
        "--release",
        "r1",
        "--split-map",
        str(tmp_path / "M.json"),
        "--min-groups",
        "3",
        "--local-only",
        "--no-near-dup",
    ]
    for sid, root in roots.items():
        argv += ["--local", f"{sid}={root}"]
    assert main(argv) == 0
    written = load_split_map(tmp_path / "M.json")
    assert written is not None and len(written.assignments) > 12
    assert main(argv) == 1
    assert "never overwritten" in capsys.readouterr().err
    assert IMAGE_SUFFIXES  # imported for the walk-layout fixtures' suffix


_REASONS = ("context-layer", "non-image-modality", "no-staged-images")


def test_every_releasable_registry_source_is_covered_or_carries_an_explicit_reason() -> None:
    """The R2e gate passes for a full dry run without ``--allow-skip`` (WP-R2f): a releasable
    source is enumerated (any loader layout) or skipped with a registry ``release_skip_reason``
    naming one of the triage reasons; ``metadata-only`` never means "covered"."""
    from marinedata.licence_class import release_excluded, source_class
    from marinedata.registry import Registry

    registry = Registry.load()
    for source in registry.sources:
        if source_class(source.id) not in ("open", "restricted-nc") or release_excluded(source.id):
            continue
        reason = release_skip_reason(source)
        layout = source.loader.layout if source.loader is not None else None
        if reason is not None:
            assert source.release_skip_reason, (source.id, reason)  # never a layout skip
            assert source.release_skip_reason.split(":")[0] in _REASONS, source.id
        assert reason is not None or layout != "metadata-only", (source.id, "metadata-only")
