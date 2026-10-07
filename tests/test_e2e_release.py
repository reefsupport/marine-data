"""D12 (5star-rubric): a network-free, end-to-end fixture run of the real pipeline —
ingest/stage -> release build -> HF export — asserting the sha256 of every output
byte-for-byte against a golden manifest below. If any output changes (a dependency
bump, an algorithm tweak, a Pillow upgrade), this test fails; that is the point.

The corpus is two tiny fixture sources (~15 real PNGs each, no masks) committed under
``tests/fixtures/e2e_release/``. Ingest zips them in-memory (mimicking a real upstream
archive) and stages them through the real :func:`marinedata.ingest._stage_with_plan` —
``fetch.get_bytes`` is monkeypatched so nothing touches the network. Images are
lossless PNG (no JPEG re-encode variance) so the release build's LANCZOS near-dup
dHash is stable across machines.
"""

from __future__ import annotations

import io
import re
import time
import zipfile
from datetime import date
from pathlib import Path

import pytest

from marinedata import fetch
from marinedata.cli import main
from marinedata.enums import (
    AccessClass,
    AccessMethod,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from marinedata.hf_export import build_layout, collect_rows, export
from marinedata.ingest import ArchivePlan, _stage_with_plan
from marinedata.models import (
    Access,
    Coverage,
    Licence,
    LoaderSpec,
    Profile,
    Source,
    SplitGroupRule,
    Verification,
)
from marinedata.registry import Registry
from marinedata.schema import Axis, LabelSchema
from marinedata.task import TaskKind, TaskSpec

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "e2e_release"
SOURCES = ("e2e-src-a", "e2e-src-b")


def _never_match() -> re.Pattern[str]:
    return re.compile(r"^__never_matches__$")


def _build_zip(images_dir: Path) -> bytes:
    """A tiny in-memory archive mimicking ``ROOT/images/<stem>.png`` for one source —
    built from the real, committed fixture bytes on disk (never generated here)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(images_dir.glob("*.png")):
            zf.writestr(f"ROOT/images/{path.name}", path.read_bytes())
    return buf.getvalue()


def _source(source_id: str) -> Source:
    return Source(
        id=source_id,
        name=source_id,
        description="D12 e2e fixture source (WP-4).",
        version="v1",
        licence=Licence(id="CC0-1.0", name="CC0 1.0", tier=Tier.PERMISSIVE),
        access_class=AccessClass.OPEN,
        verification=Verification(
            verified_on=date(2026, 9, 25), verified_by="wp4-e2e-fixture", method="licence-file"
        ),
        legal_basis=LegalBasis.LICENCE,
        provenance=Provenance.PUBLIC,
        access=Access(
            method=AccessMethod.HTTP,
            uri="https://example.invalid",
            params={"sample_url": f"https://example.invalid/{source_id}.zip"},
        ),
        modalities=(Modality.IMAGE,),
        capabilities=(Capability.BENTHIC_SEGMENTATION,),
        coverage=Coverage(regions=(Region.GLOBAL,)),
        # "staged-tree" (not the ingest-time archive layout) is what the release
        # builder's generic reader expects once a source is staged — see
        # ``release.py``'s own docstrings.
        loader=LoaderSpec(layout="staged-tree", params={}),
        annotations=(),
        # One group per stem (not the default one-per-source) so the 15 fixture
        # images give the splitter enough groups for a real, non-empty split.
        split_group=SplitGroupRule(
            pattern=r"(.+)", match_field="stem", template="{source_id}/{group}"
        ),
    )


def _registry() -> Registry:
    # DatasetBuilder's default schema_id ("rs-benthic-v1") is what a schema-less
    # (self-supervised) task resolves to (see hf_export.DEFAULT_SCHEMA_ID) — match it.
    schema = LabelSchema(id="rs-benthic-v1", name="Fixture", axes=(Axis.TAXON,), nodes=())
    tasks = {"pretrain-set": TaskSpec(id="pretrain-set", kind=TaskKind.SELF_SUPERVISED)}
    profile = Profile(
        id="ship-open",
        description="fixture",
        allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT),
        allow_access_classes=("open",),
    )
    return Registry(
        sources={sid: _source(sid) for sid in SOURCES},
        licences={},
        profiles={
            "ship-open": profile,
            "ship-noncommercial": Profile(
                id="ship-noncommercial",
                description="fixture",
                allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
                allow_access_classes=("open", "restricted-nc"),
                public_release=True,
            ),
        },
        schemas={"rs-benthic-v1": schema},
        tasks=tasks,
    )


def _stage_all(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Real ingest (D1 §7): zip the committed fixtures, monkeypatch the network fetch,
    and run every source through ``_stage_with_plan`` — no annotations, so the mask
    path (and its Pillow re-encode) never runs; images are copied byte-for-byte."""
    roots: dict[str, Path] = {}
    for source_id in SOURCES:
        images_dir = FIXTURE_ROOT / source_id.replace("e2e-src-", "source-")
        n = len(list(images_dir.glob("*.png")))
        zip_bytes = _build_zip(images_dir)
        monkeypatch.setattr(fetch, "get_bytes", lambda url, _b=zip_bytes, **kw: _b)
        plan = ArchivePlan(
            image_pattern=re.compile(r"^ROOT/(?P<split>images)/(?P<stem>[^/]+)\.png$"),
            mask_pattern=_never_match(),
            expected_images=n,
            expected_masks=0,
            partition="default",
            version="v1",
            classes=0,
            license_text="CC0 1.0 (fixture)\n",
        )
        result = _stage_with_plan(
            _source(source_id),
            plan,
            cache_root=tmp_path / "cache" / source_id,
            out_root=tmp_path / "staged",
            profile=Profile(
                id="research",
                description="fixture",
                allow_tiers=(Tier.OWN, Tier.PERMISSIVE, Tier.COPYLEFT, Tier.NONCOMMERCIAL),
            ),
        )
        assert result.images == n
        roots[source_id] = result.root
    return roots


_VOLATILE_JSON_KEYS = {"generated_at", "split_map_sha256"}
"""Fields whose value is a real wall-clock timestamp (or a hash derived from one) —
legitimately different on every run, not a content regression. Blanked out before
hashing so the golden manifest checks everything else byte-for-byte."""


def _blank_volatile(value: object) -> object:
    if isinstance(value, dict):
        return {
            k: ("<normalized>" if k in _VOLATILE_JSON_KEYS else _blank_volatile(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_blank_volatile(v) for v in value]
    return value


def _canonical_bytes(path: Path) -> bytes:
    """Raw bytes for most files; for ``.json`` outputs, a canonical form with
    ``_VOLATILE_JSON_KEYS`` blanked so a real generation timestamp never fails the
    golden-hash check."""
    import json

    if path.suffix == ".json":
        data = _blank_volatile(json.loads(path.read_text()))
        return json.dumps(data, sort_keys=True).encode()
    return path.read_bytes()


def _sha256_tree(root: Path) -> dict[str, str]:
    import hashlib

    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = hashlib.sha256(
                _canonical_bytes(path)
            ).hexdigest()
    return out


# Golden manifest: relative path -> sha256, for every file this test produces. A
# change here means a release output changed — regenerate deliberately, never blindly.
GOLDEN_RELEASE = {
    "releases/wp4-e2e/open/RELEASE.json": (
        "b18b4eb8623dc7273c659cf708e607af3233d5f5cf0c53bb92d4b55d60d6aeea"
    ),
    "SPLIT_MAP.json": "383a640ac3afec6cf8c961388261d1a4fe0306e3fd90b6c26e928e6a513a4a48",
    "releases/wp4-e2e/open/tasks/pretrain-set.tsv": (
        "6ea2774662feb0d80e5737b24d66d9159f0c208969c1cdac1cdbe7568212dc53"
    ),
}

GOLDEN_EXPORT = {
    "data/images/train-00000-of-00001.parquet": (
        "c20ee71a549cc1427799fd4e80ea19e77de46b46e1cbe157ae5719e3bae006ee"
    ),
    "data/images/validation-00000-of-00001.parquet": (
        "c120243a030be8b83dc9637b78db1e8267ea5e4168e888d48edd30504ba49aca"
    ),
    "data/pretrain-set/train-00000-of-00001.parquet": (
        "27b4ec35913e97cb6b6cc48f95204e5060f10ed06ca06145b7ff53035648f49f"
    ),
    "data/pretrain-set/validation-00000-of-00001.parquet": (
        "3010aa1c52437c2ecbac30e495af5da2479cbe3c22b92ddae1e57e6d8266ab79"
    ),
}


def test_e2e_ingest_release_hf_export_is_reproducible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = time.monotonic()
    registry = _registry()
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls, root=None: registry))
    monkeypatch.setenv("MARINEDATA_CACHE", str(tmp_path / "dhash-cache"))

    roots = _stage_all(tmp_path, monkeypatch)

    out_dir = tmp_path / "release"
    split_map = out_dir / "SPLIT_MAP.json"
    code = main(
        [
            "release",
            "build",
            "--release",
            "wp4-e2e",
            "--split-map",
            str(split_map),
            "--out",
            str(out_dir),
            "--flavour",
            "open",
            "--profile",
            "ship-open",
            "--generate-split-map",
            "--no-near-dup",
            "--seed",
            "0",
            *[f"--local={sid}={root}" for sid, root in roots.items()],
        ]
    )
    assert code == 0

    got_release = _sha256_tree(out_dir)
    for rel_path, expected in GOLDEN_RELEASE.items():
        assert rel_path in got_release, f"missing release output {rel_path}"
        assert got_release[rel_path] == expected, f"{rel_path} changed: {got_release[rel_path]}"

    release_dir = out_dir / "releases" / "wp4-e2e" / "open"
    rows = collect_rows(registry, roots, release_dir, "ship-open")
    layout = build_layout(rows)
    export_dir = tmp_path / "hf_export"
    export(layout, export_dir)

    got_export = _sha256_tree(export_dir)
    assert got_export == GOLDEN_EXPORT, f"hf_export output changed: {sorted(got_export)}"

    elapsed = time.monotonic() - started
    assert elapsed < 60, f"e2e fixture run took {elapsed:.1f}s, must stay under 60s"
