"""Integration tests — real data, real layouts.

Marked ``integration`` and skipped unless ``MARINEDATA_INTEGRATION=1``, because they hit
the network. Run them in a nightly job, not on every commit:

    MARINEDATA_INTEGRATION=1 pytest -m integration

**Why these exist, given the synthetic fixtures.** The fixtures prove a *reader* works.
These prove a *declaration* is right. Those fail differently: a reader bug raises, while
a wrong declaration quietly finds nothing or finds the wrong thing. Coralscapes was
declared here as image/mask directories and is in fact HuggingFace parquet with
``image``/``label`` columns — a synthetic fixture would have passed forever.

~100 items per source is enough to catch that and small enough to stay a test rather
than a download.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from marinedata import Registry
from marinedata.fetch import FetchNotSupported, fetch_sample
from marinedata.loaders import build_loader
from marinedata.schema import Axis
from marinedata.verify import unverified, verify_source

pytestmark = pytest.mark.integration

RUN = os.environ.get("MARINEDATA_INTEGRATION") == "1"
requires_network = pytest.mark.skipif(
    not RUN, reason="set MARINEDATA_INTEGRATION=1 to run tests that hit the network"
)

# Sources with a working automated fetch path, each verified against live data.
# Others need a human (gated, request-only, S3, scrape) or are blocked upstream.
FETCHABLE = ["coralscapes", "noaa-pifsc-bleaching", "mouss-detection"]


@pytest.fixture(scope="module")
def sample_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("marinedata-samples")


@requires_network
@pytest.mark.parametrize("source_id", FETCHABLE)
def test_declared_layout_matches_reality(
    registry: Registry, source_id: str, sample_root: Path
) -> None:
    """Fetch ~100 real items and confirm the declared loader reads them."""
    result = verify_source(
        registry.source(source_id),
        registry=registry,
        limit=100,
        root=sample_root / source_id,
    )
    assert result.ok, f"{source_id}: {result.status} — {result.detail}"
    assert result.items > 0


@requires_network
def test_coralscapes_yields_image_mask_pairs(registry: Registry, sample_root: Path) -> None:
    """Concrete expectations for the flagship dense-segmentation source."""
    source = registry.source("coralscapes")
    fetched = fetch_sample(source, limit=20, root=sample_root / "coralscapes-pairs")

    loader = build_loader(source, fetched.root)
    loader.bind_harmonizer(registry.harmonizer_for("coralscapes"))
    samples = list(loader)

    assert len(samples) == fetched.items
    for sample in samples:
        assert sample.image is not None and Path(sample.image).is_file()
        assert sample.mask is not None and Path(sample.mask).is_file()
        assert Path(sample.image).stem == Path(sample.mask).stem
        assert Axis.TAXON in sample.supervised
        assert sample.licence_tier is not None


@requires_network
def test_gated_sources_report_unfetchable_not_failed(registry: Registry) -> None:
    """A human-gated source is not a broken source, and must not read as one."""
    gated = [s for s in registry if s.access.gated or s.access.method.value == "request"]
    if not gated:
        pytest.skip("no gated sources registered")
    with pytest.raises(FetchNotSupported):
        fetch_sample(gated[0], limit=1)


@requires_network
def test_fetch_is_bounded(registry: Registry, sample_root: Path) -> None:
    """A verification fetch must never turn into a bulk download."""
    fetched = fetch_sample(registry.source("coralscapes"), limit=5, root=sample_root / "bounded")
    assert fetched.items <= 5
    assert fetched.truncated is True


# ── these run without the network ─────────────────────────────────────────


@pytest.mark.parametrize("source_id", FETCHABLE)
def test_fetchable_sources_declare_fetch_params(registry: Registry, source_id: str) -> None:
    """A source we claim to fetch automatically must carry the parameters to do so."""
    access = registry.source(source_id).access
    assert access.params, f"{source_id}: no access.params — the fetcher cannot resolve it"


def test_unverified_layouts_are_reported_not_hidden(registry: Registry) -> None:
    """Unverified layouts are allowed — silently unverified ones are not.

    This test documents the current backlog rather than failing on it. Tighten the
    bound as sources get verified; do not delete the visibility.
    """
    pending = unverified(registry)
    loadable = [s for s in registry if s.loader and s.loader.layout != "metadata-only"]
    assert len(pending) <= len(loadable), "unverified set cannot exceed loadable set"
    # Surfaced in the report so the number is visible in CI output.
    print(f"\n{len(pending)}/{len(loadable)} loadable sources have unverified layouts")
