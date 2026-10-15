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
from conftest import make_source

from marinedata import Registry
from marinedata.fetch import FetchError, FetchNotSupported, FetchResult, fetch_sample
from marinedata.loaders import LoaderError, build_loader
from marinedata.schema import Axis
from marinedata.verify import VerifyResult, unverified, verify_source

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


def _assert_layout_verified(result: VerifyResult) -> None:
    """Skip on a network/fetch failure; fail on a genuine layout mismatch.

    ``verify_source`` already tags the two differently: a ``FetchError`` (host
    unreachable, timeout, 5xx, DNS...) lands as ``status="failed"`` with a
    ``"fetch: "``-prefixed detail, while a real layout problem (``LoaderError``, or a
    loader that yields nothing) does not carry that prefix. Only the former is safe to
    treat as "try again later" — an upstream outage is not a declaration bug. The
    latter is exactly the signal this job exists to carry, and must not be trained
    around.
    """
    if result.status == "failed" and result.detail.startswith("fetch:"):
        pytest.skip(f"{result.source_id}: network/fetch failure — {result.detail}")
    assert result.ok, f"{result.source_id}: {result.status} — {result.detail}"
    assert result.items > 0


@requires_network
@pytest.mark.parametrize("source_id", FETCHABLE)
def test_declared_layout_matches_reality(
    registry: Registry, source_id: str, sample_root: Path
) -> None:
    """Fetch ~100 real items and confirm the declared loader reads them.

    A network/fetch failure skips (upstream blip, not our bug); a fetch that succeeds
    against a wrong declared layout fails. See ``_assert_layout_verified``.
    """
    result = verify_source(
        registry.source(source_id),
        registry=registry,
        limit=100,
        root=sample_root / source_id,
    )
    _assert_layout_verified(result)


@requires_network
def test_coralscapes_yields_image_mask_pairs(registry: Registry, sample_root: Path) -> None:
    """Concrete expectations for the flagship dense-segmentation source."""
    source = registry.source("coralscapes")
    fetched = fetch_sample(source, limit=20, root=sample_root / "coralscapes-pairs")

    # A sampled root is legitimately incomplete: metadata.parquet names every image,
    # the sample holds 20.
    loader = build_loader(source, fetched.root, partial=fetched.truncated)
    loader.bind_harmonizer(registry.harmonizer_for("coralscapes"))
    samples = list(loader)

    images = [p for p in (fetched.root / "images").rglob("*") if p.is_file()]
    assert len(samples) == len(images) > 0
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
    # `items` counts every file the s3-manifest route fetched (metadata, shared and per-image
    # labels, images), so the bound that matters is the number of images on disk.
    images = [p for p in (fetched.root / "images").rglob("*") if p.is_file()]
    assert 0 < len(images) <= 5
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


def test_network_failure_skips_rather_than_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dead upstream must skip the layout check, not redden the nightly run.

    Injects a ``FetchError`` at the point ``verify_source`` calls ``fetch_sample`` —
    the same exception a real DNS failure, timeout or 5xx would raise — and drives it
    through the real ``verify_source``/``_assert_layout_verified`` path. No network.
    """
    source = make_source("flat-images", source_id="offline-fixture")

    def _unreachable(*args: object, **kwargs: object) -> FetchResult:
        raise FetchError("Name or service not known")

    monkeypatch.setattr("marinedata.verify.fetch_sample", _unreachable)

    result = verify_source(source, limit=5)
    assert result.status == "failed" and result.detail.startswith("fetch:")
    with pytest.raises(pytest.skip.Exception):
        _assert_layout_verified(result)


def test_layout_mismatch_fails_rather_than_skips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A fetch that succeeds against a wrong declared layout must fail CI.

    The fetch itself is faked (no network); only the loader raises, the same way a
    real declaration drift would via ``LoaderError``.
    """
    source = make_source("flat-images", source_id="offline-fixture")

    def _fetched(*args: object, **kwargs: object) -> FetchResult:
        return FetchResult(
            source_id=source.id, root=tmp_path, items=3, method="fake", truncated=True
        )

    def _mismatch(*args: object, **kwargs: object) -> None:
        raise LoaderError("declared 'flat-images' but no class directories were found")

    monkeypatch.setattr("marinedata.verify.fetch_sample", _fetched)
    monkeypatch.setattr("marinedata.verify.build_loader", _mismatch)

    result = verify_source(source, limit=5)
    assert result.status == "failed" and not result.detail.startswith("fetch:")
    with pytest.raises(AssertionError):
        _assert_layout_verified(result)
