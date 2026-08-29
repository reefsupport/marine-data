"""`marinedata doctor` — no-network completeness reporting (T2).

Verification (`verify_source`/`verify_all`) needs the network and is covered in
`test_integration.py`. `doctor` answers a narrower, cheaper question — is a source's
*registry entry* complete — from the registry alone, so these tests never fetch.
"""

from __future__ import annotations

from conftest import make_source

from marinedata import Registry
from marinedata.enums import AccessMethod
from marinedata.fetch import auto_fetchable
from marinedata.verify import doctor, doctor_totals, unverified


def test_layout_unverified_count_matches_the_existing_metric(registry: Registry) -> None:
    """`doctor`'s layout column must agree with `unverified()` — two views of the same
    fact, and if they diverge, one of them is wrong."""
    rows = doctor(registry)
    assert sum(1 for r in rows if r.layout_status == "missing") == len(unverified(registry))


def test_metadata_only_source_is_not_a_layout_defect(registry: Registry) -> None:
    """A `metadata-only` layout has nothing to fetch or verify — n/a, not missing."""
    row = next(r for r in doctor(registry) if r.source_id == "marineinst20m")
    assert row.layout_status == "n/a"
    assert row.fetchable_status == "n/a"


def test_metadata_only_source_needs_no_crosswalk_either(registry: Registry) -> None:
    """⭐ obis declares `supervises: [taxon]` (it genuinely has taxonomic data) but is
    `metadata-only` — no loader will ever emit a Sample for it, so there is nothing a
    crosswalk could harmonise. Reporting it as "missing" was noise, not a real gap."""
    row = next(r for r in doctor(registry) if r.source_id == "obis")
    assert row.crosswalk_status == "n/a"


def test_open_vocabulary_schema_needs_no_static_crosswalk(registry: Registry) -> None:
    """⭐ worms-species and sonotype declare zero nodes BY DESIGN (see
    registry/schemas/open-vocabs.yaml) — a source using them is meant to resolve
    species/sonotypes dynamically, not via a hand-authored crosswalk YAML. Reporting
    fish4knowledge/watkins-marine-mammal/etc. as crosswalk="missing" for this reason
    was the same false-positive class as the metadata-only one, just for a different
    root cause: real supervision the registry's own design says needs no crosswalk_id
    to be complete.
    """
    for source_id in ("fish4knowledge", "watkins-marine-mammal", "marrs-reef-soundscapes"):
        row = next(r for r in doctor(registry) if r.source_id == source_id)
        assert row.crosswalk_status == "n/a", source_id


def test_dataset_native_with_zero_nodes_still_needs_a_crosswalk(registry: Registry) -> None:
    """The complement: dataset-native ALSO has zero nodes, but its own description
    says "no published crosswalk yet" — a real, closeable gap, not an open vocabulary.
    Must not be swept into the same n/a bucket as worms-species/sonotype."""
    row = next(r for r in doctor(registry) if r.source_id == "satellite-coral-mapping")
    assert row.crosswalk_status == "missing"


def test_unlabelled_source_needs_no_crosswalk(registry: Registry) -> None:
    """sweet-corals declares no supervision at all — a missing crosswalk would be a
    false positive here, not a real gap."""
    row = next(r for r in doctor(registry) if r.source_id == "sweet-corals")
    assert row.crosswalk_status == "n/a"


def test_crosswalked_source_reports_present(registry: Registry) -> None:
    row = next(r for r in doctor(registry) if r.source_id == "coralscapes")
    assert row.crosswalk_status == "ok"


def test_real_supervision_without_a_crosswalk_is_missing(registry: Registry) -> None:
    """satellite-coral-mapping declares real taxon supervision (a dense label raster)
    but has no crosswalk (dataset-native, "no published crosswalk yet" — the raster's
    pixel values have no documented legend anywhere, unlike suim's, so guessing one
    would be exactly the kind of unverified crosswalk this registry's own history
    warns against) — this is exactly the gap `doctor` exists to surface. (reefnet was
    this example until its worms-genus crosswalk was wired up; fathomnet was until its
    worms-species schema was recognised as an open vocabulary needing no static
    crosswalk_id; suim was until its 8-class scene scheme was decoded and crosswalked
    — see the two tests below and registry/crosswalks/suim-8class.yaml.)"""
    row = next(r for r in doctor(registry) if r.source_id == "satellite-coral-mapping")
    assert row.crosswalk_status == "missing"


def test_reefnet_crosswalk_is_wired(registry: Registry) -> None:
    """The one crosswalk this pass actually added: reefnet's genus labels already
    speak worms-genus vocabulary, so it reuses that existing identity crosswalk rather
    than needing a new one."""
    row = next(r for r in doctor(registry) if r.source_id == "reefnet")
    assert row.crosswalk_status == "ok"


def test_complete_requires_every_column_to_clear(registry: Registry) -> None:
    rows = doctor(registry)
    for row in rows:
        assert row.complete == (
            "missing"
            not in (
                row.layout_status,
                row.licence_status,
                row.crosswalk_status,
                row.fetchable_status,
            )
        )


def test_doctor_totals_reports_every_column(registry: Registry) -> None:
    rows = doctor(registry)
    totals = doctor_totals(rows)
    assert "fully complete" in totals
    assert "layout unverified" in totals
    assert "crosswalk missing" in totals


def test_auto_fetchable_is_false_for_gated_sources() -> None:
    source = make_source("flat-images").model_copy(
        update={
            "access": make_source("flat-images").access.model_copy(
                update={"gated": True, "notes": "x"}
            )
        }
    )
    assert not auto_fetchable(source)


def test_auto_fetchable_is_true_for_a_known_method() -> None:
    source = make_source("flat-images")
    assert source.access.method is AccessMethod.HTTP
    assert auto_fetchable(source)


def test_auto_fetchable_is_false_for_scrape() -> None:
    source = make_source("flat-images").model_copy(
        update={
            "access": make_source("flat-images").access.model_copy(
                update={"method": AccessMethod.SCRAPE}
            )
        }
    )
    assert not auto_fetchable(source)
