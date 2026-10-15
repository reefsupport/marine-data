"""Registry items R1-R6 for marine-data v1.1.0 (design 2026-10-15, section 6)."""

from __future__ import annotations

from marinedata.registry import Registry


def test_r6_worms_and_meow_carry_attribution_terms() -> None:
    licences = Registry.load()._licences
    worms, meow = licences["WORMS-CC-BY-4.0"], licences["MEOW-WORKING-GROUP"]
    assert worms.flags.attribution_required and "doi:10.14284/170" in worms.notes
    assert meow.flags.attribution_required and "never published" in meow.notes
    assert "doi:10.1641/B570707" in meow.notes and "MEOW Working Group" in meow.notes
