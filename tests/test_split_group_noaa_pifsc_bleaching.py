"""``noaa-pifsc-bleaching``'s split_group rule (WS-D S29) against 5 real HF filenames.

Upstream's own train/val/test split is not honoured (2,328 parent-image groups exist
below the site level, of which 1,146 already straddle >1 upstream split — see
``registry/sources/coral-benthic.yaml`` for the full count). The registry rule groups
by *site* instead, the coarser and leak-safe option: two on the dashed station-code
convention (``FFS-B013``, ``GUA-2783``), two on the concatenated-name convention
(``LaehouShallow``, ``KawaihaeShallow``), one more dashed one from a different atoll
(``KUR-B010``) — five real stems pulled straight from the HF file listing, not
invented.
"""

from __future__ import annotations

import pytest

from marinedata import Registry


@pytest.mark.parametrize(
    ("stem", "expected_group"),
    [
        ("FFS-B013_2019_15_1024", "noaa-pifsc-bleaching/FFS-B013"),
        ("GUA-2783_2022_A_07_2055", "noaa-pifsc-bleaching/GUA-2783"),
        ("KUR-B010_2019_08_5160", "noaa-pifsc-bleaching/KUR-B010"),
        ("LaehouShallow2015IMG_2682_7264", "noaa-pifsc-bleaching/LaehouShallow"),
        ("KawaihaeShallow2015IMG_5247_2563", "noaa-pifsc-bleaching/KawaihaeShallow"),
    ],
)
def test_noaa_pifsc_bleaching_groups_by_site(
    registry: Registry, stem: str, expected_group: str
) -> None:
    source = registry.source("noaa-pifsc-bleaching")
    resolved = source.split_group_for(
        stem=stem, upstream_path=f"train/CORAL/{stem}.PNG", partition="default"
    )
    assert resolved == expected_group
