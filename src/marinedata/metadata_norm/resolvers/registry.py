"""Resolvers keyed by source id."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from . import coralscapes, noaa_pifsc, reef_support_own, seaview
from .common import Resolver

RESOLVERS: Mapping[str, Resolver] = MappingProxyType(
    {
        "seaview-survey-imagery": seaview.resolve,
        "noaa-pifsc-bleaching": noaa_pifsc.resolve,
        "reef-support-benthic-own": reef_support_own.resolve,
        "reef-support-bleaching": reef_support_own.resolve,
        "coralscapes": coralscapes.resolve,
    }
)


def resolver_for(source_id: str) -> Resolver:
    try:
        return RESOLVERS[source_id]
    except KeyError:
        raise KeyError(
            f"no metadata resolver for {source_id!r}; known: {sorted(RESOLVERS)}"
        ) from None
