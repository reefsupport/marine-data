"""marinedata — a licence-aware registry and dataloaders for marine datasets.

Quick start::

    import marinedata as md

    # What can we legally train a shippable benthic segmenter on?
    result = md.find(task="benthic-segmentation", profile="ship-commercial")
    print(result.summary())

    # What did the licence gate remove, and why?
    for decision in result.excluded:
        print(decision.reason)

The registry is metadata, not legal advice. See ``docs/LEGAL.md``.
"""

from __future__ import annotations

from .builder import Dataset, DatasetBuilder
from .enums import (
    AccessMethod,
    AnnotationKind,
    Capability,
    LegalBasis,
    Modality,
    Provenance,
    Region,
    Tier,
)
from .gate import Decision, LicenceViolation, allowed, enforce, evaluate
from .harmonize import HarmonizationError, Harmonized, Harmonizer
from .labelindex import IGNORE_INDEX, AxisIndex, LabelIndex
from .lineage import Lineage, build_lineage
from .loaders import DataNotAvailable, LoaderError, build_loader, registered_layouts
from .models import Licence, LoaderSpec, Profile, Source
from .query import QueryResult, find
from .registry import Registry, RegistryError
from .sample import LabelValue, Sample
from .schema import Axis, Crosswalk, Fidelity, LabelSchema

__version__ = "0.1.0"

__all__ = [
    "IGNORE_INDEX",
    "AccessMethod",
    "AnnotationKind",
    "Axis",
    "AxisIndex",
    "Capability",
    "Crosswalk",
    "DataNotAvailable",
    "Dataset",
    "DatasetBuilder",
    "Decision",
    "Fidelity",
    "HarmonizationError",
    "Harmonized",
    "Harmonizer",
    "LabelIndex",
    "LabelSchema",
    "LabelValue",
    "LegalBasis",
    "Licence",
    "LicenceViolation",
    "Lineage",
    "LoaderError",
    "LoaderSpec",
    "Modality",
    "Profile",
    "Provenance",
    "QueryResult",
    "Region",
    "Registry",
    "RegistryError",
    "Sample",
    "Source",
    "Tier",
    "allowed",
    "build_lineage",
    "build_loader",
    "enforce",
    "evaluate",
    "find",
    "registered_layouts",
]
