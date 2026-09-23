"""Source loaders.

Importing this package registers every built-in layout. Sources declare a layout in
their registry entry; :func:`build_loader` resolves it.

    from marinedata.loaders import build_loader
    loader = build_loader(registry.source("reef-support-benthic"), "/data/reef_support")
    for sample in loader:
        ...
"""

from __future__ import annotations

from .base import (
    DataNotAvailable,
    LoaderError,
    SourceLoader,
    build_loader,
    loader_for,
    register_loader,
    registered_layouts,
)
from .coralseg import CoralsegRMaskLoader
from .generic import (
    AudioClipsLoader,
    CocoJsonLoader,
    CsvPointsLoader,
    ImageFolderLoader,
    ImageMaskPairLoader,
    MetadataOnlyLoader,
)
from .labelbox import LabelboxNdjsonLoader, LabelboxRgbMaskLoader

__all__ = [
    "AudioClipsLoader",
    "CocoJsonLoader",
    "CoralsegRMaskLoader",
    "CsvPointsLoader",
    "DataNotAvailable",
    "ImageFolderLoader",
    "ImageMaskPairLoader",
    "LabelboxNdjsonLoader",
    "LabelboxRgbMaskLoader",
    "LoaderError",
    "MetadataOnlyLoader",
    "SourceLoader",
    "build_loader",
    "loader_for",
    "register_loader",
    "registered_layouts",
]
