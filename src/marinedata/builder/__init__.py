"""Building a training set from many sources.

Two decisions are baked in because getting them wrong is both easy and invisible:

**The licence gate runs before anything is read.** You cannot accidentally build a
shippable dataset containing research-only data, because the sources never enter.

**Splits are group-wise by default, never random.** Consecutive frames from a transect
overlap heavily; the same colony appears in dozens of them. A random split puts near
duplicates on both sides and inflates every metric — the JBG060 groups did exactly this
and reported numbers nobody could reproduce. ``split(by="site")`` is the honest default,
and ``by="random"`` exists only so that choosing it is deliberate and visible.
"""

from __future__ import annotations

# Re-exports: every name ``marinedata.builder`` exposed before the split stays importable here.
from ..enums import AnnotationKind  # noqa: F401
from ..gate import Decision, evaluate  # noqa: F401
from ..labelindex import LabelIndex  # noqa: F401
from ..lineage import Lineage, build_lineage  # noqa: F401
from ..loaders import DataNotAvailable, LoaderError, build_loader  # noqa: F401
from ..models import Source  # noqa: F401
from ..registry import Registry  # noqa: F401
from ..sample import Sample  # noqa: F401
from ..scan import CorpusScan, assign_splits, check_ratios, group_key, scan  # noqa: F401
from ..schema import Axis  # noqa: F401
from ..splitmap import load_split_map, resolve_splits  # noqa: F401
from ..task import SourceFit, TaskKind, fit_source  # noqa: F401
from .dataset import Dataset  # noqa: F401
from .dataset_builder import DatasetBuilder  # noqa: F401
from .defaults import (  # noqa: F401
    _SCALAR_LABEL_KINDS,
    SELF_SUPERVISED_DEFAULT_RATIOS,
    SUPERVISED_DEFAULT_RATIOS,
    SplitName,
    _default_ratios,
    _to_pretrain_vocabulary,
)
from .exclusions import PartialAbstainExclusion, SourcesDroppedError  # noqa: F401
from .streaming import StreamingDataset  # noqa: F401
