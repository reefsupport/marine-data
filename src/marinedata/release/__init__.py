"""Release driver — ``marinedata release build`` (WS-D S7k).

Loads every admitted source already resolved to a local ``roots`` path, builds every
task declared under ``registry/tasks/*.yaml`` against one frozen ``SPLIT_MAP.json``, and
writes the resulting per-task split manifests plus a ``RELEASE.json`` under
``<out>/releases/<id>/``. Nothing is uploaded.

Resolving ``roots`` (fetching a pinned staged tree per admitted source, or accepting a
local override for a tree that has not been uploaded yet) is the CLI's job — see
:mod:`marinedata.cli_release` — so this module stays pure and directly testable: hand it
a registry and a ``roots`` mapping and it never touches the network.

Splitting under a frozen map, not allocating one, is the point: ``frozen=True`` raises if
a group an admitted source contributes is missing from the map (see
:mod:`marinedata.splitmap`), so a release can never silently grow the shared map or
disagree with a split some other task, or a previous release, already committed to.
"""

from __future__ import annotations

# Re-exports: every name ``marinedata.release`` exposed before the split stays importable here.
from ..builder import (  # noqa: F401
    SUPERVISED_DEFAULT_RATIOS,
    DatasetBuilder,
    PartialAbstainExclusion,
    SplitName,
)
from ..checksums import file_digest  # noqa: F401
from ..flavours import (  # noqa: F401
    check_profile,
    flavour_source_ids,
    release_record,
    sample_verdict,
)
from ..gate import evaluate  # noqa: F401
from ..labelcheck import release_label_gate  # noqa: F401
from ..licence_class import release_excluded, require_flavour, spec_for  # noqa: F401
from ..loaders.generic import IMAGE_SUFFIXES  # noqa: F401
from ..metadata_norm.base import NormContext  # noqa: F401
from ..metadata_norm.default import attribution_text  # noqa: F401
from ..metadata_norm.split_group import derive_split_group  # noqa: F401
from ..models import Source  # noqa: F401
from ..neardup import (  # noqa: F401
    NearDupChainError,
    NearDupConfig,
    compute_dhashes,
    near_dup_record,
    near_pairs,
)
from ..registry import Registry  # noqa: F401
from ..release_lock import ReleaseLock, ReleaseLockError  # noqa: F401
from ..sample_schema import staged_partition  # noqa: F401
from ..splitmap import MergeInfo, Row, load_split_map, resolve_splits, rows_to_counts  # noqa: F401
from ..strata import DEFAULT_MIN_GROUPS, TEST, TRAIN  # noqa: F401
from ..tables import _require_pyarrow  # noqa: F401
from ..upstream_split import honoured_test_groups  # noqa: F401
from .build import build_release  # noqa: F401
from .build_helpers import (  # noqa: F401
    ReleaseSkipError,
    SplitGroupError,
    _has_rows,
    _loader_layout,
    _memoized,
    _skipped_sources,
    _split_ratio_record,
    _with_release_group,
    release_group_index,
    require_split_groups,
)
from .entries import (  # noqa: F401
    GUARD_MIN_GROUPS,
    GUARD_MIN_ROWS,
    WALK_LAYOUTS,
    ReleaseEntry,
    _admitted_source_ids,
    _guard_degenerate_grouping,
    _never_eval_source_ids,
    _partition_stem_index,
    _release_group,
    _staged_entries,
    _staged_meta,
    _walk_entries,
    enumerate_release_rows,
    logger,
    release_skip_reason,
    source_attribution,
    source_release_entries,
)
from .records import DEFAULT_SCHEMA_ID, ReleaseResult, SplitMapStats, TaskManifest  # noqa: F401
from .split_map import (  # noqa: F401
    _check_near_dup_chain,
    _never_eval_near_dups,
    dedup_group_links,
    generate_split_map,
)
