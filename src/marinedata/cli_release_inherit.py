"""``release split-map`` flags for an inheriting split map (v1.1.0 from v1.0).

``--lock`` scopes the enumeration to the release locks' sources, ``--inherit`` names the prior
release's map to embed, ``--ood-rules``/``--no-ood``/``--ood-metadata`` drive the ``test_ood``
holdouts. Split out of :mod:`marinedata.cli_release` to keep that module under the line cap."""

from __future__ import annotations

import argparse
from pathlib import Path

from .registry import Registry
from .release_lock import load_lock
from .splitmap_inherit import load_inherited
from .splitv2.release_ood import load_ood_metadata
from .splitv2.rules import load_rules


def add_inherit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--lock",
        dest="split_map_locks",
        action="append",
        type=Path,
        metavar="LOCK",
        help="release.lock.yaml (repeatable, one per flavour): enumerate exactly the sources the "
        "locks name and fetch nothing else (default: every admitted source)",
    )
    parser.add_argument(
        "--inherit",
        type=Path,
        metavar="PRIOR_SPLIT_MAP",
        help="the prior release's SPLIT_MAP.json (v1.0): embedded verbatim and never reassigned; "
        "only new groups are allocated, around it",
    )
    parser.add_argument(
        "--ood-rules",
        dest="ood_rules",
        type=Path,
        default=None,
        metavar="YAML",
        help="with --inherit: the OOD holdouts that route new groups to test_ood "
        "(default: <registry>/splits/v2.yaml)",
    )
    parser.add_argument(
        "--no-ood",
        dest="no_ood",
        action="store_true",
        help="with --inherit: no OOD holdouts (explicit opt-out)",
    )
    parser.add_argument(
        "--ood-metadata",
        dest="ood_metadata",
        type=Path,
        default=None,
        metavar="PARQUET",
        help="optional per-image OOD fields: image_sha256 plus meow_realm, meow_province, "
        "depth_m, platform, capture_datetime (the yaml's per-source fallbacks apply without it)",
    )


def lock_scope(args: argparse.Namespace, registry: Registry) -> set[str] | None:
    """The union of the ``--lock`` files' source ids, or ``None`` (unscoped)."""
    locks = getattr(args, "split_map_locks", None)
    if not locks:
        return None
    scope: set[str] = set()
    for path in locks:
        scope |= set(load_lock(path, registry).source_ids())
    return scope


def inherit_kwargs(args: argparse.Namespace, registry: Registry) -> dict[str, object]:
    """``generate_split_map`` keyword arguments for ``--inherit`` and the OOD flags."""
    inherit = getattr(args, "inherit", None)
    ood_rules = getattr(args, "ood_rules", None)
    ood_metadata = getattr(args, "ood_metadata", None)
    if inherit is None:
        if ood_rules is not None or ood_metadata is not None:
            raise ValueError("--ood-rules and --ood-metadata need --inherit")
        return {}
    prior, digest = load_inherited(inherit)
    kwargs: dict[str, object] = {"inherit": prior, "inherit_sha256": digest}
    if getattr(args, "no_ood", False):
        return kwargs
    if ood_rules is None:
        if registry.root is None:
            raise ValueError(
                "no registry root to find splits/v2.yaml: pass --ood-rules or --no-ood"
            )
        ood_rules = Path(registry.root) / "splits" / "v2.yaml"
    kwargs["ood_rules"] = load_rules(ood_rules)
    if ood_metadata is not None:
        kwargs["ood_metadata"] = load_ood_metadata(ood_metadata)
    return kwargs
