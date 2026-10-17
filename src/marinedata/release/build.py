"""``build_release`` — the release driver itself."""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections.abc import Callable, Iterable
from pathlib import Path

from ..builder import DatasetBuilder, PartialAbstainExclusion
from ..checksums import file_digest
from ..flavours import check_profile, flavour_source_ids, release_record, sample_verdict
from ..labelcheck import release_label_gate
from ..licence_class import release_excluded, require_flavour
from ..neardup import NearDupConfig, near_dup_record
from ..registry import Registry
from ..release_lock import ReleaseLock, ReleaseLockError
from ..splitmap import load_split_map
from ..strata import TRAIN
from .build_helpers import (
    _has_rows,
    _memoized,
    _skipped_sources,
    _split_ratio_record,
    _with_release_group,
    release_group_index,
    require_split_groups,
)
from .entries import _admitted_source_ids, _never_eval_source_ids
from .records import DEFAULT_SCHEMA_ID, ReleaseResult, TaskManifest
from .split_map import _never_eval_near_dups


def build_release(
    registry: Registry,
    *,
    release: str,
    split_map: str | Path,
    roots: dict[str, str | Path],
    out_dir: str | Path,
    profile: str = "research",
    allow_unmapped: bool = False,
    near_dup: NearDupConfig | None = None,
    dedup_v2_groups: str | Path | None = None,
    decon: bool = False,
    dedup_crop: bool = False,
    split_v2: bool = False,
    tasks: str = "v1",
    v2: bool = False,
    tasklabels_root: str | Path | None = None,
    digest: Callable[[Path], str] = file_digest,
    flavour: str | None = None,
    allow_skip: Iterable[str] = (),
    lock: ReleaseLock | None = None,
    lock_sha256: str | None = None,
    v11: bool = False,
    allow_unsplit: bool = False,
    privacy_review: str | Path | None = None,
    v1_dir: str | Path | None = None,
) -> ReleaseResult:
    """Build every registry task against a frozen split map and write the release.

    ``flavour`` (WP-L1b, ``open`` | ``nc``): ship only that flavour's sources, under one of
    its shipping profiles, into ``<out_dir>/releases/<release>/<flavour>/``; RELEASE.json
    records the flavour, class counts and every excluded source. ``None`` (default) is the
    unfiltered v1 build, byte-identical to before.

    ``v11`` (PR-B, needs ``tasks="v2"``): the task-layer configs become the v1.1.0 tables.
    ``points``, ``bleaching``, ``boxes``, ``semseg`` and ``instances`` are stamped with ``split``,
    ``taxonomy_version``, ``fidelity`` and ``bytes_config`` and typed by the WP-U1 schemas;
    ``metadata`` is the spine (:mod:`marinedata.task_layers.v11_tables`).
    ``task_layers/V11_TABLES.json`` records the row, image and unsplit counts. ``False``
    (default) is byte-identical to before. An admitted image with no split in the split map fails
    the build (``UnsplitImagesError``) unless ``allow_unsplit`` (rehearsal only): its rows then
    keep ``split = null`` and the report records ``unsplit``. The images the privacy review list
    (``privacy_review`` or ``registry/privacy/review.yaml``, :mod:`marinedata.privacy.review`)
    marks ``withhold`` are dropped from every table and counted in ``withheld_images``. A group
    the frozen split map lists under ``withheld`` is known, not missing: its samples are left out of
    every task manifest and its images out of every table, counted in ``withheld_images``,
    ``withheld_groups`` and ``withheld_by_source`` (never as ``unsplit``).

    ``v1_dir`` (``v11`` only): the frozen v1.0 release dir (``data/<config>/*.parquet``, as for
    ``release gate --v11 --v1-dir``). An admitted image keeps a v1.0 ``bytes_config`` only when
    its ``image_sha256`` is in that config; any other image, a new image of a v1.0 source
    included, is stamped ``images`` and exported with its bytes. A lock that admits a v1.0 source
    needs it: without it the build fails at once (``MissingV1TreeError``, naming ``--v1-dir``).

    ``tasks`` (WP-8c, charter D-X): ``"v1"`` (default) never touches the task-layer
    configs below and produces byte-identical output to before this parameter existed.
    ``"v2"`` additionally builds the 5 WP-8 task-layer configs (``points``, ``vqa``,
    ``semseg``, ``benthic-coarse``, ``benthic-cover`` — see
    :mod:`marinedata.task_layers.configs`) from ``<tasklabels_root>/_tasklabels/**``
    (D-Z2 producers) and writes them as parquet under
    ``<out_dir>/releases/<release>/task_layers/``. Additive only: no v1 file changes
    shape or content when ``tasks="v2"`` is passed, so v1 byte-identity holds regardless
    of this flag. ``tasklabels_root`` is the directory holding ``_tasklabels/`` (default:
    ``<repo>/data``, derived from the registry's location, never the cwd — INT-core3c);
    see :func:`marinedata.task_layers.configs.resolve_tasklabels_root`.

    ``decon`` (WP-12 P2) and ``dedup_crop`` (WP-10c) are independent switches: the latter
    only changes decon's S5 patch/crop stage when ``decon=True`` (INT-core2, D-T2).
    ``split_v2`` (WP-11/12 P3, INT-core2) validates the split-v2 config — ``registry/
    splits/v2.yaml`` plus ``registry/benchmarks.yaml`` load and hash cleanly — and records
    the hashes in ``RELEASE.json``. The full per-sample split-v2 gate needs
    per-sample geo columns, not yet in ``metadata.parquet``,
    so it is deferred to the v2 build; this switch is a config-only smoke check, off by
    default, that changes nothing when off (D-X).

    ``v2=True`` is a preset (INT-core2): it turns on ``decon``, ``dedup_crop``,
    ``split_v2`` and ``tasks="v2"`` together. It does not turn on the WP-10
    ``dedup_v2_groups`` split-leak gate, which needs an explicit groups.parquet path.
    Default off, so passing no flags stays byte-identical to before this preset existed.

    Raises if ``split_map`` does not exist yet (a release never allocates one — see
    :mod:`marinedata.splitmap`), or if any admitted source contributes a group in neither its
    ``assignments`` nor its ``withheld`` (``SplitMapError``, propagated from
    ``Dataset.split(frozen=True)``) — both left uncaught deliberately, since either means the
    release is not reproducible yet.

    ``near_dup`` (WS-D S47 rule A): every image of a ``never-eval`` source within
    ``near_dup.exclude_max`` dHash Hamming of any row of an eval-capable admitted source
    (any split) is dropped from every task manifest, and recorded in ``RELEASE.json`` as
    a count plus the sorted sha256 list, alongside the check's own parameters.

    Hard gate (WP-R2e, flavour builds): a releasable source with rows > 0 that the split map
    cannot cover FAILS the build, unless it is named in ``allow_skip`` (``--allow-skip``) or
    carries a registry ``release_skip_reason``. Every skipped source is recorded, with its
    reason and what allowed it, under ``skipped_sources`` in RELEASE.json.
    """
    digest = _memoized(digest)
    if v2:
        decon = True
        dedup_crop = True
        split_v2 = True
        tasks = "v2"
    tasks_mode = tasks
    if v11 and tasks_mode != "v2":
        raise ValueError(
            "v11 builds the v1.1.0 task tables from the task-layer configs: needs tasks='v2'"
        )
    split_map_path = Path(split_map)
    if load_split_map(split_map_path) is None:
        raise ValueError(
            f"{split_map_path} does not exist — generate it first with "
            "`marinedata splitmap generate`."
        )

    require_flavour(flavour, "build_release")
    if flavour is not None:
        check_profile(flavour, profile)
    admitted = _admitted_source_ids(registry, roots, profile)
    if flavour is not None:
        admitted = flavour_source_ids(registry, admitted, flavour)
    else:  # test escape only: release-excluded sources never ship
        admitted = [s for s in admitted if not release_excluded(s)]
    not_in_map: dict[str, tuple[str, str]] = {}
    if flavour is not None:  # the map enumerator never saw these (WP-R2d): don't root them
        not_in_map = _skipped_sources(registry, admitted, roots, frozenset(allow_skip))
        admitted = [sid for sid in admitted if sid not in not_in_map]
    if not admitted:
        raise ValueError(
            f"no admitted source under profile {profile!r}"
            + (f" in flavour {flavour!r}" if flavour else "")
            + " has a resolved root"
        )
    if lock is not None and (unlocked := sorted(set(admitted) - lock.source_ids())):
        raise ReleaseLockError(f"source not in the release lock: {', '.join(unlocked)}")
    v1_config_shas = None
    if v11:  # before any digest walk: a missing v1.0 tree must not cost a whole build
        from ..task_layers.v11_tables import load_v1_config_shas

        v1_config_shas = load_v1_config_shas(v1_dir, admitted)
    # WP-7: the label gate — 0 silent drops, >= 95% mapped (or a documented exception),
    # every canonical taxon node anchored. Skipped for in-memory registries (no root).
    unstaged = (  # flavour builds already dropped rowless sources in _skipped_sources
        [] if flavour is not None else [sid for sid in admitted if not _has_rows(Path(roots[sid]))]
    )
    taxonomy_stamp = (
        {} if allow_unmapped else release_label_gate(registry, admitted, empty=unstaged)
    )
    admitted_roots = {source_id: Path(roots[source_id]) for source_id in admitted}
    never_eval_sources = _never_eval_source_ids(registry, admitted)
    walk_groups = release_group_index(registry, admitted_roots, digest)

    release_root = Path(out_dir) / "releases" / release / (flavour or "")
    manifests_dir = release_root / "tasks"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    map_bytes = split_map_path.read_bytes()
    (release_root / "SPLIT_MAP.json").write_bytes(map_bytes)
    map_sha256 = hashlib.sha256(map_bytes).hexdigest()

    tasks: list[TaskManifest] = []
    skipped: list[tuple[str, str]] = []
    never_eval_excluded = 0
    licence_dropped: dict[str, dict[str, set[str]]] = {}
    near_dup_excluded: list[str] = []
    if near_dup is not None and never_eval_sources:
        near_dup_excluded = _never_eval_near_dups(
            registry, admitted_roots, profile, never_eval_sources, near_dup, digest=digest
        )
    near_dup_excluded_set = frozenset(near_dup_excluded)
    near_dup_rows = 0
    # schema 2: images in the prior-release band, on the side their component did not keep
    band_excluded = frozenset(load_split_map(split_map_path).band_excluded)
    phash_excluded = frozenset(load_split_map(split_map_path).phash_excluded) - band_excluded
    band_rows = 0
    phash_rows = 0
    partial_abstain_excluded: list[PartialAbstainExclusion] = []
    task_source_exclusions: list[dict[str, str]] = []

    for task in sorted(registry.tasks, key=lambda t: t.id):
        builder = DatasetBuilder(
            registry,
            profile=profile,
            roots=admitted_roots,
            schema_id=task.schema_id or DEFAULT_SCHEMA_ID,
            task_id=task.id,
            allow_unmapped=allow_unmapped,
            exclude_unmapped=True,
        )
        try:
            dataset = builder.build()
        except ValueError as exc:
            # No admitted source can serve this task (e.g. a partial dry-run roots) —
            # not a defect in the release, just an empty task this time.
            skipped.append((task.id, str(exc)[:200]))
            continue

        partial_abstain_excluded.extend(dataset.partial_abstain_excluded)
        task_source_exclusions.extend(
            {"task": task.id, "source": sid, "reason": reason}
            for sid, reason in builder.unmapped_excluded
        )
        dataset.samples = [
            _with_release_group(sample, walk_groups, admitted_roots) for sample in dataset.samples
        ]
        require_split_groups(dataset.samples, task.id)
        dataset.split(by="group", split_map=split_map_path, frozen=True, tolerance=None)

        rows: list[tuple[str, str]] = []
        for split_name, positions in dataset.splits.items():
            for position in positions:
                sample = dataset.samples[position]
                if sample.image is None:
                    continue
                verdict = sample_verdict(registry, sample, flavour)
                if verdict is not None:  # PA-P2.2: per-row licence null / outside the flavour
                    by_reason = licence_dropped.setdefault(sample.source_id, {})
                    by_reason.setdefault(verdict, set()).add(str(sample.image))
                    continue
                sha256 = digest(Path(sample.image))
                if sample.source_id in never_eval_sources and sha256 in near_dup_excluded_set:
                    near_dup_rows += 1
                    continue
                if sha256 in band_excluded:  # per image: the group keeps its split
                    band_rows += 1
                    continue
                if sha256 in phash_excluded:  # per image: a cross-split pHash-64 pair (schema 2)
                    phash_rows += 1
                    continue
                if split_name != TRAIN and sample.source_id in never_eval_sources:
                    # A never-eval-only merged component is already forced to `train`
                    # (see generate_split_map); this is the mixed-component case — some
                    # of the group's images are real evaluation data, so the group still
                    # lands wherever they send it, but the machine-generated rows must
                    # never enter the resulting evaluation split. Counted, never moved
                    # to train — that would leak the same content the split holds out.
                    never_eval_excluded += 1
                    continue
                rows.append((sha256, split_name))
        rows.sort()

        manifest_path = manifests_dir / f"{task.id}.tsv"
        with manifest_path.open("w", newline="") as fh:
            fh.write("image_sha256\tsplit\n")
            for sha256, split_name in rows:
                fh.write(f"{sha256}\t{split_name}\n")

        tasks.append(TaskManifest(task_id=task.id, rows=tuple(rows)))

    release_json = {
        "release": release,
        **taxonomy_stamp,
        "split_map_sha256": map_sha256,
        "split_ratio": _split_ratio_record(map_bytes),
        "sources": [
            {
                "id": source_id,
                "version": registry.source(source_id).version,
                "root_digest": (
                    registry.source(source_id).checksums.root_digest
                    if registry.source(source_id).checksums is not None
                    else ""
                ),
            }
            for source_id in admitted
        ],
        "tasks": [task.task_id for task in tasks],
        "skipped_tasks": [{"task": task_id, "reason": reason} for task_id, reason in skipped],
        "partial_abstain_excluded": [
            {
                "source": exclusion.source_id,
                "task": exclusion.task_id,
                "abstaining_labels": list(exclusion.abstaining_labels),
                "rows_dropped": exclusion.rows_dropped,
            }
            for exclusion in partial_abstain_excluded
        ],
    }
    if task_source_exclusions:  # only when a source was left out of a task (golden bytes stay)
        release_json["task_source_exclusions"] = task_source_exclusions
    if not_in_map:
        release_json["skipped_sources"] = [
            {"id": sid, "reason": reason, "allowed_by": allowed_by}
            for sid, (reason, allowed_by) in sorted(not_in_map.items())
        ]
    if near_dup is not None:
        release_json["near_dup"] = near_dup_record(near_dup)
        release_json["never_eval_near_dup_excluded"] = {
            "count": len(near_dup_excluded),
            "sha256": near_dup_excluded,
        }
    if band_excluded:
        release_json["prior_band_excluded"] = {
            "count": len(band_excluded),
            "sha256": sorted(band_excluded),
        }
    if phash_excluded:
        release_json["phash_excluded"] = {
            "count": len(phash_excluded),
            "sha256": sorted(phash_excluded),
        }
    if dedup_v2_groups is not None:
        # WP-10 split-leak gate, OFF by default: when off nothing below changes, so v1
        # outputs stay byte-identical. The gate report is written even on failure, and a
        # failing release never gets a RELEASE.json.
        from ..dedup.groups import DedupGateError, gate_release, write_gate_report

        gate = gate_release(release_root, dedup_v2_groups)
        write_gate_report(gate, release_root / "DEDUP_GATE.json")
        if not gate.ok:
            raise DedupGateError(
                f"split-leak gate failed: {len(gate.spanning)} group(s) span splits, "
                f"{len(gate.upstream_test_in_train)} upstream-test group(s) in train, "
                f"{len(gate.ungrouped)} ungrouped image(s) — see {release_root / 'DEDUP_GATE.json'}"
            )
        release_json["dedup_v2"] = {
            "groups_sha256": file_digest(Path(dedup_v2_groups)),
            "gate": "pass",
            "groups": gate.groups,
        }
    if split_v2:  # WP-11/12 P3 hook, off by default (INT-core2); config-only smoke
        # check — see the build_release docstring for why the full per-sample gate
        # is deferred to the v2 build.
        from ..benchmarks import BenchmarkRegistry, benchmarks_sha256
        from ..splitv2.rules import load_rules, rules_sha256

        rules = load_rules("registry/splits/v2.yaml")
        bench = BenchmarkRegistry.load()
        release_json["split_v2"] = {
            "rules_sha256": rules_sha256(rules),
            "benchmarks_sha256": benchmarks_sha256(bench),
            "status": "config-validated-only; per-sample gate deferred to the v2 build",
        }
    if decon:  # WP-12 P2 hook, off by default; the integrator flips it at the v2 build
        from ..decon import decon_record, run_decon_gate

        decon_result = run_decon_gate(release_root, registry, admitted_roots, dedup_crop=dedup_crop)
        release_json["decon"] = decon_record(decon_result)
    if flavour is not None:
        release_json.update(release_record(registry, flavour, profile, admitted))
        dropped = {
            sid: {
                "total": sum(len(paths) for paths in by_reason.values()),
                **{reason: len(paths) for reason, paths in sorted(by_reason.items())},
            }
            for sid, by_reason in sorted(licence_dropped.items())
        }
        if dropped:  # absent = nothing dropped: a release without per-row drops keeps its bytes
            release_json["licence_rows_dropped"] = dropped
    if lock is not None:  # Phase A (P1.3): the pin set this release was built from
        release_json["lock_sha256"] = lock_sha256
        release_json["lock_sources"] = [e.model_dump(mode="json") for e in lock.sources]
    release_json_text = json.dumps(release_json, indent=2, sort_keys=True) + "\n"
    (release_root / "RELEASE.json").write_text(release_json_text)

    task_layer_configs: dict[str, int] = {}
    staged_layers: tuple = ()
    if tasks_mode == "v2":
        from ..task_layers.configs import build_all_configs, resolve_tasklabels_root, write_configs

        # INT-core3c: never the cwd — explicit root, else derived from the registry.
        base_dir = resolve_tasklabels_root(registry, tasklabels_root)
        v11_inputs = None
        with contextlib.ExitStack() as scope:
            if v11:
                from ..privacy.review import load_review
                from ..task_layers.staged_annotations import staged_overlay
                from ..task_layers.v11_tables import inputs_from_build

                review = load_review(privacy_review, registry_root=registry.root)
                v11_inputs = inputs_from_build(
                    taxonomy_version=taxonomy_stamp.get("taxonomy_version"),
                    roots=admitted_roots,
                    manifests=tasks,
                    split_map=load_split_map(split_map_path),
                    dedup_groups=dedup_v2_groups,
                    allow_unsplit=allow_unsplit,
                    digest=digest,
                    privacy_withheld=review.withheld,
                    v1_config_shas=v1_config_shas,
                )
                # The tasklabels root holds no points / boxes / instances annotations of the
                # staged trees: build them from the admitted roots into an overlay of that root
                # (same memoised digest as the spine, so the images are hashed once).
                overlay = scope.enter_context(
                    staged_overlay(registry, base_dir, admitted_roots, digest=digest)
                )
                base_dir, staged_layers = overlay.root, overlay.layers
            results = build_all_configs(registry, base_dir, flavour, v11=v11_inputs)
        if v11_inputs is None:
            write_configs(results, release_root / "task_layers")
        else:
            from ..task_layers.staged_annotations import layer_record
            from ..task_layers.v11_tables import v11_schemas, write_v11_report

            write_configs(results, release_root / "task_layers", v11_schemas(results))
            write_v11_report(
                results,
                release_root / "task_layers",
                staged_layers=[layer_record(layer) for layer in staged_layers],
            )
        task_layer_configs = {config_id: result.n_images for config_id, result in results.items()}

    return ReleaseResult(
        release=release,
        out_dir=release_root,
        sources=tuple(admitted),
        tasks=tuple(tasks),
        skipped_tasks=tuple(skipped),
        never_eval_excluded=never_eval_excluded,
        partial_abstain_excluded=tuple(partial_abstain_excluded),
        never_eval_near_dup_excluded=tuple(near_dup_excluded),
        never_eval_near_dup_rows=near_dup_rows,
        prior_band_excluded=tuple(sorted(band_excluded)),
        prior_band_rows=band_rows,
        phash_excluded=tuple(sorted(phash_excluded)),
        phash_rows=phash_rows,
        task_layer_configs=task_layer_configs,
        staged_layers=staged_layers,
    )
