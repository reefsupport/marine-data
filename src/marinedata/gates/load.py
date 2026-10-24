# ruff: noqa: E501
"""Load gate (P1.6): the build loads through ``datasets`` and agrees with RELEASE.json + the card.

``datasets`` infers the builder of a local folder from the FIRST ``configs:`` entry of the README and
ignores the config asked for, and an additive release's build dir lacks the frozen v1.0 configs, so the
gate reads the card: every ``configs:`` entry is resolved against the build (a pattern that matches
nothing there falls back to ``--v1-dir``, see :mod:`.load_plan`) and loaded with
``load_dataset("parquet", data_files=..., streaming=True)``; the first 50 rows of each split are read
(decoding every image), and

* ``missing_config`` / ``unlisted_config``  RELEASE.json ``configs`` and the shards disagree on the config set;
* ``release_without_configs`` / ``no_release``  RELEASE.json has no ``configs`` block (hf_export writes it
  after the export), so the row counts cannot be checked;
* ``count_mismatch``   rows in the shards (parquet footers) != RELEASE.json ``configs[config][split]``;
* ``schema_mismatch``  the feature schema of a shard or of the loaded dataset != the card's
  ``dataset_info.features`` for the config (the first shard's embedded schema until the card has one);
* ``config_files_missing``  a pattern of a card ``configs:`` entry matches no file in the build or ``--v1-dir``.
  Exception, an inherited v1.0 phantom: the config is a folder of the v1.0 tree, its v1.0 card entry equals the
  build's and no v1.0 file matches the pattern either. It is counted ``inherited_v1_phantom_patterns`` (listed
  under ``inherited_v1_phantom_patterns_list``), not failed; see :func:`.load_plan.inherited_phantoms`;
* ``first_config_unresolved``  the card's FIRST config resolves to no files there (the Hub infers the loader from it);
* ``load_error`` / ``split_mismatch`` / ``iterate_error`` / ``empty_split``  ``datasets`` cannot load or read it,
  the card has no entry for a built config, or it lists other splits than the shards.

A frozen v1.0 config (coral-masks, scene-masks, instance-masks, fish-boxes) that resolves nowhere while
no ``--v1-dir`` was given is counted ``skipped_frozen`` and does not fail; every other config must load.

A patch release (``prior``, ``release gate --patch-of``) loads only the configs it adds: every config of
the prior release is frozen on the Hub (the ``frozen`` and ``card`` gates hold it byte- and entry-
identical), so it is counted ``skipped_frozen_prior`` and not resolved against the build."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import GateResult, finish
from .load_plan import ConfigFiles, plan_configs
from .patch_base import PatchBase
from .shards import CardError, config_shards, norm_split, read_card, read_release, split_counts

NAME = "load"
SAMPLE_ROWS = 50


def embedded_features(path: Path):
    import pyarrow.parquet as pq
    from datasets import Features

    return Features.from_arrow_schema(pq.read_schema(path))


def card_features(front: dict, config: str):
    """``Features`` of ``dataset_info`` for the config in the card front matter, else ``None``."""
    from datasets import Features

    info = front.get("dataset_info") or []
    for entry in [info] if isinstance(info, dict) else info:
        if (
            isinstance(entry, dict)
            and entry.get("config_name", "default") == config
            and entry.get("features")
        ):
            return Features._from_yaml_list(entry["features"])
    return None


def _footer_rows(paths) -> int:
    import pyarrow.parquet as pq

    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def run_load_gate(
    build_dir: Path,
    *,
    release_dir: Path | None = None,
    sample_rows: int = SAMPLE_ROWS,
    v1_dir: Path | None = None,
    prior: PatchBase | None = None,
) -> GateResult:
    from datasets import load_dataset

    build_dir = Path(build_dir)
    v1_dir = Path(v1_dir) if v1_dir else None
    shards = config_shards(build_dir)
    if not shards:
        return finish(NAME, {}, [f"no_data: no data/<config>/*.parquet under {build_dir}"])
    counts: Counter = Counter()
    failures: list[str] = []
    skipped: list[str] = []
    inherited: list[str] = []

    def fail(kind: str, where: str, message: str) -> None:
        counts[f"failed_{kind}"] += 1
        failures.append(f"{where} {kind}: {message}")

    release = read_release(build_dir, release_dir)
    expected = (release or {}).get("configs")
    if release is None:
        fail(
            "no_release",
            "RELEASE.json",
            "not in the build dir or --release-dir; counts cannot be checked",
        )
    elif not expected:
        fail(
            "release_without_configs",
            "RELEASE.json",
            "no `configs` block (hf_export writes it after the export)",
        )
    else:
        for config in sorted(set(expected) - set(shards)):
            fail(
                "missing_config", config, "RELEASE.json lists it but the build has no shards for it"
            )
        for config in sorted(set(shards) - set(expected)):
            fail("unlisted_config", config, "the build has shards RELEASE.json does not list")
    try:
        front, card_configs, _ = read_card(build_dir)
    except CardError:
        front, card_configs = {}, {}  # the card gate reports an unreadable card
    plan = plan_configs(card_configs, build_dir, v1_dir)

    def check(item: ConfigFiles) -> None:
        config, files = item.config, shards.get(item.config) or []
        counts["configs"] += 1
        for split, pattern in item.inherited:
            counts["inherited_v1_phantom_patterns"] += 1
            inherited.append(f"{config}/{split}: {pattern}")
        for split, pattern in item.missing:
            fail(
                "config_files_missing",
                f"{config}/{split}",
                f"pattern {pattern!r} matches no file in the build or --v1-dir",
            )
        if not item.splits:
            return
        have = split_counts(files) if files else {}
        want = {norm_split(s): int(n) for s, n in ((expected or {}).get(config) or {}).items()}
        for split in sorted(set(have) | set(want)) if want and files else ():
            if have.get(split, 0) != want.get(split, 0):
                fail(
                    "count_mismatch",
                    f"{config}/{split}",
                    f"{have.get(split, 0)} rows in the shards, RELEASE.json says {want.get(split, 0)}",
                )
        listed = {norm_split(s) for s in item.splits}
        if files and listed != set(have):
            fail(
                "split_mismatch",
                config,
                f"the card lists splits {sorted(listed)}, the shards have {sorted(have)}",
            )
        declared = card_features(front, config)
        reference, origin = (
            (declared, "card")
            if declared is not None
            else (
                embedded_features(files[0] if files else next(iter(item.splits.values()))[0]),
                "first shard",
            )
        )
        for path in files:
            if embedded_features(path).to_dict() != reference.to_dict():
                fail(
                    "schema_mismatch",
                    f"{config}/{path.name}",
                    f"parquet features differ from the {origin}",
                )
        data_files = {s: [str(p) for p in paths] for s, paths in item.splits.items()}
        try:
            loaded = load_dataset("parquet", data_files=data_files, streaming=True)
        except Exception as exc:
            fail("load_error", config, f"{type(exc).__name__}: {str(exc)[:160]}")
            return
        for split, stream in sorted(loaded.items()):
            where, read = f"{config}/{split}", 0
            try:
                features = stream.features or stream._resolve_features().features
                if features.to_dict() != reference.to_dict():
                    fail("schema_mismatch", where, f"loaded features differ from the {origin}")
                for _ in stream:
                    read += 1
                    if read >= sample_rows:
                        break
            except Exception as exc:
                fail(
                    "iterate_error",
                    where,
                    f"after {read} rows: {type(exc).__name__}: {str(exc)[:160]}",
                )
                continue
            counts["rows_read"] += read
            if not read and _footer_rows(item.splits[split]):
                fail("empty_split", where, "datasets returned no rows")

    frozen_prior = prior.configs if prior is not None else frozenset()
    if (
        plan
        and not plan[0].splits
        and not (plan[0].frozen and v1_dir is None)
        and plan[0].config not in frozen_prior
    ):
        fail(
            "first_config_unresolved",
            plan[0].config,
            "the card's first `configs:` entry resolves to no files in the build or --v1-dir; the Hub infers the loader from it",
        )
    for config in sorted(set(shards) - {item.config for item in plan}):
        fail("load_error", config, "the card's `configs:` has no entry for it")
    for item in plan:
        if item.config in frozen_prior:
            counts["skipped_frozen_prior"] += 1
        elif item.frozen and not item.splits and v1_dir is None:
            counts["skipped_frozen"] += 1
            skipped.append(item.config)
        else:
            check(item)
    extra = {
        **({"skipped_frozen_configs": skipped} if skipped else {}),
        **({"inherited_v1_phantom_patterns_list": inherited} if inherited else {}),
    }
    return finish(NAME, {**counts, "sample_rows": sample_rows, **extra}, failures)
