# ruff: noqa: E501
"""Load gate (P1.6): the build loads through ``datasets`` and agrees with RELEASE.json + the card.

For every config of the build: ``datasets.load_dataset(build_dir, config, streaming=True)``, the first
50 rows of each split are read (decoding every image), and

* ``missing_config`` / ``unlisted_config``  RELEASE.json ``configs`` and the shards disagree on the config set;
* ``release_without_configs`` / ``no_release``  RELEASE.json has no ``configs`` block (hf_export writes it
  after the export), so the row counts cannot be checked;
* ``count_mismatch``   rows in the shards (parquet footers) != RELEASE.json ``configs[config][split]``;
* ``schema_mismatch``  the feature schema of a shard or of the loaded dataset != the card's
  ``dataset_info.features`` for the config (the first shard's embedded schema until the card has one);
* ``load_error`` / ``split_mismatch`` / ``iterate_error`` / ``empty_split``  ``datasets`` cannot load or read it."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import GateResult, finish
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


def run_load_gate(
    build_dir: Path, *, release_dir: Path | None = None, sample_rows: int = SAMPLE_ROWS
) -> GateResult:
    from datasets import load_dataset

    build_dir = Path(build_dir)
    shards = config_shards(build_dir)
    if not shards:
        return finish(NAME, {}, [f"no_data: no data/<config>/*.parquet under {build_dir}"])
    counts: Counter = Counter()
    failures: list[str] = []

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
        front = read_card(build_dir)[0]
    except CardError:
        front = {}  # the card gate reports an unreadable card
    for config, files in sorted(shards.items()):
        counts["configs"] += 1
        have = split_counts(files)
        want = {norm_split(s): int(n) for s, n in ((expected or {}).get(config) or {}).items()}
        for split in sorted(set(have) | set(want)) if want else ():
            if have.get(split, 0) != want.get(split, 0):
                fail(
                    "count_mismatch",
                    f"{config}/{split}",
                    f"{have.get(split, 0)} rows in the shards, RELEASE.json says {want.get(split, 0)}",
                )
        declared = card_features(front, config)
        reference, origin = (
            (declared, "card")
            if declared is not None
            else (embedded_features(files[0]), "first shard")
        )
        for path in files:
            if embedded_features(path).to_dict() != reference.to_dict():
                fail(
                    "schema_mismatch",
                    f"{config}/{path.name}",
                    f"parquet features differ from the {origin}",
                )
        try:
            loaded = load_dataset(str(build_dir), config, streaming=True)
        except Exception as exc:
            fail("load_error", config, f"{type(exc).__name__}: {str(exc)[:160]}")
            continue
        if set(loaded) != set(have):
            fail(
                "split_mismatch",
                config,
                f"datasets sees splits {sorted(loaded)}, the shards have {sorted(have)}",
            )
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
            if not read and have.get(norm_split(split), 0):
                fail("empty_split", where, "datasets returned no rows")
    return finish(NAME, {**counts, "sample_rows": sample_rows}, failures)
