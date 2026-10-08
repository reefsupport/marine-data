# ruff: noqa: E501
"""Card gate (P1.6): ``README.md`` parses as a dataset card and its numbers match the release.

The card itself is generated (hf_card / P1.9); this gate only verifies it. Parsed with
``huggingface_hub.DatasetCard`` and the ``datasets`` metadata-config reader. Failures:

* ``unparseable_card``      README.md missing or not a dataset card;
* ``missing_config_section`` a config of the build / RELEASE.json has no ``configs:`` entry;
  ``extra_config_section`` the reverse; ``split_mismatch`` the entry's splits differ from the build;
* ``count_mismatch``        ``dataset_info[config].splits[].num_examples`` != the release rows;
* ``size_mismatch``         ``dataset_info[config].download_size`` != the bytes of the config's shards;
* ``size_category``         ``size_categories`` missing, or no listed band matches the row counts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from . import GateResult, finish
from .shards import CardError, config_shards, norm_split, read_card, read_release, split_counts

NAME = "card"
_BANDS = (
    (1_000, "n<1K"),
    (10_000, "1K<n<10K"),
    (100_000, "10K<n<100K"),
    (1_000_000, "100K<n<1M"),
    (10_000_000, "1M<n<10M"),
    (100_000_000, "10M<n<100M"),
    (1_000_000_000, "100M<n<1B"),
    (10_000_000_000, "1B<n<10B"),
    (100_000_000_000, "10B<n<100B"),
    (1_000_000_000_000, "100B<n<1T"),
)


def size_category(rows: int) -> str:
    """The Hugging Face ``size_categories`` band of a row count."""
    return next((label for limit, label in _BANDS if rows < limit), "n>1T")


def _info(front: dict) -> dict[str, dict]:
    info = front.get("dataset_info") or []
    entries = [info] if isinstance(info, dict) else info
    return {str(e.get("config_name", "default")): e for e in entries if isinstance(e, dict)}


def _listed_splits(config: dict) -> set[str]:
    files = config.get("data_files") or []
    return {norm_split(str(d["split"])) for d in files if isinstance(d, dict) and d.get("split")}


def run_card_gate(build_dir: Path, *, release_dir: Path | None = None) -> GateResult:
    build_dir = Path(build_dir)
    try:
        front, card_configs, _ = read_card(build_dir)
    except CardError as exc:
        return finish(NAME, {"failed_unparseable_card": 1}, [f"README.md unparseable_card: {exc}"])
    shards = config_shards(build_dir)
    expected = (read_release(build_dir, release_dir) or {}).get("configs") or {}
    info = _info(front)
    counts: Counter = Counter(card_configs=len(card_configs))
    failures: list[str] = []

    def fail(kind: str, where: str, message: str) -> None:
        counts[f"failed_{kind}"] += 1
        failures.append(f"{where} {kind}: {message}")

    totals: list[int] = []
    for config in sorted(set(shards) | set(expected)):
        have = split_counts(shards.get(config, []))
        want = {norm_split(s): int(n) for s, n in (expected.get(config) or {}).items()} or have
        totals.append(sum(want.values()))
        if config not in card_configs:
            fail("missing_config_section", config, "README.md `configs:` has no entry for it")
            continue
        listed = _listed_splits(card_configs[config])
        if listed and listed != set(want):
            fail(
                "split_mismatch",
                config,
                f"card lists splits {sorted(listed)}, the release has {sorted(want)}",
            )
        entry = info.get(config) or {}
        for item in entry.get("splits") or ():
            split, rows = norm_split(str(item.get("name"))), item.get("num_examples")
            if rows is not None and int(rows) != want.get(split):
                fail(
                    "count_mismatch",
                    f"{config}/{split}",
                    f"card says {rows} rows, the release has {want.get(split)}",
                )
        if entry.get("download_size") is not None:
            actual = sum(p.stat().st_size for p in shards.get(config, []))
            if int(entry["download_size"]) != actual:
                fail(
                    "size_mismatch",
                    config,
                    f"card download_size {entry['download_size']}, shards total {actual} bytes",
                )
    for config in sorted(set(card_configs) - set(shards) - set(expected)):
        fail("extra_config_section", config, "the card lists a config the build does not have")
    listed_bands = front.get("size_categories") or []
    listed_bands = [listed_bands] if isinstance(listed_bands, str) else list(listed_bands)
    accepted = {size_category(n) for n in totals} | {size_category(sum(totals))}
    stray = [b for b in listed_bands if b not in accepted]
    if not listed_bands or stray:
        fail(
            "size_category",
            "README.md",
            f"size_categories {listed_bands} vs rows {totals}: expected one of {sorted(accepted)}",
        )
    counts["configs"] = len(totals)
    return finish(NAME, counts, failures)
