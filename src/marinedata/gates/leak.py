"""G3 leak: the open upload set names no nc source id, no nc repo and no private bucket.

Scans every file of the build for the private names (:data:`marinedata.hf_export_v11.PRIVATE_NAMES`,
plus the nc repo id from ``registry/flavours.yaml``) and the ids of every source that is not
open-class: file names, text files (card, croissant, NOTICE, provenance), parquet footers and every
string column of every shard (the bytes columns are never read). The nc card may name the nc repo,
never a bucket. A failure line carries a hash of the leaked name, never the name: GATES.json is
uploaded under ``provenance/``. ``release gate`` scans GATES.json again once every gate of the run
has written to it (:func:`rescan_gates_json`), and the open export refuses an nc id in any text it
writes (:func:`id_tags`)."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Collection, Iterable
from pathlib import Path

from . import GateResult, finish
from .v11_common import tree_files

NAME = "leak"
TEXT_SUFFIXES = frozenset(
    {"", ".md", ".json", ".jsonl", ".yaml", ".yml", ".txt", ".cff", ".lock", ".csv", ".tsv",
     ".toml", ".html", ".xml", ".ini", ".sha256"}
)  # fmt: skip


def _tag(kind: str, value: str) -> str:
    return f"{kind}#{hashlib.sha256(value.encode()).hexdigest()[:8]}"


def _token(pattern: str) -> re.Pattern[str]:
    """``pattern`` as a whole token: no letter, digit, ``_`` or ``-`` on either side."""
    return re.compile(rf"(?<![A-Za-z0-9_-])(?:{pattern})(?![A-Za-z0-9_-])", re.I)


def _id_regex(ids: Iterable[str]) -> tuple[re.Pattern[str], dict[str, str]]:
    """One regex over all ``ids`` and ``{lower-cased id: its label}``."""
    tags = {i.lower(): _tag("nc_id", i) for i in set(ids) if i}
    names = "|".join(re.escape(i) for i in sorted(tags, key=len, reverse=True))
    return _token(names or "(?!)"), tags


def id_tags(text: str, ids: Iterable[str]) -> list[str]:
    """The sorted ``nc_id#`` labels of the ``ids`` that occur in ``text`` as whole tokens."""
    rx, tags = _id_regex(ids)
    return sorted({tags[m.group(0).lower()] for m in rx.finditer(text)})


def redact(text: str, ids: Iterable[str]) -> str:
    """``text`` with every ``ids`` token replaced by its ``nc_id#`` label (what a failure line
    carries): a published record keeps the shape of its text and loses the name."""
    rx, tags = _id_regex(ids)
    return rx.sub(lambda m: tags[m.group(0).lower()], text)


class _Finder:
    def __init__(self, names: Iterable[str], ids: Iterable[str]) -> None:
        self.names = [(n, n.lower()) for n in names if n]
        self.ids = [(i, _token(re.escape(i))) for i in sorted(set(ids))]

    def hits(self, text: str) -> list[str]:
        low = text.lower()
        found = [_tag("private_name", n) for n, lowered in self.names if lowered in low]
        return found + [_tag("nc_id", i) for i, rx in self.ids if rx.search(text)]


def _scan_parquet(path: Path, finder: _Finder) -> set[str]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    shard = pq.ParquetFile(path)
    found: set[str] = set()
    for key, value in (shard.schema_arrow.metadata or {}).items():
        found.update(finder.hits(key.decode(errors="ignore") + " " + value.decode(errors="ignore")))
    text = (pa.types.is_string, pa.types.is_large_string)
    columns = [
        f.name
        for f in shard.schema_arrow
        if any(t(f.type) for t in text)
        or (
            (pa.types.is_list(f.type) or pa.types.is_large_list(f.type))
            and any(t(f.type.value_type) for t in text)
        )
    ]
    if not columns:
        return found
    for batch in shard.iter_batches(batch_size=8192, columns=columns):
        for array in batch.columns:
            if pa.types.is_list(array.type) or pa.types.is_large_list(array.type):
                array = pc.list_flatten(array)
            for value in pc.unique(array).to_pylist():
                if value:
                    found.update(finder.hits(value))
    return found


def _finder(
    nc_ids: Collection[str],
    flavour: str,
    private_names: Collection[str] | None = None,
    nc_repo_names: Collection[str] | None = None,
) -> _Finder:
    from ..hf_export_v11 import PRIVATE_NAMES

    privates = tuple(PRIVATE_NAMES[1:] if private_names is None else private_names)
    repos = tuple(PRIVATE_NAMES[:1] if nc_repo_names is None else nc_repo_names)
    open_flavour = flavour == "open"
    return _Finder((*privates, *(repos if open_flavour else ())), nc_ids if open_flavour else ())


def run_leak_gate(
    build_dir: Path,
    *,
    nc_ids: Collection[str],
    flavour: str = "open",
    private_names: Collection[str] | None = None,
    nc_repo_names: Collection[str] | None = None,
) -> GateResult:
    """``nc_ids``: every source id that must not appear (open flavour only). ``private_names``
    (buckets, remotes) are banned in both flavours; ``nc_repo_names`` only in the open one."""
    build = Path(build_dir)
    finder = _finder(nc_ids, flavour, private_names, nc_repo_names)
    counts: Counter = Counter()
    failures: list[str] = []
    for path in tree_files(build):
        rel = path.relative_to(build).as_posix()
        counts["files"] += 1
        found = set(finder.hits(rel))
        if path.suffix == ".parquet":
            counts["parquet_files"] += 1
            found |= _scan_parquet(path, finder)
        elif path.suffix.lower() in TEXT_SUFFIXES:
            counts["text_files"] += 1
            found.update(finder.hits(path.read_text(encoding="utf-8", errors="ignore")))
        else:
            counts["binary_skipped"] += 1
        for tag in sorted(found):
            counts["leaks"] += 1
            failures.append(f"{rel} leak: {tag} in the {flavour} upload set")
    if not counts["files"]:
        failures.append(f"no files under {build}")
    return finish(NAME, counts, failures)


def rescan_gates_json(
    path: Path, *, flavour: str = "open", extra_nc_ids: Collection[str] = ()
) -> GateResult | None:
    """The ``leak`` result for a GATES.json that names a private resource, else ``None``.

    G3 runs before the other gates of its ``release gate`` call write their entries, so it only
    sees the file of an earlier call. This reads the merged file once the call is done, with the
    ids G3 uses (the registry's non-open sources plus ``extra_nc_ids``); the result keeps the
    ``leak`` entry's own counts and failures and adds, first, one line per leaked label (never
    the name). A scan that cannot run fails closed."""
    from .v11_runner import nc_source_ids

    target = Path(path)
    try:
        finder = _finder(nc_source_ids() | frozenset(extra_nc_ids), flavour)
        found = sorted(set(finder.hits(target.read_text(errors="ignore"))))
        if not found:
            return None
        before = json.loads(target.read_text()).get(NAME)
    except Exception as exc:
        return finish(NAME, {"failed_gate_error": 1}, [f"{NAME} gate_error: {type(exc).__name__}"])
    before = before if isinstance(before, dict) else {}
    counts = {k: v for k, v in (before.get("counts") or {}).items() if k != "failures_total"}
    counts["leaks"] = int(counts.get("leaks") or 0) + len(found)
    counts["gates_json_leaks"] = len(found)
    lines = [f"{target.name} leak: {tag} in the {flavour} upload set" for tag in found]
    return finish(NAME, counts, [*lines, *(before.get("failures") or [])])
