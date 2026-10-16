"""G7 decontamination: no release image is a hackathon hidden-test or benchmark image.

Exact ``sha256`` overlap between the release (``image_sha256`` and ``bytes_sha256`` of the
``images`` and ``metadata`` configs, v1.0 images included) and every manifest given with
``--hidden-test`` and ``--benchmarks`` (a file or a folder of parquet / json / jsonl / csv / tsv /
txt manifests: any 64-hex digest in them counts). Both inputs are required: without them the gate
fails; with ``--rehearsal`` it is ``skipped`` instead, which ``release publish`` still refuses.
Near-duplicate and embedding decontamination stay with ``marinedata decon check`` (S0-S5)."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from . import GateResult, finish, skip
from .v11_common import HEX64, IMAGES, METADATA, column_values, shards, tree_files

NAME = "decon"
SHA_COLUMNS = ("image_sha256", "sha256", "bytes_sha256", "sha")
TEXT_SUFFIXES = frozenset({"", ".json", ".jsonl", ".csv", ".tsv", ".txt", ".sha256"})
_HEX = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")


def _clean(value: object) -> str:
    text = str(value or "").strip().lower()
    return text if HEX64.match(text) else ""


def manifest_shas(path: Path) -> set[str]:
    """Every sha256 a manifest file (or a folder of them) lists."""
    found: set[str] = set()
    for file in tree_files(path) if Path(path).is_dir() else [Path(path)]:
        suffix = file.suffix.lower()
        if suffix == ".parquet":
            found.update(_clean(v) for col in SHA_COLUMNS for v in column_values([file], col))
        elif suffix in TEXT_SUFFIXES:
            found.update(m.lower() for m in _HEX.findall(file.read_text(errors="ignore")))
    found.discard("")
    return found


def release_shas(build_dir: Path) -> dict[str, str]:
    """``sha256 -> config`` over the release's keys and byte digests."""
    out: dict[str, str] = {}
    for config in (IMAGES, METADATA):
        for column in ("image_sha256", "bytes_sha256"):
            for value in column_values(shards(build_dir, config), column):
                if sha := _clean(value):
                    out.setdefault(sha, config)
    return out


def run_decon_gate(
    build_dir: Path,
    *,
    hidden_test: Sequence[Path] = (),
    benchmarks: Sequence[Path] = (),
    rehearsal: bool = False,
) -> GateResult:
    required = (("hidden-test manifest", hidden_test), ("benchmark manifests", benchmarks))
    missing = [label for label, paths in required if not paths]
    if missing:
        reason = "required input missing: " + ", ".join(missing)
        return skip(NAME, f"rehearsal: {reason}") if rehearsal else finish(NAME, {}, [reason])
    release = release_shas(Path(build_dir))
    if not release:
        return finish(NAME, {}, ["the build holds no image_sha256 to check"])
    counts: Counter = Counter({"release_shas": len(release)})
    failures: list[str] = []
    for kind, paths in (("hidden_test", hidden_test), ("benchmark", benchmarks)):
        for path in map(Path, paths):
            if not path.exists():
                failures.append(f"{kind} manifest {path.name}: not found")
                continue
            shas = manifest_shas(path)
            counts[f"{kind}_manifests"] += 1
            counts[f"{kind}_shas"] += len(shas)
            if not shas:
                failures.append(f"{kind} manifest {path.name}: empty (no sha256 found)")
            for sha in sorted(shas & release.keys()):
                counts["overlaps"] += 1
                failures.append(f"{kind}/{path.name} overlap: {sha[:12]} ships in {release[sha]}")
    return finish(NAME, counts, failures)
