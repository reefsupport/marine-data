"""Public benchmark registry: load, validate, and hash ``registry/benchmarks.yaml`` (WP-11).

Design: ``docs/design/eval-decontamination-split-v2.md`` §1-2. One entry per public
benchmark whose test images could leak into our train. This module owns loading and
strict validation only; the decontamination gate that consumes it is WP-12/P2's
``marinedata.decon``, and the split allocator that honours ``policy``/pins is P3's
``marinedata.splitv2``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

try:  # libyaml is ~4x faster and present in most environments
    from yaml import CSafeLoader as _Loader
except ImportError:  # pragma: no cover - pure-Python fallback
    from yaml import SafeLoader as _Loader

TaskKind = Literal[
    "cls", "det", "inst-seg", "sem-seg", "points", "vqa", "enhancement", "salient", "depth"
]
ObtainStatus = Literal["staged", "registry-only", "w1", "w2", "w3", "needs-yohan", "unfetchable"]
Policy = Literal["route-to-our-test", "exclude"]


class BenchmarksError(Exception):
    """``registry/benchmarks.yaml`` is malformed, inconsistent, or fails a load-time rule."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UpstreamSplit(_Frozen):
    rule: str = Field(min_length=1)
    eval_split: str = Field(min_length=1)
    heldout_val: str | None = None
    counts: dict[str, int | None] = Field(default_factory=dict)
    definition_url: str = Field(min_length=1)


class Obtain(_Frozen):
    status: ObtainStatus
    via: str = Field(min_length=1)


class BenchmarkEntry(_Frozen):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9\-]*$")
    name: str = Field(min_length=1)
    task: TaskKind
    catalog_id: str = Field(min_length=1)
    registry_id: str | None = None
    upstream_split: UpstreamSplit
    split_verified: bool
    verified_by: str = Field(min_length=1)
    obtain: Obtain
    policy: Policy
    policy_reason: str = Field(min_length=1)
    chain: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _verified_needs_evidence(self) -> BenchmarkEntry:
        # split_verified only flips true on the strength of something actually seen
        # (a URL, an API response, a repo tree) — "not fetched" can never coexist with it.
        if self.split_verified and self.verified_by.strip().lower() in {
            "not fetched",
            "unverified",
            "",
        }:
            raise ValueError(
                f"{self.id}: split_verified is true but verified_by gives no evidence "
                f"('{self.verified_by}')"
            )
        return self

    @model_validator(mode="after")
    def _exclude_has_reason(self) -> BenchmarkEntry:
        if self.policy == "exclude" and not self.policy_reason.strip():
            raise ValueError(f"{self.id}: policy 'exclude' requires a non-empty policy_reason")
        return self


class Thresholds(_Frozen):
    s3_dhash64_candidate_max: int = Field(gt=0)
    s3_phash256_confirm_max: int = Field(gt=0)
    s3_phash256_review_max: int = Field(gt=0)
    s3_min_entropy_bits: float = Field(ge=0)
    s4_embedding_model: str = Field(min_length=1)
    s4_embedding_revision: str | None = None
    s4_tau_dedup: float = Field(gt=0, le=1)
    s4_tau_dedup_calibration_sha256: str | None = None
    s4_tau_decon_offset: float
    s4_review_width: float = Field(ge=0)
    s4_ann_top_k: int = Field(gt=0)
    s5_parent_ncc_min: float = Field(gt=0, le=1)
    s5_parent_ncc_review_min: float = Field(gt=0, le=1)
    gate_review_band_max: dict[str, float]
    gate_manifest_min_coverage: float = Field(gt=0, le=1)

    @property
    def tau_decon(self) -> float:
        return min(0.95, max(0.85, self.s4_tau_dedup + self.s4_tau_decon_offset))


class BenchmarkRegistry(_Frozen):
    """An immutable, validated view over ``registry/benchmarks.yaml``.

    Construct via :meth:`load`, never directly — the constructor performs no checking.
    """

    schema_version: int = Field(ge=1)
    thresholds: Thresholds
    benchmarks: tuple[BenchmarkEntry, ...]
    raw: dict[str, Any] = Field(repr=False)
    """The parsed YAML document, kept for :func:`benchmarks_sha256` — hashing the model
    instead would silently ignore a field the schema doesn't know about yet."""

    @model_validator(mode="after")
    def _unique_ids(self) -> BenchmarkRegistry:
        seen: dict[str, int] = {}
        for entry in self.benchmarks:
            seen[entry.id] = seen.get(entry.id, 0) + 1
        dupes = sorted(k for k, n in seen.items() if n > 1)
        if dupes:
            raise ValueError(f"duplicate benchmark id(s): {dupes}")
        return self

    @model_validator(mode="after")
    def _chains_resolve(self) -> BenchmarkRegistry:
        # A chain id is a free-form S57 tag (c1, c3, ...), not a benchmark id — just
        # require it be non-empty so a stray `chain: [""]` fails loudly.
        for entry in self.benchmarks:
            for tag in entry.chain:
                if not tag.strip():
                    raise ValueError(f"{entry.id}: empty chain tag")
        return self

    def by_id(self, benchmark_id: str) -> BenchmarkEntry:
        for entry in self.benchmarks:
            if entry.id == benchmark_id:
                return entry
        raise KeyError(benchmark_id)

    def manifest_path(self, benchmark_id: str, root: Path | None = None) -> Path:
        base = root if root is not None else Path(__file__).resolve().parents[2] / "registry"
        return base / "benchmarks" / "manifests" / f"{benchmark_id}.parquet"

    def pending(self, root: Path | None = None) -> list[str]:
        """Benchmark ids with no manifest parquet yet — the "still missing" list."""
        return [e.id for e in self.benchmarks if not self.manifest_path(e.id, root).is_file()]

    @classmethod
    def load(cls, path: str | Path | None = None) -> BenchmarkRegistry:
        """Load and validate ``registry/benchmarks.yaml``. Raises on any inconsistency."""
        base = Path(path) if path is not None else _default_path()
        if not base.is_file():
            raise BenchmarksError(f"benchmark registry not found: {base}")
        try:
            with base.open(encoding="utf-8") as fh:
                data = yaml.load(fh, Loader=_Loader)
        except yaml.YAMLError as exc:
            raise BenchmarksError(f"{base}: invalid YAML — {exc}") from exc
        if not isinstance(data, dict):
            raise BenchmarksError(f"{base}: expected a mapping at top level")
        try:
            return cls(
                schema_version=data.get("schema_version"),
                thresholds=data.get("thresholds"),
                benchmarks=tuple(data.get("benchmarks") or ()),
                raw=data,
            )
        except Exception as exc:
            raise BenchmarksError(f"{base}: {exc}") from exc


def _default_path() -> Path:
    packaged = Path(__file__).parent / "_registry" / "benchmarks.yaml"
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "registry" / "benchmarks.yaml"


def benchmarks_sha256(registry: BenchmarkRegistry) -> str:
    """A stable digest of the whole document (thresholds + every entry).

    Changing any threshold or benchmark field changes this hash and therefore the
    split-map hash (§3.5) — canonical JSON (sorted keys, no whitespace variance) over
    the parsed document, so comment or formatting edits never move it.
    """
    canonical = json.dumps(registry.raw, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
