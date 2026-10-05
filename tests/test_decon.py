"""WP-12/P2: benchmark decontamination gate (S0-S5).

Design: docs/design/eval-decontamination-split-v2.md §2, acceptance criteria §5 (P2
row). Every stage is exercised through :func:`classify_pair` directly using synthetic
records — real image bytes only where a stage actually decodes pixels (S3's entropy
guard is skipped by omitting a loader; S5 needs real crop/parent pixels for NCC). No
network, no real benchmark or corpus data.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")

from PIL import Image  # noqa: E402

from marinedata.benchmarks import (  # noqa: E402
    BenchmarkEntry,
    Obtain,
    Thresholds,
    UpstreamSplit,
)
from marinedata.decon import (  # noqa: E402
    BenchmarkOverlap,
    ImageRecord,
    check_benchmark,
    classify_pair,
)
from marinedata.dedup.features import features_from_bytes  # noqa: E402

# ── fixtures ─────────────────────────────────────────────────────────────────


def _thresholds(**overrides: object) -> Thresholds:
    base = {
        "s3_dhash64_candidate_max": 12,
        "s3_phash256_confirm_max": 32,
        "s3_phash256_review_max": 48,
        "s3_min_entropy_bits": 4.0,
        "s4_embedding_model": "sscd_disc_mixup",
        "s4_embedding_revision": None,
        "s4_tau_dedup": 0.93,
        "s4_tau_dedup_calibration_sha256": None,
        "s4_tau_decon_offset": -0.03,
        "s4_review_width": 0.05,
        "s4_ann_top_k": 10,
        "s5_parent_ncc_min": 0.90,
        "s5_parent_ncc_review_min": 0.80,
        "gate_review_band_max": {"absolute": 5, "fraction_of_eval": 0.01},
        "gate_manifest_min_coverage": 0.99,
    }
    base.update(overrides)
    return Thresholds(**base)


def _rec(
    sha256: str,
    pixel_sha256: str | None = None,
    dhash: int = 0,
    phash: bytes = b"\x00" * 32,
    *,
    upstream_id: str | None = None,
    splits: frozenset[str] = frozenset(),
    embedding: np.ndarray | None = None,
    loader=None,
    area: int = 1000,
) -> ImageRecord:
    return ImageRecord(
        sha256=sha256,
        pixel_sha256=pixel_sha256 or f"pixel-{sha256}",
        dhash=dhash,
        phash=phash,
        phash64=0,
        margin=10.0,
        area=area,
        upstream_id=upstream_id,
        splits=splits,
        embedding=embedding,
        loader=loader,
    )


def _entry(
    benchmark_id: str = "fixture-bench",
    *,
    policy: str = "route-to-our-test",
    status: str = "staged",
    eval_n: int = 5,
) -> BenchmarkEntry:
    return BenchmarkEntry(
        id=benchmark_id,
        name="Fixture Bench",
        task="cls",
        catalog_id=benchmark_id,
        registry_id=None,
        upstream_split=UpstreamSplit(
            rule="train/test",
            eval_split="test",
            heldout_val=None,
            counts={"train": 50, "test": eval_n},
            definition_url="https://example.org/fixture",
        ),
        split_verified=False,
        verified_by="not fetched",
        obtain=Obtain(status=status, via="test fixture"),
        policy=policy,
        policy_reason="known contamination" if policy == "exclude" else "fixture",
        chain=[],
    )


def _png(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _crop_parent_pair() -> tuple[ImageRecord, ImageRecord]:
    """A real synthetic crop-in-parent pair: a 30x30 exact sub-region of a 96x96
    noise image. dhash/phash distances are large (no S3 collision); no embeddings (no
    S4 collision) — only S5's NCC patch match can classify this pair."""
    rng = np.random.default_rng(1)
    parent_arr = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    patch_arr = parent_arr[10:40, 10:40].copy()
    parent_bytes = _png(parent_arr)
    patch_bytes = _png(patch_arr)
    pf = features_from_bytes(parent_bytes)
    qf = features_from_bytes(patch_bytes)
    parent = _rec(
        pf.sha256,
        pf.pixel_sha256,
        pf.dhash,
        pf.phash,
        loader=lambda: parent_bytes,
        area=pf.width * pf.height,
    )
    patch = _rec(
        qf.sha256,
        qf.pixel_sha256,
        qf.dhash,
        qf.phash,
        loader=lambda: patch_bytes,
        area=qf.width * qf.height,
    )
    return patch, parent


# ── classify_pair: one planted hit per stage ────────────────────────────────


def test_s0_upstream_id_match_confirms() -> None:
    bench = _rec("bsha", upstream_id="up-1")
    corp = _rec("csha", upstream_id="up-1")
    stage, band, _ = classify_pair(bench, corp, _thresholds())
    assert (stage, band) == ("S0", "confirmed")


def test_s1_sha256_match_confirms() -> None:
    bench = _rec("same-sha")
    corp = _rec("same-sha")
    stage, band, _ = classify_pair(bench, corp, _thresholds())
    assert (stage, band) == ("S1", "confirmed")


def test_s2_pixel_sha256_match_confirms() -> None:
    bench = _rec("bsha", pixel_sha256="same-pixel")
    corp = _rec("csha", pixel_sha256="same-pixel")
    stage, band, _ = classify_pair(bench, corp, _thresholds())
    assert (stage, band) == ("S2", "confirmed")


def test_s3_phash_match_confirms_without_a_loader() -> None:
    # Identical dhash/phash, no loader — the entropy guard treats "no loader" as
    # high-entropy (never a false low-texture pass-through in a unit test).
    bench = _rec("bsha", dhash=0, phash=b"\x00" * 32)
    corp = _rec("csha", dhash=0, phash=b"\x00" * 32)
    stage, band, _ = classify_pair(bench, corp, _thresholds())
    assert (stage, band) == ("S3", "confirmed")


def test_s4_embedding_cosine_confirms() -> None:
    vec = np.zeros(8, dtype=np.float32)
    vec[0] = 1.0
    # dhash/phash maximally far apart so S3 cannot fire first.
    bench = _rec("bsha", dhash=(1 << 64) - 1, phash=b"\xff" * 32, embedding=vec)
    corp = _rec("csha", dhash=0, phash=b"\x00" * 32, embedding=vec)
    thresholds = _thresholds(s3_dhash64_candidate_max=1)
    stage, band, signal = classify_pair(bench, corp, thresholds)
    assert (stage, band) == ("S4", "confirmed")
    assert signal == pytest.approx(1.0)


def test_s5_crop_stage_off_by_default_and_on_when_enabled() -> None:
    patch, parent = _crop_parent_pair()
    thresholds = _thresholds(s3_dhash64_candidate_max=4, s5_parent_ncc_min=0.5)

    off_stage, off_band, _ = classify_pair(patch, parent, thresholds)
    assert (off_stage, off_band) == ("", "")  # D-T: off by default

    on_stage, on_band, _ = classify_pair(patch, parent, thresholds, dedup_crop=True)
    assert on_stage == "S5"
    assert on_band == "confirmed"


# ── check_benchmark: gate semantics (§2.2 / §5 P2 row) ──────────────────────


def test_clean_fixture_passes() -> None:
    entry = _entry(eval_n=2)
    bench = [_rec("b1"), _rec("b2")]
    corp = [
        _rec("b1", splits=frozenset({"test"})),
        _rec("b2", splits=frozenset({"test"})),
    ]
    overlap = check_benchmark(entry, _thresholds(), bench, corp)
    assert isinstance(overlap, BenchmarkOverlap)
    assert overlap.status == "clean"
    assert overlap.ok is True


def test_exclude_policy_hit_in_test_fails() -> None:
    entry = _entry(policy="exclude", eval_n=2)
    bench = [_rec("b1")]
    corp = [_rec("b1", splits=frozenset({"test"}))]
    overlap = check_benchmark(entry, _thresholds(), bench, corp)
    assert overlap.status == "contaminated"
    assert overlap.ok is False
    assert overlap.excluded == 1


def test_coverage_below_99_percent_fails() -> None:
    entry = _entry(status="staged", eval_n=100)
    bench = [_rec("only-one")]  # 1 << 0.99 * 100
    overlap = check_benchmark(entry, _thresholds(), bench, [])
    assert overlap.coverage_fail is True
    assert overlap.ok is False


def test_review_band_limit_enforced() -> None:
    entry = _entry(eval_n=5)
    # dhash within the candidate radius; phash strictly between confirm_max and
    # review_max — a review hit, not a confirmed one.
    bench = [_rec("b1", dhash=0, phash=b"\x00" * 32)]
    review_phash = bytes([0xFF] * 5 + [0x00] * 27)  # exactly 40 bits set
    corp = [_rec("c1", dhash=0, phash=review_phash, splits=frozenset({"train"}))]
    thresholds = _thresholds(
        s3_phash256_confirm_max=32,
        s3_phash256_review_max=48,
        gate_review_band_max={"absolute": 0, "fraction_of_eval": 0.0},
    )
    overlap = check_benchmark(entry, thresholds, bench, corp)
    assert overlap.review_band_fail is True
    assert overlap.ok is False
    assert not any(n for stages in overlap.counts.values() for n in stages.values())


# -- decon_exempt_reason: the coverage gate is default-deny (WP-R8) ---------------


def _registry(*entries: BenchmarkEntry):
    from marinedata.benchmarks import BenchmarkRegistry

    return BenchmarkRegistry(
        schema_version=1, thresholds=_thresholds(), benchmarks=tuple(entries), raw={}
    )


def _exempt(entry: BenchmarkEntry, reason: str) -> BenchmarkEntry:
    return entry.model_copy(update={"decon_exempt_reason": reason})


def test_no_manifest_with_exemption_passes_and_is_recorded(tmp_path) -> None:
    from marinedata.decon import check, decon_record
    from marinedata.hf_card import decon_limitations

    reg = _registry(_exempt(_entry("b-one", status="registry-only"), "not staged"))
    result = check(tmp_path, reg, [], manifests_root=tmp_path)
    assert result.ok and result.exempt == {"b-one": "not staged"}
    record = decon_record(result)
    assert record["exempt"] == {"b-one": "not staged"} and record["gate"] == "pass"
    card = "\n".join(decon_limitations({"decon": record}))
    assert "## Limitations" in card and "Decontamination not verified against: `b-one`" in card
    assert decon_limitations({"decon": {"exempt": {}}}) == []


@pytest.mark.parametrize("status", ["staged", "registry-only", "needs-yohan"])
def test_no_manifest_and_no_exemption_fails(tmp_path, status: str) -> None:
    from marinedata.decon import check

    reg = _registry(_entry("b-two", status=status))
    result = check(tmp_path, reg, [], manifests_root=tmp_path)
    assert not result.ok
    assert result.failures == ["b-two: no manifest and no decon_exempt_reason"]
    assert reg.uncovered(tmp_path) == ["b-two"]


def test_blank_exemption_reason_is_rejected() -> None:
    with pytest.raises(ValueError, match="decon_exempt_reason"):
        _exempt(_entry("b-three"), "  ").model_validate(
            _exempt(_entry("b-three"), "  ").model_dump()
        )
