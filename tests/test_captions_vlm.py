"""WP-13 Tier B harness: prompt grounding, deterministic stratified sampling, and
`run_pilot` against a fake `VLMBackend` (no network, no model download — the real
`mlx-vlm`/`transformers` backends are exercised only by a manual pilot run, see
docs/captions.md)."""

import pandas as pd

from marinedata.captions.facts import CaptionFacts
from marinedata.captions.vlm import (
    build_prompt,
    cooldown_then_requeue,
    prompt_sha256,
    run_pilot,
    stratified_sample,
)


class FakeBackend:
    model_id = "fake/test-vlm"
    revision = "test"

    def __init__(self):
        self.calls = []

    def generate(self, image_path: str, prompt: str) -> str:
        self.calls.append((image_path, prompt))
        return "A fake caption."


def test_build_prompt_includes_grounding_facts_and_forbids_contradiction():
    facts = CaptionFacts(image_sha256="x", habitat="coral_reef", bleaching_status="BLEACHED")
    prompt = build_prompt(facts)
    assert "coral_reef" in prompt
    assert "BLEACHED" in prompt
    assert "Do not contradict" in prompt


def test_build_prompt_with_no_facts_still_produces_a_valid_prompt():
    prompt = build_prompt(CaptionFacts(image_sha256="x"))
    assert "no additional metadata" in prompt


def test_prompt_sha256_is_stable_and_sensitive_to_content():
    p1 = prompt_sha256("hello")
    p2 = prompt_sha256("hello")
    p3 = prompt_sha256("hello!")
    assert p1 == p2
    assert p1 != p3
    assert len(p1) == 64


def _metadata(n_a=8, n_b=2):
    rows = [
        {"image_sha256": f"a{i}", "source_id": "src-a", "habitat": "coral_reef"} for i in range(n_a)
    ]
    rows += [
        {"image_sha256": f"b{i}", "source_id": "src-b", "habitat": "seagrass"} for i in range(n_b)
    ]
    return pd.DataFrame(rows)


def test_stratified_sample_is_deterministic_for_a_fixed_seed():
    metadata = _metadata()
    s1 = stratified_sample(metadata, n=5, seed=0)
    s2 = stratified_sample(metadata, n=5, seed=0)
    assert list(s1["image_sha256"]) == list(s2["image_sha256"])


def test_stratified_sample_covers_both_strata():
    metadata = _metadata()
    sample = stratified_sample(metadata, n=5, seed=0)
    assert set(sample["source_id"]) == {"src-a", "src-b"}


def test_stratified_sample_returns_all_rows_when_n_exceeds_population():
    metadata = _metadata(n_a=2, n_b=2)
    sample = stratified_sample(metadata, n=100, seed=0)
    assert len(sample) == len(metadata)


def test_run_pilot_records_full_provenance_per_row():
    facts = CaptionFacts(image_sha256="s1", habitat="coral_reef")
    backend = FakeBackend()
    rows = [("s1", "/tmp/s1.jpg", facts)]
    out = run_pilot(rows, backend=backend, seed=7)
    assert len(out) == 1
    r = out[0]
    assert r.image_sha256 == "s1"
    assert r.caption_vlm == "A fake caption."
    assert r.vlm_model == "fake/test-vlm"
    assert r.vlm_revision == "test"
    assert r.seed == 7
    assert r.label_origin == "model"
    assert len(r.prompt_sha256) == 64


def test_cooldown_then_requeue_stops_after_max_attempts():
    sleeps = []
    assert cooldown_then_requeue(1, sleep=sleeps.append) is True
    assert cooldown_then_requeue(2, sleep=sleeps.append) is True
    assert cooldown_then_requeue(3, sleep=sleeps.append) is False
    assert len(sleeps) == 2  # the 3rd (final) attempt does not sleep again
