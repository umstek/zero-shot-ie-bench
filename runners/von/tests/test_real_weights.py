"""The one test that loads the real von-1.0 weights.

Skipped unless the pinned checkpoint — including the trained
option_marker.pt — is already in the local HF cache (cached_snapshot()
uses local_files_only=True, so CI never downloads weights). Marked slow
(docs/plan/02: "tests that load real weights are marked slow and run
locally on a machine that can hold the model"). To seed the cache:

    uv run python -c "from von_runner.von import load_von_backend; load_von_backend()"

The snapshot_download inside load_von_backend is pinned to local-only
mode here, so even this test cannot pull anything from the network.
"""

from functools import partial

import pytest

from von_runner.von import (
    MODEL_ID,
    cached_snapshot,
    load_von_backend,
    load_von_decider,
    load_von_judge,
    load_von_rate,
    provenance,
)

pytestmark = pytest.mark.slow

SENTIMENT_CHOICES = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
}
# the shared sample texts (legacy demos/von_demo.py SHARED_TEXTS[0..1])
POSITIVE_TEXT = "This is the best laptop I have ever owned."
NEGATIVE_TEXT = "The food was cold and the waiter was rude."


def _require_cached_checkpoint(monkeypatch: pytest.MonkeyPatch):
    checkpoint = cached_snapshot()
    if checkpoint is None:
        pytest.skip(
            f"the pinned {MODEL_ID} checkpoint (option_marker.pt "
            "included) is not in the local HF cache; seed it with "
            "load_von_backend() once, locally"
        )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        partial(huggingface_hub.snapshot_download, local_files_only=True),
    )
    return checkpoint


def test_real_backend_loads_strictly(monkeypatch: pytest.MonkeyPatch):
    _require_cached_checkpoint(monkeypatch)
    backend = load_von_backend()
    # _get_model() already ran inside load_von_backend — the trained head
    # validated (weights_only=True, strict load_state_dict).
    assert backend._get_model() is not None
    assert isinstance(backend._default_temp, (int, float))


def test_real_decide_matches_the_legacy_call_path(monkeypatch: pytest.MonkeyPatch):
    _require_cached_checkpoint(monkeypatch)
    backend = load_von_backend()
    decide = load_von_decider(backend)

    answer = decide(
        state=POSITIVE_TEXT,
        choices=SENTIMENT_CHOICES,
        instructions="What is the overall sentiment of this text?",
    )
    assert answer.choice in SENTIMENT_CHOICES
    assert set(answer.probabilities) == set(SENTIMENT_CHOICES)
    assert abs(sum(answer.probabilities.values()) - 1.0) < 1e-3
    assert 0.0 <= answer.confidence <= 1.0

    flipped = decide(
        state=NEGATIVE_TEXT,
        choices=SENTIMENT_CHOICES,
        instructions="What is the overall sentiment of this text?",
    )
    assert flipped.choice in SENTIMENT_CHOICES


def test_real_judge_and_rate_share_the_backend(monkeypatch: pytest.MonkeyPatch):
    _require_cached_checkpoint(monkeypatch)
    backend = load_von_backend()

    judged = load_von_judge(backend)(
        state="I was charged twice for my subscription this month and want a refund.",
        instructions="Is the customer asking for a refund?",
    )
    assert 0.0 <= judged.noul <= 1.0

    rated = load_von_rate(backend)(
        state="I was charged twice for my subscription this month and want a refund.",
        rubric=["could wait a few days", "should be fixed soon", "needs immediate action"],
        instructions="How urgent is this issue?",
    )
    # the expectation over three levels stays within their span
    assert 0.0 <= rated.score <= 2.0
    assert set(rated.probabilities) == {"0", "1", "2"}


def test_provenance_is_stable_for_run_records():
    assert provenance()["model_revision"] == ("fbc0196efa4fa1825557a8d5adad464fa1e872bb")
