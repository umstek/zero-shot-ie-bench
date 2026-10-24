"""The one suite that loads the real nanodiff checkpoints.

Skipped per checkpoint unless its blob is already in the local HF cache
(cached_checkpoint() uses local_files_only=True, so CI never downloads
weights). Marked slow (docs/plan/02: "tests that load real weights are
marked slow and run locally on a machine that can hold the model"). To
seed the cache locally (v1, ~0.7 GB; add CKPT_V2 for the retrain):

    uv run python -c "from nanodiff_runner import engine; engine.load_model('cpu')"

Both checkpoints are ~350M params; one predict is the single masked
bidirectional forward (~10 s/question on CPU, the reason the legacy bench
culled the pair).
"""

import pytest
from conftest import NEGATIVE_TEXT, POSITIVE_TEXT, SENTIMENT_OPTIONS

import nanodiff_runner.engine as engine

pytestmark = pytest.mark.slow

# the demo's multi-decision tour (legacy demos/nanodiff_demo.py, stop 4)
TICKET_STATE = "I was charged twice for my subscription this month and want a refund."
TICKET_QUESTIONS = [
    ("Which team should handle this ticket?", ["billing", "technical", "account"]),
    ("Is the customer asking for a refund?", ["no", "yes"]),
    ("How urgent is this issue?", ["could wait", "soon", "immediately"]),
]


def _require_cached(ckpt: str):
    if engine.cached_checkpoint(ckpt) is None:
        pytest.skip(
            f"{engine.repo_of(ckpt)}::{engine.filename_of(ckpt)} is not in the "
            "local HF cache; seed it with engine.load_model('cpu') once, locally"
        )


def test_real_v1_lam1_loads_and_answers(monkeypatch: pytest.MonkeyPatch):
    _require_cached(engine.CKPT)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: engine.cached_checkpoint(engine.CKPT),
    )
    model, iter_num = engine.load_model("cpu", engine.CKPT)
    assert isinstance(iter_num, int)  # the step-3000 artifact the bench recorded

    pick, probs = engine.predict(
        model,
        POSITIVE_TEXT,
        engine.QUESTION["sentiment"],
        SENTIMENT_OPTIONS,
        "cpu",
    )
    assert pick in SENTIMENT_OPTIONS
    assert set(probs) == set(SENTIMENT_OPTIONS)
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_real_v2_retrain_loads_through_the_same_engine(monkeypatch: pytest.MonkeyPatch):
    _require_cached(engine.CKPT_V2)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: engine.cached_checkpoint(engine.CKPT_V2),
    )
    model, _ = engine.load_model("cpu", engine.CKPT_V2)
    pick, probs = engine.predict(
        model, NEGATIVE_TEXT, engine.QUESTION["sentiment"], SENTIMENT_OPTIONS, "cpu"
    )
    assert pick in SENTIMENT_OPTIONS
    assert abs(sum(probs.values()) - 1.0) < 1e-3


def test_real_multi_forward_answers_three_questions(monkeypatch: pytest.MonkeyPatch):
    _require_cached(engine.CKPT)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda **kwargs: engine.cached_checkpoint(engine.CKPT),
    )
    model, _ = engine.load_model("cpu", engine.CKPT)
    results = engine.predict_multi(model, TICKET_STATE, TICKET_QUESTIONS, "cpu")
    assert len(results) == len(TICKET_QUESTIONS)
    for (_, options), (pick, probs) in zip(TICKET_QUESTIONS, results, strict=True):
        assert pick in options
        assert set(probs) == set(options)
        assert abs(sum(probs.values()) - 1.0) < 1e-3
