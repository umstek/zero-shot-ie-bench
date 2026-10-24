"""Shared stubs + payload helpers for the nanodiff runner tests.

Offline by construction: the server tests never touch the HF cache —
engine.load_model is patched to return a tiny randomly-initialized
NanoDiff (same class, same forward, ~1/1000 the trained size), so the
real format math, masking and letter-softmax all run over stub weights.
The weight-gated tests (test_real_weights.py) skip unless the checkpoint
blob is already cached; nothing here ever downloads.
"""

from __future__ import annotations

import pytest

import nanodiff_runner.engine as engine

# the shared sample texts (legacy demos/nanodiff_demo.py SHARED_TEXTS)
POSITIVE_TEXT = "This is the best laptop I have ever owned."
NEGATIVE_TEXT = "The food was cold and the waiter was rude."
SENTIMENT_OPTIONS = ["positive", "negative", "neutral"]


def tiny_nanodiff(seed: int = 0):
    """A randomly-initialized NanoDiff at a test-sized config: same class,
    same bidirectional forward and letter-softmax path as the trained 350M
    (n_layer 16 / n_head 20 / n_embd 1280 shrink to 1 / 2 / 8; block_size
    and the vocab stay — the sequences are fixed 512-token windows over the
    GPT-2 BPE + [MASK] vocabulary). Seeded so failures reproduce."""
    import torch

    torch.manual_seed(seed)
    return engine.NanoDiff(
        engine.Config(
            n_layer=1,
            n_head=2,
            n_embd=8,
            block_size=engine.ARCH["block_size"],
            dtype="bfloat16",
        )
    )


def choice_item(
    item_id: str = "cls-001", text: str = POSITIVE_TEXT, **question: object
) -> dict[str, object]:
    """A decision-track item: suite shape (question.type choice)."""
    question.setdefault("type", "choice")
    question.setdefault("options", SENTIMENT_OPTIONS)
    return {"item_id": item_id, "input": {"text": text}, "question": question}


@pytest.fixture(autouse=True)
def _no_cached_models():
    """Keep the per-checkpoint model cache empty around every test: the
    server tests install their stub via load_model; nothing may leak a
    loaded (or stubbed) model into another test."""
    engine.reset_models()
    yield
    engine.reset_models()
