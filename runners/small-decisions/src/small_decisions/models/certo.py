"""Altslate's Certo (421M, calibrated non-generative decision model) —
ported, NOT registered on /info (no catalog yaml exists yet; there is no
catalog identity to serve — models/__init__ keeps it out of MODELS until
one lands).

The only engine of this island that ever ran on the legacy main venv's
transformers 4.57 (ModernBERT backbone); it is equally happy on this
island's 5.17.0 — the vendored inference path touches AutoModel /
AutoTokenizer only.

The inference surface is vendored verbatim from the certo GitHub repo
(github.com/AltSlate-Labs/certo, revision
5ae9dc40ec015e77606e4e7ec53aee87f89577d8, MIT) under
models/_vendored/certo/ — the checkpoint card documents no PyPI package,
usage is "from infer import DecisionModel" against a clone of that repo.
One forward pass, no text generation: the model scores each option from
its description against the state (runtime options; order-invariant by
construction).

Weights: huggingface.co/altslate/certo-decision-model (card revision
7628c0da918c5fce96a6629c548630ac6bed8c58, MIT) — CERTO_WEIGHTS points at a
local snapshot directory, otherwise the HF cache (available()) or a
download at load time supplies it.
"""

from __future__ import annotations

import threading
from typing import Any

from . import _base, _systemone

MODEL_ID_CANDIDATE = "altslate/certo-decision-model"
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}
HF_REPO = "altslate/certo-decision-model"

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def available() -> bool:
    """CERTO_WEIGHTS is set or the checkpoint is in the local HF cache."""
    import os

    if os.environ.get("CERTO_WEIGHTS"):
        return True
    return _base.hf_snapshot(HF_REPO) is not None


def _weights_path() -> str:
    """The checkpoint directory: CERTO_WEIGHTS, else the HF-cache snapshot,
    else the hub id (the loader then downloads on first use)."""
    import os

    explicit = os.environ.get("CERTO_WEIGHTS")
    if explicit:
        return explicit
    snapshot = _base.hf_snapshot(HF_REPO)
    return str(snapshot) if snapshot else HF_REPO


def load_engine(device: str = "cpu") -> Any:
    """DecisionModel over the checkpoint (heavy imports stay inside)."""
    from ._vendored.certo import DecisionModel

    return DecisionModel.load(_weights_path(), device=device)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    Certo is a choice scorer (per-option calibrated probabilities); noul
    and score questions are refused with a named ValueError (the protocol
    turns that into the item's error). Choices map to decide()'s
    [{"id", "description"}] options — the bare label when a criterion has
    no description, since the description is what the model scores."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    if question["type"] != "choice":
        raise ValueError(
            f"certo scores choice questions only, got {question['type']!r} "
            "(its head is a per-option scorer over described options)"
        )
    with _LOCK:
        if not _ENGINE:
            _ENGINE.append(load_engine())
        model = _ENGINE[0]
    options = [
        {"id": label, "description": description if description else label}
        for label, description in question["criteria"].items()
    ]
    res = model.decide(state, options)
    ids = [option["id"] for option in options]
    row = _systemone.choice_row(ids, [res["probs"][i] for i in ids])
    return _systemone.answer_to_result("choice", row)
