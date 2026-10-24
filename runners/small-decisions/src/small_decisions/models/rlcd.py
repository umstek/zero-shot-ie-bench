"""notnotsamuel's LFM2.5-350M-RLCD (constrained-decoding decision engine) —
ported, NOT registered on /info (no catalog yaml exists yet; there is no
catalog identity to serve — models/__init__ keeps it out of MODELS until
one lands).

notnotsamuel/LFM2.5-350M-RLCD keeps the LiquidAI/LFM2.5-350M weights and
adds RLCD training; inference runs through its constrained-decoding
engine, vendored verbatim from the HF repo (revision
deb589d803d141cabd158ef55f6617b128529f36; inference code MIT per the
card's LICENSE-CODE, the LiquidAI/LFM2.5-350M base weights under the LFM
Open License v1.0) as models/_vendored/rlcd_engine.py. The engine prefills
the context once, branches the attention/convolution state across every
field's candidate values, scores all branches in one batched pass, and
assembles the JSON from the argmax likelihoods.

Needs transformers 5.17 + jsonschema (the vendored engine imports both at
module level), so it lives in this island (.venv-von in legacy — the main
4.57 venv never ran it).

Weights load from the hub at the pinned revision (MODEL_ID/REVISION live
in the vendored engine) — the HF cache satisfies later loads.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from . import _base, _systemone

RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def available() -> bool:
    """The pinned LiquidAI/LFM2.5-350M base weights are in the HF cache."""
    return _base.hf_snapshot("LiquidAI/LFM2.5-350M") is not None


def load_engine(device: str = "cpu", dtype: str = "float32") -> Any:
    """The vendored Engine (heavy imports stay inside)."""
    from ._vendored.rlcd_engine import Engine

    return Engine(device=device, dtype=dtype)


def _schema_for(field: str, instructions: str, candidates: list[str]) -> dict[str, Any]:
    """Closed, flat, one-field schema — the engine's supported subset."""
    return {
        "type": "object",
        "properties": {field: {"type": "string", "description": instructions, "enum": candidates}},
        "required": [field],
        "additionalProperties": False,
    }


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    The typed question maps onto one enum field: choice -> the option
    labels; score -> the level texts (the answer is the winning level's
    text, not an expected value — the constrained head picks one branch);
    noul is refused (a boolean field exists upstream but the benched path
    is choice; the runner surface keeps to what the demo exercised).

    The answer carries the candidates' full log-likelihoods, not
    calibrated probabilities — prob stays None (Brier/ECE cannot use
    branch log-likelihoods honestly)."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    if question["type"] not in ("choice", "score"):
        raise ValueError(
            f"rlcd answers choice and score questions (as enum fields), got {question['type']!r}"
        )
    candidates = list(question["criteria"])
    with _LOCK:
        if not _ENGINE:
            _ENGINE.append(load_engine())
        engine = _ENGINE[0]
    res = engine.constrained(state, _schema_for("q", question["instructions"], candidates))
    try:
        chosen = json.loads(res["text"])["q"]
    except (KeyError, ValueError):
        chosen = None
    if chosen is not None and question["type"] == "score":
        chosen = candidates.index(chosen)  # the winning level's index, like the other engines
    return {"prediction": chosen, "prob": None}
