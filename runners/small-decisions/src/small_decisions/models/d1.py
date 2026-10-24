"""LiquidAI's open d1 decision models (d1-3B, d1-omni-600M) — ported, NOT
registered on /info.

The catalog assigns the local d1 deployments to `runner: von`
(catalog/models/liquid/d1-3b.yaml deployment `d1-3b.local-von`,
d1-omni-600m.yaml `d1-omni-600m.local-von`): the models load through the
card's trust_remote_code AutoModel route (auto_map -> modeling_d1.D1Model
over the lfm2_vl backbone), which needs transformers >=5.14 — pins this
island actually satisfies (5.17.0), but the catalog data, not the pins,
routes d1 local to the von island. This module keeps the ported loader and
its runner-contract predict surface so a catalog reassignment to
small-decisions is a one-line registration (add the two entries to
models/__init__.py's MODELS/_PREDICT).

Ported from legacy/engines/d1_client.py: single-pass decision models that
take a state (text, JSON, or images) plus named noul/choice/score
questions and answer every question from the model's distribution over the
options — zero output tokens, no generation, calibrated probabilities.

  - d1-3B (3.12B on LFM2.5-VL-3B; SigLIP2 NaFlex 400M vision encoder)
  - d1-omni-600M (587M: text, images and <=30 s speech in one state)

Homes: D1_HOME (a prefix directory; each model's snapshot sits at
<prefix>/<dir-name>) or the default C:\src base. fp32 on the CPU — the
house dtype. A cold load of d1-3B moves ~6.3 GB of weights.

predict(state, questions, images=None) is the card's system_one contract
minus HTTP plumbing; answer shapes are d1's own: noul -> {"noul": p_true};
choice -> {choice, confidence, probabilities}; score -> {score (expected
level), confidence, probabilities, legend}.
r"""

from __future__ import annotations

import os
import threading
from typing import Any

from . import _systemone

# benchmark/demo shorthand -> snapshot directory name under the home prefix
MODELS = {
    "d1-3b": "d1-3B",
    "d1-omni-600m": "d1-omni-600M",
}
DEFAULT_MODEL = "d1-3b"

# What the catalog would claim (liquid/d1-3b.yaml: version d1-3b@2026-10,
# runtime torch/null-dtype/cpu) — kept here for the reassignment, unused
# while the deployment points at runner: von.
CATALOG_ENTRIES: list[dict[str, Any]] = [
    {"model_id": "liquid/d1-3b", "version": "d1-3b@2026-10",
     "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "liquid/d1-omni-600m", "version": "d1-omni-600m@2026-10",
     "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"}},
]
MODEL_IDS = {
    "liquid/d1-3b": "d1-3b",
    "liquid/d1-omni-600m": "d1-omni-600m",
}

_ENGINE: dict[str, Any] = {}
_LOCK = threading.Lock()


def normalize_model(model: Any) -> str:
    """Registry key for a model ('d1-3B', full HF id or None).
    ValueError names the problem so callers can surface it."""
    if model is None or not str(model).strip():
        return DEFAULT_MODEL
    key = str(model).strip().lower()
    for canonical in MODELS:
        if key == canonical.lower():
            return canonical
    for canonical, repo in MODELS.items():
        if key == repo.lower() or key == f"LiquidAI/{repo}".lower():
            return canonical
    raise ValueError(f"unknown d1 model {model!r} (one of {', '.join(MODELS)})")


def model_dir(model: Any = DEFAULT_MODEL) -> str:
    """Snapshot directory: D1_HOME prefix, default C:\\src."""
    base = os.environ.get("D1_HOME", r"C:\src")
    return os.path.join(base, MODELS[normalize_model(model)])


def available(model: Any = DEFAULT_MODEL) -> bool:
    """The snapshot's config.json is locally present."""
    return os.path.isfile(os.path.join(model_dir(model), "config.json"))


def _require_snapshot(model: str) -> str:
    home = model_dir(model)
    if not os.path.isfile(os.path.join(home, "config.json")):
        raise FileNotFoundError(
            f"no {MODELS[model]} snapshot at {home} — "
            f"pull it with `hf download LiquidAI/{MODELS[model]}"
            f" --local-dir {home}` (or point D1_HOME at the prefix directory)"
        )
    return home


class D1Engine:
    """One loaded d1 model; predict() answers every question of a request
    from one forward pass (the card's system_one, warmed by the caller)."""

    def __init__(self, handle: Any, name: str):
        self._model = handle
        self.name = name

    def predict(self, state: Any, questions: dict[str, Any], images: Any = None, audio: Any = None) -> dict[str, Any]:
        """One forward over state+questions; images=[PIL] and audio= (16 kHz
        mono int16 numpy array, d1-omni-600M only) ride the same call."""
        kwargs: dict[str, Any] = {}
        if images:
            kwargs["images"] = images
        if audio is not None:
            kwargs["audio"] = audio
        out = self._model.system_one(state, questions, **kwargs)
        return {"answers": out["answers"], "model": self.name, "input_tokens": out["usage"]["input_tokens"]}


def load_engine(model: Any = DEFAULT_MODEL, device: str = "cpu", dtype: str = "float32") -> D1Engine:
    """D1Engine over the local snapshot (fp32 on the CPU).

    Heavy imports stay inside (offline tests import this module freely).
    The card's bf16 checkpoints upcast to fp32 here."""
    import torch
    from transformers import AutoModel

    model = normalize_model(model)
    handle = AutoModel.from_pretrained(
        _require_snapshot(model), trust_remote_code=True, dtype=getattr(torch, dtype)
    ).to(device).eval()
    return D1Engine(handle, MODELS[model])


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    Wired for the catalog d1 ids; unreachable through the server while the
    catalog keeps runner: von (see the module docstring)."""
    shorthand = MODEL_IDS.get(model_id)
    if shorthand is None:
        raise KeyError(f"unknown model_id: {model_id!r} (known: {sorted(MODEL_IDS)})")
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        engine = _ENGINE.get(shorthand)
        if engine is None:
            engine = load_engine(shorthand)
            _ENGINE[shorthand] = engine
    out = engine.predict(state, {"q": question})
    return _systemone.systemone_result(question["type"], out["answers"]["q"])
