"""In-process loader for LiquidAI's open d1 decision models (local snapshots).

Open d1 (released 2026-10-05; blog "open d1") is the open-weights branch of
the same line as the hosted `liquid/d1` this repo already benches on
OpenRouter: single-pass decision models that take a state (text, JSON, or
images) plus named noul/choice/score questions and answer every question
from the model's distribution over the options — zero output tokens, no
generation, calibrated probabilities. Two checkpoints here:

- d1-3B (3.12B on LFM2.5-VL-3B; SigLIP2 NaFlex 400M vision encoder; 32,768
  ctx; 17 languages; card-claimed best-under-10B on Decision Index 0.2.1)
- d1-omni-600M (587M: a 381M trunk on LFM2.5-Encoder-350M + 94M SigLIP2
  vision + 112M FastConformer audio; 16,384 ctx; text, images and <=30 s
  speech in one state — with images the text budget is trained at 896
  tokens)

Both load through the card's own route — transformers AutoModel with
trust_remote_code=True (auto_map -> modeling_d1.D1Model over the lfm2_vl
backbone) — which needs transformers >=5.14, so this runs in .venv-von
(5.17.0 + torch 2.14.0+cpu, torchvision present for the vision tower).
Weights are LFM Open License v1.0 (lfm1.0), the same license family as
the benched LFM2.5-RLCD 350M weights.

Homes: D1_HOME (a prefix directory; each model's snapshot sits at
<prefix>/<dir-name>) or the default C:\\src base — engines resolve to
C:\\src\\d1-3B and C:\\src\\d1-omni-600M (the repo's own D: drive is
full; K2-Type, Intern-Decision and Decision 2.0 live on C: for the same
reason). fp32 on the CPU — the house dtype (the k2type note about bf16
CPU forwards buying nothing applies to the neighbors too).

predict(state, questions, images=None) is the card's system_one contract
minus HTTP plumbing; questions map ids to {"type", "instructions",
"criteria"}:

    engine = d1_client.load_engine()
    engine.predict("some text", {"q": {"type": "choice",
                                       "instructions": "...",
                                       "criteria": {label: desc}}})
    -> {"answers": {"q": {"choice": ..., "probabilities": {...},
                          "confidence": ...}},
        "model": "d1-3B", "input_tokens": 123}

Answer shapes are d1's own: noul -> {"noul": p_true}; choice ->
{choice, confidence, probabilities}; score -> {score (expected level),
confidence, probabilities, legend}. state may be None when images carry
the whole request (d1-3B's photo-of-a-form pattern).
"""

from __future__ import annotations

import os

# benchmark/demo shorthand -> snapshot directory name under the home prefix
MODELS = {
    "d1-3b": "d1-3B",
    "d1-omni-600m": "d1-omni-600M",
}
DEFAULT_MODEL = "d1-3b"


def normalize_model(model) -> str:
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
    raise ValueError(f"unknown d1 model {model!r} "
                     f"(one of {', '.join(MODELS)})")


def model_dir(model=DEFAULT_MODEL) -> str:
    """Snapshot directory: D1_HOME prefix, default C:\\src."""
    base = os.environ.get("D1_HOME", r"C:\src")
    return os.path.join(base, MODELS[normalize_model(model)])


def _require_snapshot(model=DEFAULT_MODEL) -> str:
    home = model_dir(model)
    if not os.path.isfile(os.path.join(home, "config.json")):
        raise FileNotFoundError(
            f"no {MODELS[normalize_model(model)]} snapshot at {home} — "
            f"pull it with `hf download LiquidAI/{MODELS[normalize_model(model)]}"
            f" --local-dir {home}` (or point D1_HOME at the prefix directory)")
    return home


class D1Engine:
    """One loaded d1 model; predict() answers every question of a request
    from one forward pass (the card's system_one, warmed by the caller)."""

    def __init__(self, handle, name: str):
        self._model = handle
        self.name = name

    def predict(self, state, questions: dict, images=None,
                audio=None) -> dict:
        """One forward over state+questions; images=[PIL] and audio= (16 kHz
        mono int16 numpy array, d1-omni-600M only) ride the same call."""
        kwargs = {}
        if images:
            kwargs["images"] = images
        if audio is not None:
            kwargs["audio"] = audio
        out = self._model.system_one(state, questions, **kwargs)
        return {"answers": out["answers"], "model": self.name,
                "input_tokens": out["usage"]["input_tokens"]}

    def predict_batch(self, requests: list) -> list:
        """The card's packed no-padding batch: [(state, questions), ...]
        (a third tuple element is accepted as the images list)."""
        packed = [tuple(r) if len(r) == 3 else (r[0], r[1], None)
                  for r in requests]
        if not any(imgs for _, _, imgs in packed):
            packed = [(state, qs) for state, qs, _ in packed]
        out = self._model.system_one_batch(packed)
        return [{"answers": row["answers"], "model": self.name,
                 "input_tokens": row["usage"]["input_tokens"]}
                for row in out]


def load_engine(model=DEFAULT_MODEL, device: str = "cpu",
                dtype: str = "float32") -> D1Engine:
    """D1Engine over the local snapshot (fp32 on the CPU).

    Heavy imports stay inside (offline tests import this module freely).
    The card's bf16 checkpoints upcast to fp32 here; a cold load of d1-3B
    moves ~6.3 GB of weights before the first forward."""
    import torch

    from transformers import AutoModel

    model = normalize_model(model)
    handle = AutoModel.from_pretrained(
        _require_snapshot(model), trust_remote_code=True,
        dtype=getattr(torch, dtype)).to(device).eval()
    return D1Engine(handle, MODELS[model])
