"""The island's model registry: catalog deployment ids -> model modules.

Every entry mirrors a catalog deployment with `runner: gliner`
(catalog/models/{knowledgator,jackboyla,convaiinnovations}/*.yaml): the
model_id is the deployment id, the version the catalog version id, runtime
the catalog runtime block (torch, fp32-weights, CPU — the shape every
deployment in this family declares).

`predict` is the island's per-item entry point for the runner protocol:
predict(model_id, payload) with payload = the item's {"input", "question",
"params"}. The track is not echoed by the protocol, so it is inferred from
the question shape (the shapes the v0 benches used — see resolve_track).
Weights load lazily on first use through the module's loader; `load` errors
and track mismatches surface as per-item error strings, never as a failed
request.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

_MODULES = ("gliformer", "gliner_relex", "glirel", "gliclass", "laya")
gliformer, gliner_relex, glirel, gliclass, laya = (
    importlib.import_module(f"gliner_runner.models.{name}") for name in _MODULES)


@dataclass(frozen=True)
class ModelSpec:
    """One served checkpoint. `loader` names the module function that
    produces the model object (laya.router uses the Router builder)."""

    model_id: str
    version: str
    hf_repo: str
    module: Any
    tracks: tuple[str, ...]
    loader: str = "load"
    runtime: dict[str, Any] = field(
        default_factory=lambda: {"engine": "torch", "dtype": None, "quant": None,
                                 "device": "cpu"})


# The catalog slice this island serves, in catalog file order.
_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("gliformer-base-v1.local", "gliformer-base-v1@unversioned",
              "knowledgator/gliformer-base-v1", gliformer, ("classify", "ner")),
    ModelSpec("gliformer-large-v1.local", "gliformer-large-v1@unversioned",
              "knowledgator/gliformer-large-v1", gliformer, ("classify", "ner")),
    ModelSpec("gliner-relex-multi-v1.0.local", "gliner-relex-multi-v1.0@unversioned",
              "knowledgator/gliner-relex-multi-v1.0", gliner_relex,
              ("ner", "relation", "classify")),
    ModelSpec("glirel-large-v0.local", "glirel-large-v0@unversioned",
              "jackboyla/glirel-large-v0", glirel, ("relation",)),
    ModelSpec("gliclass-base-v3.0.local", "gliclass-base-v3.0@unversioned",
              "knowledgator/gliclass-base-v3.0", gliclass, ("classify",)),
    ModelSpec("gliclass-modern-base-v3.0.local", "gliclass-modern-base-v3.0@unversioned",
              "knowledgator/gliclass-modern-base-v3.0", gliclass, ("classify",)),
    ModelSpec("gliclass-large-v3.0.local", "gliclass-large-v3.0@unversioned",
              "knowledgator/gliclass-large-v3.0", gliclass, ("classify",)),
    ModelSpec("laya.local", "laya@unversioned", "convaiinnovations/laya",
              laya, ("decision",)),
    ModelSpec("laya.router", "laya@unversioned", "convaiinnovations/laya",
              laya, ("decision",), loader="load_router"),
    ModelSpec("laya-typed-decisions.local", "laya-typed-decisions@unversioned",
              "convaiinnovations/laya-typed-decisions", laya, ("decision",)),
)

MODELS: dict[str, ModelSpec] = {spec.model_id: spec for spec in _REGISTRY}

# The create_app models table: {model_id, version, runtime} per served model.
SERVED_MODELS: list[dict[str, Any]] = [
    {"model_id": spec.model_id, "version": spec.version, "runtime": spec.runtime}
    for spec in _REGISTRY
]


def resolve_track(spec: ModelSpec, question: dict[str, Any]) -> str:
    """Pick the track a question is asking for, constrained to the model's
    capabilities. The protocol does not echo the track to predict(), so the
    shapes the v0 benches used decide:

    - a Jev typed question ("type": choice/score/noul) is always decision
    - "relations"/"relation_labels"/"ner" keys ask for relation extraction
    - {"labels": {label: description}} asks for classification (the benches
      pass label DESCRIPTIONS as a dict and the engines read the bare keys)
    - {"labels": [str, ...]} asks for NER — or classification on a
      classify-only model (gliclass scored bare label lists too)

    Anything unrecognised falls back to the model's first capability.
    """
    qtype = question.get("type")
    if isinstance(qtype, str) and qtype in ("choice", "score", "noul"):
        return "decision"
    inferred = None
    if any(key in question for key in ("relations", "relation_labels", "ner")):
        inferred = "relation"
    elif isinstance(question.get("labels"), dict):
        inferred = "classify"
    elif isinstance(question.get("labels"), list):
        inferred = "ner"
    if inferred in spec.tracks:
        return inferred
    if inferred == "ner" and "classify" in spec.tracks:
        return "classify"
    return spec.tracks[0]


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Answer one item for the runner protocol.

    payload is the item's {"input", "question", "params"} (the protocol
    strips item_id). Returns {"prediction", "prob", "error"} — per-item
    errors for everything item-level (unknown model, track mismatch, load
    failure, inference exception) so one bad item cannot kill a request;
    the protocol captures the same shape anyway.
    """
    spec = MODELS.get(model_id)
    if spec is None:
        return {"prediction": None, "prob": None,
                "error": f"unknown model_id: {model_id!r}"}
    question = payload.get("question") or {}
    try:
        track = resolve_track(spec, question)
        run = getattr(spec.module, f"run_{track}", None)
        if run is None:
            raise ValueError(f"model {model_id!r} has no {track!r} implementation")
        model = getattr(spec.module, spec.loader)(spec.hf_repo)
        prediction, prob = run(model, payload.get("input"), question,
                               payload.get("params") or {})
        return {"prediction": prediction, "prob": prob, "error": None}
    except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
        return {"prediction": None, "prob": None,
                "error": f"{type(exc).__name__}: {exc}"}
