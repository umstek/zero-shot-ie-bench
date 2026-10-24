"""The island's model registry: catalog model ids -> model modules.

Every entry mirrors a catalog deployment with `runner: gliner`
(catalog/models/{knowledgator,jackboyla,convaiinnovations}/*.yaml): the
served_id is the catalog MODEL id — the wire convention every island uses,
so the bench can drive the deployment directly — the alias also accepts the
catalog deployment id, the version the catalog version id, runtime the
catalog runtime block (torch, fp32-weights, CPU — the shape every
deployment in this family declares). The one model with two deployments on
this runner (convaiinnovations/laya: plain vs router mode) serves its
router deployment under the deployment id (`laya.router`), which is what
the bench sends for it.

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
    """One served checkpoint. `served_id` is the wire model_id — the CATALOG
    MODEL id (knowledgator/gliformer-base), the convention every island
    uses, so the bench can drive it directly; `alias` also accepts the
    catalog deployment id. The two Laya modes share one catalog model, so
    the router keeps its deployment id as its served_id."""

    served_id: str
    version: str
    hf_repo: str
    module: Any
    tracks: tuple[str, ...]
    alias: str = ""
    loader: str = "load"
    runtime: dict[str, Any] = field(
        default_factory=lambda: {"engine": "torch", "dtype": None, "quant": None,
                                 "device": "cpu"})


# The catalog slice this island serves. served_id = the catalog MODEL id
# (what the bench sends); alias = the catalog deployment id (also accepted).
_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec("knowledgator/gliformer-base", "gliformer-base-v1@unversioned",
              "knowledgator/gliformer-base-v1", gliformer, ("classify", "ner"),
              alias="gliformer-base-v1.local"),
    ModelSpec("knowledgator/gliformer-large", "gliformer-large-v1@unversioned",
              "knowledgator/gliformer-large-v1", gliformer, ("classify", "ner"),
              alias="gliformer-large-v1.local"),
    ModelSpec("knowledgator/gliner-relex", "gliner-relex-multi-v1.0@unversioned",
              "knowledgator/gliner-relex-multi-v1.0", gliner_relex,
              ("ner", "relation", "classify"), alias="gliner-relex-multi-v1.0.local"),
    ModelSpec("jackboyla/glirel", "glirel-large-v0@unversioned",
              "jackboyla/glirel-large-v0", glirel, ("relation",),
              alias="glirel-large-v0.local"),
    ModelSpec("knowledgator/gliclass-base", "gliclass-base-v3.0@unversioned",
              "knowledgator/gliclass-base-v3.0", gliclass, ("classify",),
              alias="gliclass-base-v3.0.local"),
    ModelSpec("knowledgator/gliclass-modern-base", "gliclass-modern-base-v3.0@unversioned",
              "knowledgator/gliclass-modern-base-v3.0", gliclass, ("classify",),
              alias="gliclass-modern-base-v3.0.local"),
    ModelSpec("knowledgator/gliclass-large", "gliclass-large-v3.0@unversioned",
              "knowledgator/gliclass-large-v3.0", gliclass, ("classify",),
              alias="gliclass-large-v3.0.local"),
    ModelSpec("convaiinnovations/laya", "laya@unversioned", "convaiinnovations/laya",
              laya, ("decision",), alias="laya.local"),
    # the router mode shares the catalog model above, so it keeps its
    # deployment id as the wire id (the bench targets it explicitly)
    ModelSpec("laya.router", "laya@unversioned", "convaiinnovations/laya",
              laya, ("decision",), loader="load_router"),
    ModelSpec("convaiinnovations/laya-typed-decisions",
              "laya-typed-decisions@unversioned",
              "convaiinnovations/laya-typed-decisions", laya, ("decision",),
              alias="laya-typed-decisions.local"),
)

MODELS: dict[str, ModelSpec] = {}
for _spec in _REGISTRY:
    MODELS.setdefault(_spec.served_id, _spec)
    if _spec.alias:
        MODELS.setdefault(_spec.alias, _spec)

# The create_app models table: {model_id, version, runtime} per served model.
SERVED_MODELS: list[dict[str, Any]] = [
    {"model_id": spec.served_id, "version": spec.version, "runtime": spec.runtime}
    for spec in _REGISTRY
]


def normalize_suite_payload(spec: ModelSpec, payload: dict[str, Any]):
    """Suites speak {input: {text}, question: {type, options}}; the engine
    modules speak (text, {labels: ...} | typed question). Translate so a
    suite item reaches inference unchanged in meaning:

    - input {"text": ...} (images unused on this island) -> the text string
    - a choice question on a classify-capable model -> {"labels": {opt: None}}
    - an options list on any model -> {"labels": [...]} (NER label set)
    - decision-only models (laya) keep their typed question as-is
    """
    raw_input = payload.get("input")
    if isinstance(raw_input, dict) and "decision" not in spec.tracks:
        input_value = raw_input.get("text", "")
    else:
        input_value = raw_input  # decision specs take rich states as-is
    question = dict(payload.get("question") or {})
    qtype = question.get("type")
    options = question.get("options")
    if qtype == "choice" and isinstance(options, list) and "classify" in spec.tracks:
        return input_value, {"labels": {opt: None for opt in options}}
    if isinstance(options, list) and not question.get("labels"):
        return input_value, {"labels": options}
    return input_value, question


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
    input_value, question = normalize_suite_payload(spec, payload)
    try:
        track = resolve_track(spec, question)
        run = getattr(spec.module, f"run_{track}", None)
        if run is None:
            raise ValueError(f"model {model_id!r} has no {track!r} implementation")
        model = getattr(spec.module, spec.loader)(spec.hf_repo)
        prediction, prob = run(model, input_value, question,
                               payload.get("params") or {})
        return {"prediction": prediction, "prob": prob, "error": None}
    except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
        return {"prediction": None, "prob": None,
                "error": f"{type(exc).__name__}: {exc}"}
