"""The island's served models: catalog identity, registry, dispatch.

Every entry transcribes a catalog yaml whose local deployment assigns
`runner: small-decisions` (model_id = the catalog model id, version = the
exact version id the run identity tuple records, runtime = the
deployment's runtime dict echoed):

- vllm-sr/decision-2.0-{kai,eos,sol}          -> models/decision2.py
- vllm-sr/d3-{edge,lite,nano}                 -> models/d3.py
- ifm/k2-type                                 -> models/k2type.py
- molemo-lab/mojev                            -> models/mojev.py
- internlm/intern-decision-{0.8b,2b,4b}       -> models/intern_decision.py
- vllm-sr/decision + the nine daemon models   -> models/ollaya.py (HTTP
  proxies; the Ollaya daemon does the serving)
- liquid/d1-3b and liquid/d1-omni-600m         -> models/d1.py (the catalog
  routing moved from runner: von to this island — the code lives here and
  the 5.17 stack covers d1's pins)

Deliberately NOT registered: julia/certo/rlcd (no catalog yamls exist —
they are not v0 system-table systems; models/{julia,certo,rlcd}.py carry
the ported engines for the day they gain catalog identity).

Module surfaces: multi-size engines and the Ollaya proxies predict
(model_id, payload); single-engine modules predict(payload). _PREDICT
records which shape each entry takes.

Importing this package (or server.py) imports the engine modules but
never their heavy stacks — every weight load happens inside predict().
"""

from __future__ import annotations

from typing import Any

from . import d1, decision2, d3, intern_decision, k2type, mojev, ollaya

MODELS: list[dict[str, Any]] = [
    *decision2.ENTRIES,
    *d3.ENTRIES,
    *intern_decision.ENTRIES,
    k2type.ENTRY,
    mojev.ENTRY,
    *ollaya.ENTRIES,
    *d1.CATALOG_ENTRIES,
]

# model_id -> (predict fn, takes model_id?)
_PREDICT: dict[str, tuple[Any, bool]] = {
    **{model_id: (decision2.predict, True) for model_id in decision2.MODEL_IDS},
    **{model_id: (d3.predict, True) for model_id in d3.MODEL_IDS},
    **{model_id: (intern_decision.predict, True) for model_id in intern_decision.MODEL_IDS},
    **{model_id: (ollaya.predict, True) for model_id in ollaya.TAGS},
    **{model_id: (d1.predict, True) for model_id in d1.MODEL_IDS},
    k2type.MODEL_ID: (k2type.predict, False),
    mojev.MODEL_ID: (mojev.predict, False),
}


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one item to its engine: {input, question} -> {prediction, prob}.

    The protocol factory (runner_protocol.create_app) catches exceptions
    per item, so an engine whose weights are absent fails its items with a
    named error instead of taking the run down — check /info and each
    module's available() to know before the run.
    """
    entry = _PREDICT.get(model_id)
    if entry is None:
        raise KeyError(f"unknown model_id: {model_id!r} (served: {sorted(_PREDICT)})")
    fn, takes_model_id = entry
    if takes_model_id:
        return fn(model_id, payload)
    return fn(payload)
