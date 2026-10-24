"""The island's served models: catalog identity, registry, dispatch.

MODEL_ID is the catalog deployment's version id short form
(catalog/models/{jaredpalmer,mapika,iapp,malevrigns}/*.yaml); VERSION is
the exact version id the run identity tuple records. Importing this
package (or server.py) imports the engine modules but never their heavy
stacks — every weight load happens inside predict().
"""

from __future__ import annotations

from typing import Any

from . import agentjev, decider, kev, openthai

MODELS: list[dict[str, Any]] = [
    {"model_id": engine.MODEL_ID, "version": engine.VERSION, "runtime": engine.RUNTIME}
    for engine in (kev, decider, openthai, agentjev)
]

_ENGINES = {engine.MODEL_ID: engine for engine in (kev, decider, openthai, agentjev)}


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one item to its engine: {input, question} -> {prediction, prob}.

    The protocol factory (runner_protocol.create_app) catches exceptions
    per item, so an engine whose weights are absent fails its items with a
    named error instead of taking the run down — check /info and each
    module's available() to know before the run.
    """
    engine = _ENGINES.get(model_id)
    if engine is None:
        raise KeyError(f"unknown model_id: {model_id!r} (served: {sorted(_ENGINES)})")
    return engine.predict(payload)
