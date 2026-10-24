"""von runner island — wfzyx/von-1.0 on the transformers 5.x CPU stack.

Serves the complete, trained option-marker checkpoint (396M ModernBERT +
option-marker head + calibration) behind the shared runner protocol:
`POST /v1/decision` (choice / noul / score question types) and
`POST /v1/score`. One model, loaded lazily from the HF cache on the first
predict — never at import, never in tests unless already cached.

The package is `von_runner`, not `von`: the pinned von-sdk git dependency
installs a top-level `von` package and the island must not shadow it.
Entry points: `von_runner.von` (the engine, ported from legacy
engines/von_client.py) and `von_runner.server` (the FastAPI app,
`uvicorn von_runner.server:app`).
"""

__all__ = ["server", "von"]
