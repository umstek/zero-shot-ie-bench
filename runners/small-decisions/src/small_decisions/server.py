"""The small-decisions runner app.

One FastAPI app (runner_protocol.create_app) serves the island's
catalog-assigned decision models under POST /v1/decision, /v1/score and
/v1/classify (the NLI classifiers claim classify; the System One engines
claim decision + score), plus GET /info and GET /healthz. There is no
Gradio UI here: these engines' self-host UIs were the legacy demos; the
playground app covers them.

Start:

    uv run --project runners/small-decisions python -m small_decisions.server
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from runner_protocol import create_app

from .models import MODELS, predict

# decision + score: every catalog model's claim here; classify: the two
# MoritzLaurer NLI classifiers (catalog claims classify + text).
TRACKS = ["decision", "classify", "score"]


def build_app() -> FastAPI:
    """The island's app: no Gradio UI, nothing heavy at import time."""
    return create_app(runner="small-decisions", models=MODELS, predict=predict, tracks=TRACKS)


app = build_app()


def main() -> None:
    """uvicorn entry point (`python -m small_decisions.server`)."""
    import uvicorn  # noqa: PLC0415 — keep the module import server-agnostic

    uvicorn.run(
        app,
        host=os.environ.get("RUNNER_HOST", "127.0.0.1"),
        port=int(os.environ.get("RUNNER_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
