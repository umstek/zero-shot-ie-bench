"""The systemone-local runner app.

One FastAPI app (runner_protocol.create_app) serves the island's four
System One-contract decision engines under POST /v1/decision, plus
GET /info and GET /healthz. There is no Gradio UI here: these engines'
self-host UIs were the legacy demos; the playground app covers them.

Start (historical ports for reference — the native System One servers
each owned 8009/8018/8029 and AgentJev 8149; this island serves all four
models on one port, default 8000):

    uv run --project runners/systemone-local python -m systemone_local.server
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from runner_protocol import create_app

from .models import MODELS, predict

TRACKS = ["decision"]  # the catalog's claimed track for all four models


def build_app() -> FastAPI:
    """The island's app: no Gradio UI, nothing heavy at import time."""
    return create_app(runner="systemone-local", models=MODELS, predict=predict, tracks=TRACKS)


app = build_app()


def main() -> None:
    """uvicorn entry point (`python -m systemone_local.server`)."""
    import uvicorn  # noqa: PLC0415 — keep the module import server-agnostic

    uvicorn.run(
        app,
        host=os.environ.get("RUNNER_HOST", "127.0.0.1"),
        port=int(os.environ.get("RUNNER_PORT", "8000")),
    )


if __name__ == "__main__":
    main()
