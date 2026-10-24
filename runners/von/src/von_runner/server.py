"""The von runner's FastAPI app: one model (wfzyx/von) on the shared protocol.

`GET /healthz`, `GET /info`, `POST /v1/decision` and `POST /v1/score`
(catalog/models/wfzyx/von.yaml claims decision + score). The model list
transcribes that yaml's single local deployment (von-1.0.local) so /info
matches the catalog identity exactly: model_id "wfzyx/von" + version
"von-1.0@<revision>".

No Gradio UI: the legacy von demo's interactive tour is served by
apps/playground against this API. Weights load lazily on the first
predict — importing this module never touches torch or the HF cache.

Run: `uv run uvicorn von_runner.server:app --host 127.0.0.1 --port 8030`
or `uv run python -m von_runner.server [--host H] [--port P]`.
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from runner_protocol import create_app as make_runner_app

from .von import MODEL_REVISION, SERVED_MODEL_ID, predict

# Transcribed from catalog/models/wfzyx/von.yaml, deployment von-1.0.local
# (runtime: engine torch, dtype null, quant null, device cpu). Kept as a
# constant rather than a runtime yaml read: the island is a self-contained
# uv project (Docker builds it without the catalog tree).
VON_MODELS: list[dict[str, object]] = [
    {
        "model_id": SERVED_MODEL_ID,
        "version": f"von-1.0@{MODEL_REVISION}",
        "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"},
    }
]

VON_TRACKS = ["decision", "score"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8030


def create_app() -> FastAPI:
    return make_runner_app(
        runner="von", models=VON_MODELS, predict=predict, tracks=VON_TRACKS, ui=None
    )


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve wfzyx/von-1.0 on the runner protocol.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
