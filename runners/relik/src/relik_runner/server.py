"""The relik runner's FastAPI app: the two ReLiK pipelines on the shared protocol.

`GET /healthz`, `GET /info`, `POST /v1/ner` and `POST /v1/relation`
(catalog/models/relik-ie/relik.yaml claims relation + ner). The model list
transcribes that yaml's single local deployment (relik-relation-
extraction-small.local -> model_id "relik-ie/relik", version
"relik-relation-extraction-small@unversioned") plus the demoed NER
sibling, keyed by its HF repo id (no catalog deployment — see README).

No Gradio UI: the legacy relik demo's tour is served by apps/playground
against this API. Pipelines load lazily on the first predict — importing
this module never touches torch or the HF cache. Importing this package
DOES install the Windows csv shim (relik_runner.relik, module import) —
that is what lets relik itself be imported anywhere in the process.

Run: `uv run uvicorn relik_runner.server:app --host 127.0.0.1 --port 8031`
or `uv run python -m relik_runner.server [--host H] [--port P]`.
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI
from runner_protocol import create_app as make_runner_app

from .relik import CATALOG_MODEL_ID, NER_MODEL_ID, predict

RUNNER = "relik"

# Transcribed from catalog/models/relik-ie/relik.yaml, deployment
# relik-relation-extraction-small.local (runtime: engine torch, dtype null,
# quant null, device cpu) + the NER sibling the yaml documents as demoed
# only. Kept as constants rather than a runtime yaml read: the island is a
# self-contained uv project (Docker builds it without the catalog tree).
RELIK_MODELS: list[dict[str, object]] = [
    {
        "model_id": CATALOG_MODEL_ID,
        "version": "relik-relation-extraction-small@unversioned",
        "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"},
    },
    {
        "model_id": NER_MODEL_ID,
        "version": "relik-relation-extraction-ner-small@unversioned",
        "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"},
    },
]

RELIK_TRACKS = ["ner", "relation"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8031


def create_app() -> FastAPI:
    return make_runner_app(
        runner=RUNNER, models=RELIK_MODELS, predict=predict, tracks=RELIK_TRACKS, ui=None
    )


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve relik-ie ReLiK on the runner protocol.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
