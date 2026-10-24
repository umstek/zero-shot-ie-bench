"""FastAPI app factory shared by every runner island.

One process serves the bench API and (optionally) the self-host Gradio UI:
`GET /healthz`, `GET /info`, `POST /v1/<track>` per docs/plan/02 §"Local
runners". `compute_ms` is measured inside the handler so HTTP overhead never
pollutes latency numbers.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, HTTPException

from .host import host_fingerprint
from .schema_models import RunnerInfo, TrackRequest, TrackResponse

PredictFn = Callable[[str, dict[str, Any]], Any]
"""predict(model_id, payload) -> result.

`payload` is the item's track-typed content: {"input", "question", "params",
...extras} (everything except `item_id`). The return value is either a dict
honouring the keys {"prediction", "prob", "error"} or a bare prediction value
(wrapped as {"prediction": <value>, "prob": None, "error": None}).
"""


def create_app(
    *,
    runner: str,
    models: list[dict[str, Any]],
    predict: PredictFn,
    tracks: list[str] = ["decision", "classify", "ner"],
    ui: Any = None,
) -> FastAPI:
    """Build the runner FastAPI app.

    runner: island name reported on /info (e.g. "gliner").
    models: entries shaped {model_id, version, runtime} — validated eagerly
        against the generated RunnerInfo model so bad input fails at startup.
    predict: called once per item; see PredictFn. Exceptions are captured
        per item as {"error": ...} so one bad item cannot kill a run.
    tracks: which /v1/<track> routes accept requests (others 404).
    ui: optional gr.Blocks/gr.App — mounted at "/" via gr.mount_gradio_app.
        gradio is imported lazily; islands without a UI never pay for it.
    """
    info = RunnerInfo(runner=runner, protocol="v1", models=models, host=host_fingerprint())

    app = FastAPI(title=f"runner:{runner}", version="1.0.0")

    @app.get("/healthz")
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/info")
    def get_info() -> RunnerInfo:
        return info

    @app.post("/v1/{track}")
    def run_track(track: str, request: TrackRequest) -> TrackResponse:
        if track not in tracks:
            raise HTTPException(status_code=404, detail=f"unknown track: {track!r}")
        start = time.perf_counter()
        results: list[dict[str, Any]] = []
        for item in request.items:
            payload = {key: value for key, value in item.model_dump().items() if key != "item_id"}
            try:
                out = predict(request.model_id, payload)
            except Exception as exc:  # noqa: BLE001 — per-item isolation by contract
                results.append({"item_id": item.item_id, "prediction": None, "prob": None, "error": str(exc)})
                continue
            if isinstance(out, dict):
                prediction = out.get("prediction")
                prob = out.get("prob")
                error = out.get("error")
            else:
                prediction, prob, error = out, None, None
            results.append({"item_id": item.item_id, "prediction": prediction, "prob": prob, "error": error})
        compute_ms = (time.perf_counter() - start) * 1000.0
        return TrackResponse(model_id=request.model_id, results=results, compute_ms=compute_ms, usage=None)

    if ui is not None:
        import gradio  # noqa: PLC0415 — lazy: heavy import, only needed with a UI; pyrefly: ignore[missing-import] (optional [ui] extra)

        gradio.mount_gradio_app(app, ui, path="/")

    return app
