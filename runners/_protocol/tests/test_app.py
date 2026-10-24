"""Protocol tests with a stub model: no weights, no network (docs/plan/02)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from runner_protocol import RunnerInfo, TrackResponse, create_app

MODELS: list[dict[str, Any]] = [
    {
        "model_id": "stub-decider",
        "version": "rev-abc123",
        "runtime": {"engine": "torch", "dtype": "fp16", "device": "cpu", "batch": 1},
    }
]

REQUEST_BODY: dict[str, Any] = {
    "model_id": "stub-decider",
    "items": [{"item_id": "i-1", "input": {"text": "hello"}, "question": "is this good?"}],
}


def fake_predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    assert model_id == "stub-decider"
    assert payload["input"] == {"text": "hello"}
    return {"prediction": "yes", "prob": 0.87}


@pytest.fixture()
def client() -> Iterator[TestClient]:
    app = create_app(runner="stub", models=MODELS, predict=fake_predict)
    with TestClient(app) as test_client:
        yield test_client


def test_healthz(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_info_shape(client: TestClient) -> None:
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["runner"] == "stub"
    assert body["protocol"] == "v1"
    assert body["models"] == MODELS
    host = body["host"]
    assert host["hostname"]
    assert host["cpu"]
    assert host["ram_gb"] > 0
    assert isinstance(host["gpus"], list)
    assert host["os"]
    # round-trips through the generated wire model
    assert RunnerInfo.model_validate(body).model_dump()["runner"] == "stub"


def test_decision_round_trip(client: TestClient) -> None:
    response = client.post("/v1/decision", json=REQUEST_BODY)
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "stub-decider"
    assert body["results"] == [{"item_id": "i-1", "prediction": "yes", "prob": 0.87, "error": None}]
    assert isinstance(body["compute_ms"], (int, float))
    assert body["compute_ms"] >= 0.0
    assert body["usage"] is None
    assert TrackResponse.model_validate(body).compute_ms == body["compute_ms"]


def test_bare_prediction_is_wrapped() -> None:
    app = create_app(runner="stub", models=MODELS, predict=lambda model_id, payload: "no")
    with TestClient(app) as bare:
        response = bare.post("/v1/decision", json=REQUEST_BODY)
    assert response.status_code == 200
    assert response.json()["results"] == [{"item_id": "i-1", "prediction": "no", "prob": None, "error": None}]


def test_per_item_error_is_captured_not_raised() -> None:
    def broken_predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("OOM on item")

    app = create_app(runner="stub", models=MODELS, predict=broken_predict)
    with TestClient(app) as broken:
        response = broken.post("/v1/decision", json=REQUEST_BODY)
    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["error"] == "OOM on item"
    assert result["prediction"] is None and result["prob"] is None


def test_extra_keys_pass_through() -> None:
    seen: dict[str, Any] = {}

    def spy_predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen.update(payload)
        return {"prediction": 1, "prob": None}

    app = create_app(runner="stub", models=MODELS, predict=spy_predict)
    body = {**REQUEST_BODY, "items": [{**REQUEST_BODY["items"][0], "params": {"temp": 0}, "note": "extra"}], "run_tag": "x"}
    with TestClient(app) as spy:
        response = spy.post("/v1/decision", json=body)
    assert response.status_code == 200
    assert seen["params"] == {"temp": 0}
    assert seen["note"] == "extra"


def test_unknown_track_is_404(client: TestClient) -> None:
    response = client.post("/v1/rerank", json=REQUEST_BODY)
    assert response.status_code == 404


def test_validation_422(client: TestClient) -> None:
    missing_model_id = {"items": REQUEST_BODY["items"]}
    assert client.post("/v1/decision", json=missing_model_id).status_code == 422

    missing_input = {"model_id": "stub-decider", "items": [{"item_id": "i-1", "question": "q"}]}
    assert client.post("/v1/decision", json=missing_input).status_code == 422


def test_models_are_validated_at_startup() -> None:
    with pytest.raises(ValidationError):
        create_app(runner="stub", models=[{"model_id": "incomplete"}], predict=fake_predict)


def test_create_app_returns_fastapi() -> None:
    assert isinstance(create_app(runner="stub", models=MODELS, predict=fake_predict), FastAPI)
