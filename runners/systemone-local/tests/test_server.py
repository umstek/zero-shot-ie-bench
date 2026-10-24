"""The island's app under TestClient with the engines stubbed — no weights.

Ports the offline import-hygiene idea of legacy/tests
(test_systemone_locals_demo, test_verdict_and_openthai_demos): the app
factory, the model registry and the dispatch must all work with
torch/transformers blocked, because nothing may load at import time.
The per-engine engines themselves are monkeypatched to stubs, so this
exercises exactly the wiring: /healthz, /info, track routing, the
{prediction, prob} contract, per-item error isolation and bare-value
wrapping (runner_protocol create_app semantics). The stub replaces
server.predict — the callable create_app captured — so the engine
modules and their heavy stacks are never imported.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest
import systemone_local.server as server_module
from fastapi.testclient import TestClient

ITEM = {
    "item_id": "mp-001",
    "input": {"text": "I absolutely love this playlist.", "images": []},
    "question": {"type": "choice", "options": ["positive", "negative", "neutral"]},
}


def stub_predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """A deterministic stand-in for every engine module's predict()."""
    return {"prediction": payload["question"]["options"][0], "prob": 0.75}


@pytest.fixture()
def client():
    with patch.object(server_module, "predict", stub_predict):
        with TestClient(server_module.build_app()) as test_client:
            yield test_client


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_info_lists_the_four_catalog_models(client):
    info = client.get("/info").json()
    assert info["runner"] == "systemone-local"
    assert info["protocol"] == "v1"
    assert [m["model_id"] for m in info["models"]] == [
        "kev-0.8b",
        "decider-0.8b",
        "openthai-systemone",
        "agentjev-v1",
    ]
    versions = {m["model_id"]: m["version"] for m in info["models"]}
    assert versions == {
        "kev-0.8b": "kev-0.8b@2026-09",
        "decider-0.8b": "decider-0.8b@unversioned",
        "openthai-systemone": "openthai-systemone@unversioned",
        "agentjev-v1": "agentjev-v1@unversioned",
    }
    for model in info["models"]:
        assert model["runtime"]["engine"] == "torch"
        assert model["runtime"]["device"] == "cpu"
    host = info["host"]
    assert host["hostname"] and host["cpu"] and host["os"]


def test_decision_round_trip(client):
    response = client.post("/v1/decision", json={"model_id": "kev-0.8b", "items": [ITEM]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "kev-0.8b"
    assert body["usage"] is None
    assert body["compute_ms"] >= 0.0
    assert body["results"] == [
        {"item_id": "mp-001", "prediction": "positive", "prob": 0.75, "error": None}
    ]


def test_unknown_track_is_404(client):
    response = client.post("/v1/ner", json={"model_id": "kev-0.8b", "items": []})
    assert response.status_code == 404
    assert "unknown track" in response.json()["detail"]


def test_unknown_model_fails_its_items_not_the_run():
    """A bad model_id is one item's error — the run and the other items go on."""
    with TestClient(server_module.build_app()) as test_client:
        response = test_client.post(
            "/v1/decision",
            json={
                "model_id": "no-such-model",
                "items": [
                    {"item_id": "a", "input": "x", "question": {"type": "choice", "options": ["p", "n"]}},
                    {"item_id": "b", "input": "y", "question": {"type": "choice", "options": ["p", "n"]}},
                ],
            },
        )
    assert response.status_code == 200
    results = response.json()["results"]
    assert [r["item_id"] for r in results] == ["a", "b"]
    assert all("unknown model_id" in r["error"] for r in results)


def test_one_bad_item_does_not_kill_the_run():
    """Per-item isolation: an engine raising on one item errors that item only."""
    calls = {"n": 0}

    def flaky(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return {"prediction": "positive", "prob": 0.9}

    with patch.object(server_module, "predict", flaky):
        with TestClient(server_module.build_app()) as test_client:
            response = test_client.post(
                "/v1/decision",
                json={
                    "model_id": "decider-0.8b",
                    "items": [
                        {
                            "item_id": f"i{n}",
                            "input": "x",
                            "question": {"type": "choice", "options": ["positive", "negative"]},
                        }
                        for n in range(3)
                    ],
                },
            )
    assert response.status_code == 200
    results = response.json()["results"]
    assert results[0]["prediction"] == "positive"
    assert results[1]["prediction"] is None and "boom" in results[1]["error"]
    assert results[2]["prediction"] == "positive"


def test_bare_value_is_wrapped_by_the_protocol():
    """An engine returning a bare prediction is wrapped as {prediction, prob: None}."""
    with patch.object(server_module, "predict", lambda model_id, payload: "positive"):
        with TestClient(server_module.build_app()) as test_client:
            response = test_client.post(
                "/v1/decision",
                json={
                    "model_id": "openthai-systemone",
                    "items": [
                        {
                            "item_id": "i",
                            "input": "x",
                            "question": {"type": "choice", "options": ["positive", "negative"]},
                        }
                    ],
                },
            )
    assert response.json()["results"] == [
        {"item_id": "i", "prediction": "positive", "prob": None, "error": None}
    ]


def test_import_pulls_no_heavy_dependencies():
    """The module-hygiene rule the legacy demo tests police: importing the
    island (and re-importing every engine module) with torch/transformers
    blocked must succeed — nothing loads at import time."""
    with patch.dict(sys.modules, {"torch": None, "transformers": None}):
        from systemone_local import models as registry

        for name in ("_systemone", "_base", "kev", "decider", "openthai", "agentjev"):
            module = importlib.import_module(f"systemone_local.models.{name}")
            importlib.reload(module)
        importlib.reload(registry)
        importlib.reload(server_module)
    # leave a clean registry behind for the other tests
    importlib.reload(server_module)
