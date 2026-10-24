"""The island's app under TestClient with the engines stubbed — no weights.

Ports the offline import-hygiene idea of the legacy demo tests (the
systemone-local island's test_server.py is the pattern): the app factory,
the model registry and the dispatch must all work with
torch/transformers blocked, because nothing may load at import time.
The dispatch is monkeypatched to a stub, so the engine modules and their
heavy stacks are never imported: /healthz, /info, track routing, the
{prediction, prob} contract, per-item error isolation and bare-value
wrapping (runner_protocol create_app semantics)."""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import patch

import pytest
import small_decisions.server as server_module
from fastapi.testclient import TestClient

ITEM = {
    "item_id": "mp-001",
    "input": {"text": "I absolutely love this playlist.", "images": []},
    "question": {"type": "choice", "options": ["positive", "negative", "neutral"]},
}

# Transcribed from the catalog yamls (model_id = the catalog id, version =
# the yaml's version id) in the registry's order.
EXPECTED_MODEL_IDS = [
    "vllm-sr/decision-2.0-kai",
    "vllm-sr/decision-2.0-eos",
    "vllm-sr/decision-2.0-sol",
    "vllm-sr/d3-edge",
    "vllm-sr/d3-lite",
    "vllm-sr/d3-nano",
    "internlm/intern-decision-0.8b",
    "internlm/intern-decision-2b",
    "internlm/intern-decision-4b",
    "ifm/k2-type",
    "molemo-lab/mojev",
    "vllm-sr/decision",
    "moritzlaurer/nli-deberta-v3-large",
    "moritzlaurer/nli-modernbert-large",
    "alibiserikbay/jevk5-4b",
    "eldanring/winnow-e4b",
    "madani/decima",
    "madani/decima-small",
    "logitlab/snap",
    "codekins/arbiter",
    "txoka/credence",
    "liquid/d1-3b",
    "liquid/d1-omni-600m",
]

EXPECTED_VERSIONS = {
    "vllm-sr/decision-2.0-kai": "decision-2.0-kai-0.6b@unversioned",
    "vllm-sr/decision-2.0-eos": "decision-2.0-eos-0.8b@unversioned",
    "vllm-sr/decision-2.0-sol": "decision-2.0-sol-2b@unversioned",
    "vllm-sr/d3-edge": "d3-edge@unversioned",
    "vllm-sr/d3-lite": "d3-lite@unversioned",
    "vllm-sr/d3-nano": "d3-nano@unversioned",
    "internlm/intern-decision-0.8b": "intern-decision-0.8b@unversioned",
    "internlm/intern-decision-2b": "intern-decision-2b@unversioned",
    "internlm/intern-decision-4b": "intern-decision-4b@unversioned",
    "ifm/k2-type": "k2-type-0.9b@unversioned",
    "molemo-lab/mojev": "mojev@0c8695b6252f4205907433d4e196a94f032e60c3",
    "vllm-sr/decision": "decision-1.0@unversioned",
    "moritzlaurer/nli-deberta-v3-large": "deberta-v3-large@unversioned",
    "moritzlaurer/nli-modernbert-large": "modernbert-large@unversioned",
    "alibiserikbay/jevk5-4b": "jevk5-4b@unversioned",
    "eldanring/winnow-e4b": "winnow-e4b@unversioned",
    "madani/decima": "decima-base-2.0@unversioned",
    "madani/decima-small": "decima-small-1.1@unversioned",
    "logitlab/snap": "snap1-2b@unversioned",
    "codekins/arbiter": "arbiter-4b@unversioned",
    "txoka/credence": "credence-v1@unversioned",
    "liquid/d1-3b": "d1-3b@2026-10",
    "liquid/d1-omni-600m": "d1-omni-600m@2026-10",
}


def stub_predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """A deterministic stand-in for the registry's predict dispatch."""
    return {"prediction": payload["question"]["options"][0], "prob": 0.75}


@pytest.fixture()
def client():
    with patch.object(server_module, "predict", stub_predict):
        with TestClient(server_module.build_app()) as test_client:
            yield test_client


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


def test_info_lists_the_23_catalog_models(client):
    info = client.get("/info").json()
    assert info["runner"] == "small-decisions"
    assert info["protocol"] == "v1"
    assert [m["model_id"] for m in info["models"]] == EXPECTED_MODEL_IDS
    versions = {m["model_id"]: m["version"] for m in info["models"]}
    assert versions == EXPECTED_VERSIONS
    for model in info["models"]:
        assert model["runtime"]["device"] == "cpu"
    host = info["host"]
    assert host["hostname"] and host["cpu"] and host["os"]


def test_the_three_tracks_route_and_others_404(client):
    ok = {"model_id": "vllm-sr/decision-2.0-kai", "items": [ITEM]}
    assert client.post("/v1/decision", json=ok).status_code == 200
    assert client.post("/v1/score", json=ok).status_code == 200
    assert client.post("/v1/classify", json=ok).status_code == 200
    missing = client.post("/v1/ner", json=ok)
    assert missing.status_code == 404
    assert "unknown track" in missing.json()["detail"]


def test_decision_round_trip(client):
    response = client.post("/v1/decision", json={"model_id": "ifm/k2-type", "items": [ITEM]})
    assert response.status_code == 200
    body = response.json()
    assert body["model_id"] == "ifm/k2-type"
    assert body["usage"] is None
    assert body["compute_ms"] >= 0.0
    assert body["results"] == [
        {"item_id": "mp-001", "prediction": "positive", "prob": 0.75, "error": None}
    ]


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
                    "model_id": "molemo-lab/mojev",
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
                    "model_id": "vllm-sr/decision",
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
        from small_decisions import models as registry

        for name in (
            "_base",
            "_systemone",
            "decision2",
            "d3",
            "intern_decision",
            "k2type",
            "mojev",
            "ollaya",
            "d1",
            "julia",
            "certo",
            "rlcd",
        ):
            module = importlib.import_module(f"small_decisions.models.{name}")
            importlib.reload(module)
        importlib.reload(registry)
        importlib.reload(server_module)
    # leave a clean registry behind for the other tests
    importlib.reload(server_module)
