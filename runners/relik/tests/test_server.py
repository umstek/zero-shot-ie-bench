"""End-to-end test of the served app: the real runner_protocol factory over
stubbed pipelines — no weights, no network.

The protocol measures compute_ms itself, validates the request against the
generated TrackRequest model and captures per-item exceptions; these tests
pin the wire shapes from catalog/schema (runner-info.json, track-request.json,
track-response.json) as served by this island.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from test_relik_predict import span, stub_pipeline

import relik_runner.relik as relik_module
from relik_runner.relik import CATALOG_MODEL_ID, NER_MODEL_ID
from relik_runner.server import DEFAULT_PORT, RELIK_MODELS, RELIK_TRACKS, RUNNER, app


def seed(model_id: str, pipeline) -> None:
    """Pre-populate the engine's cache for a wire model (get_pipeline then
    never touches load_relik)."""
    repo, _ = relik_module.WIRE_MODELS[model_id]
    relik_module._PIPELINES[repo] = pipeline


class TestServer:
    def test_healthz(self):
        assert TestClient(app).get("/healthz").json() == {"ok": True}

    def test_info_reports_the_catalog_model_and_the_sibling(self):
        body = TestClient(app).get("/info").json()
        assert body["runner"] == RUNNER == "relik"
        assert body["protocol"] == "v1"
        assert body["models"] == RELIK_MODELS
        assert [model["model_id"] for model in body["models"]] == [
            CATALOG_MODEL_ID, NER_MODEL_ID]
        # the host fingerprint the bench matches against catalog/machines/
        assert set(body["host"]) == {"hostname", "cpu", "ram_gb", "gpus", "os"}

    def test_tracks_are_the_islands_bench_surface(self):
        assert RELIK_TRACKS == ["ner", "relation"]
        # decision routes belong to other islands
        assert TestClient(app).post("/v1/decision", json={
            "model_id": CATALOG_MODEL_ID,
            "items": [{"item_id": "q1", "input": "x", "question": {}}],
        }).status_code == 404

    def test_relation_track_through_the_protocol(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        body = TestClient(app).post("/v1/relation", json={
            "model_id": CATALOG_MODEL_ID,
            "items": [{"item_id": "r1", "input": "Alice works for Acme.",
                       "question": {}}],
        }).json()
        assert body["model_id"] == CATALOG_MODEL_ID
        assert body["compute_ms"] >= 0
        assert body["usage"] is None
        assert body["results"] == [{
            "item_id": "r1",
            "prediction": [{
                "subject": {"start": 0, "end": 5, "label": "--NME--", "text": "Alice"},
                "relation": "P108",
                "object": {"start": 16, "end": 20, "label": "--NME--", "text": "Acme"},
                "confidence": 0.93,
            }],
            "prob": None,
            "error": None,
        }]

    def test_ner_track_through_the_protocol(self):
        pipeline = stub_pipeline()
        pipeline.return_value = SimpleNamespace(
            spans=[span(0, 5, "person", "Alice")], triplets=[])
        seed(NER_MODEL_ID, pipeline)
        body = TestClient(app).post("/v1/ner", json={
            "model_id": NER_MODEL_ID,
            "items": [{"item_id": "n1", "input": "Alice works for Acme.",
                       "question": {}}],
        }).json()
        # the sibling's default track is ner (no question keys needed)
        assert body["results"] == [{
            "item_id": "n1",
            "prediction": [{"start": 0, "end": 5, "label": "person", "text": "Alice"}],
            "prob": None,
            "error": None,
        }]

    def test_unknown_model_id_maps_to_an_error_row(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        body = TestClient(app).post("/v1/relation", json={
            "model_id": "relik-ie/nope",
            "items": [{"item_id": "b1", "input": "x", "question": {}}],
        }).json()
        assert body["results"][0]["error"] is not None
        assert "unknown model_id" in body["results"][0]["error"]

    def test_one_bad_item_does_not_kill_the_batch(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        body = TestClient(app).post("/v1/relation", json={
            "model_id": CATALOG_MODEL_ID,
            "items": [
                {"item_id": "ok-1", "input": "Alice works for Acme.", "question": {}},
                {"item_id": "bad-1", "input": "   ", "question": {}},
                {"item_id": "ok-2", "input": "Apple CEO Tim Cook spoke.",
                 "question": {}},
            ],
        }).json()
        results = body["results"]
        assert [row["item_id"] for row in results] == ["ok-1", "bad-1", "ok-2"]
        assert results[0]["error"] is None and results[2]["error"] is None
        assert results[1]["prediction"] is None
        assert results[1]["error"] == (
            "ValueError: relik expects a non-empty text string as `input`"
        )

    def test_documented_default_port(self):
        assert DEFAULT_PORT == 8031  # README "Run", --port overrides

    def test_the_pipeline_cache_starts_empty_on_the_served_app(self):
        # importing the app must never touch the HF cache or torch: the
        # stubs above are the only thing that ever populates _PIPELINES in
        # a test run
        assert relik_module._PIPELINES == {}
