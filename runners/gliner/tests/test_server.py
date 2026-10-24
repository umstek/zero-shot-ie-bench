"""End-to-end test of the served app: the real runner_protocol factory over
stubbed model caches — no weights, no network.

The protocol measures compute_ms itself, validates the request against the
generated TrackRequest model and captures per-item exceptions; these tests
pin the wire shapes from catalog/schema (runner-info.json, track-request.json,
track-response.json) as served by this island.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from gliner_runner.models import gliclass, gliformer, gliner_relex, laya
from gliner_runner.server import RUNNER, SERVED_MODELS, TRACKS, app


class TestServer:
    def test_healthz(self):
        client = TestClient(app)
        assert client.get("/healthz").json() == {"ok": True}

    def test_info_shape(self):
        body = TestClient(app).get("/info").json()
        assert body["runner"] == RUNNER == "gliner"
        assert body["protocol"] == "v1"
        assert body["models"] == SERVED_MODELS
        assert len(body["models"]) == 10
        # the host fingerprint the bench matches against catalog/machines/
        assert set(body["host"]) == {"hostname", "cpu", "ram_gb", "gpus", "os"}

    def test_unknown_track_404s(self):
        # glirel is relation-only and the route set is the island's bench
        # surface (["ner", "classify", "decision"]) — see README known gap
        client = TestClient(app)
        assert client.post("/v1/relation", json={
            "model_id": "glirel-large-v0.local",
            "items": [{"item_id": "q1", "input": "x", "question": {}}],
        }).status_code == 404

    def test_classify_through_the_protocol(self, monkeypatch):
        def pipe(text, labels, threshold=0.0):
            return [[{"label": "positive", "score": 0.83}]]

        monkeypatch.setitem(gliclass._LOADED, "knowledgator/gliclass-base-v3.0", pipe)
        body = TestClient(app).post("/v1/classify", json={
            "model_id": "gliclass-base-v3.0.local",
            "items": [{"item_id": "q1", "input": "I love it",
                       "question": {"labels": {"positive": "d", "negative": "d"}}}],
        }).json()
        assert body["model_id"] == "gliclass-base-v3.0.local"
        assert body["results"] == [{"item_id": "q1", "prediction": "positive",
                                    "prob": 0.83, "error": None}]
        assert body["compute_ms"] >= 0
        assert body["usage"] is None

    def test_ner_through_the_protocol(self, monkeypatch):
        def predict_entities(text, labels, threshold):
            assert threshold == 0.5
            return [{"start": 0, "end": 5, "label": "person"},
                    {"start": 7, "end": 11, "label": "company"}]

        model = SimpleNamespace(predict_entities=predict_entities)
        monkeypatch.setitem(gliformer._LOADED, "knowledgator/gliformer-base-v1", model)
        body = TestClient(app).post("/v1/ner", json={
            "model_id": "gliformer-base-v1.local",
            "items": [{"item_id": "s1", "input": "Alice joined Acme",
                       "question": {"labels": ["person", "company"]}}],
        }).json()
        # the server emits (start, end, label) tuples; JSON turns them into
        # arrays on the wire
        assert body["results"][0] == {
            "item_id": "s1", "prediction": [[0, 5, "person"], [7, 11, "company"]],
            "prob": None, "error": None}

    def test_decision_through_the_protocol(self, monkeypatch):
        def predict(state, questions):
            assert state == {"text": "The food was cold."}
            assert questions["q"]["type"] == "choice"
            return {"answers": {"q": {"choice": "negative", "confidence": 0.71}}}

        agent = SimpleNamespace(predict=predict)
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya", agent)
        body = TestClient(app).post("/v1/decision", json={
            "model_id": "laya.local",
            "items": [{"item_id": "d1", "input": "The food was cold.",
                       "question": {"type": "choice", "instructions": "i",
                                    "criteria": {"positive": None, "negative": None}}}],
        }).json()
        assert body["results"][0]["prediction"] == "negative"
        assert body["results"][0]["prob"] == 0.71

    def test_relation_dispatch_runs_off_the_question_shape(self, monkeypatch):
        # no /v1/relation route yet (see README known gap) — the relex
        # relations handler is still reachable through predict() because the
        # question shape decides the track, which is what this pins
        def inference(**kwargs):
            return (None, [[{"relation": "works for",
                             "head": {"text": "Ada", "type": "person"},
                             "tail": {"text": "Acme", "type": "company"},
                             "score": 0.81}]])

        model = SimpleNamespace(inference=inference)
        monkeypatch.setitem(gliner_relex._LOADED,
                            "knowledgator/gliner-relex-multi-v1.0", model)
        from gliner_runner.models import predict

        out = predict("gliner-relex-multi-v1.0.local", {
            "input": "Ada works for Acme",
            "question": {"entity_labels": ["person", "company"],
                         "relations": [{"name": "works for", "head": "person",
                                        "tail": "company"}]}})
        assert out == {"prediction": [
            {"relation": "works for", "head": "Ada", "head_type": "person",
             "tail": "Acme", "tail_type": "company", "score": 0.81}],
            "prob": None, "error": None}

    def test_one_bad_item_does_not_kill_the_request(self, monkeypatch):
        monkeypatch.setitem(gliclass._LOADED, "knowledgator/gliclass-base-v3.0",
                            lambda text, labels, threshold=0.0: [[]])
        body = TestClient(app).post("/v1/classify", json={
            "model_id": "gliclass-base-v3.0.local",
            "items": [{"item_id": "ok", "input": "fine",
                       "question": {"labels": ["a"]}},
                      {"item_id": "bad", "input": 12345,  # not text
                       "question": {"labels": ["a"]}}],
        }).json()
        assert [r["item_id"] for r in body["results"]] == ["ok", "bad"]
        assert body["results"][0]["error"] is None
        assert body["results"][1]["prediction"] is None
        assert body["results"][1]["error"] is not None

    def test_unknown_model_is_a_per_item_error_over_http(self):
        body = TestClient(app).post("/v1/classify", json={
            "model_id": "no-such-model.local",
            "items": [{"item_id": "q1", "input": "x", "question": {}}],
        }).json()
        assert body["model_id"] == "no-such-model.local"
        assert "unknown model_id" in body["results"][0]["error"]

    def test_request_validation_rejects_items_without_input(self):
        # catalog/schema/track-request.json: item_id, input, question required
        response = TestClient(app).post("/v1/classify", json={
            "model_id": "gliclass-base-v3.0.local",
            "items": [{"item_id": "q1", "input": "x"}],  # question missing
        })
        assert response.status_code == 422

    def test_tracks_module_constant_matches_the_protocol_routes(self):
        assert TRACKS == ["ner", "classify", "decision"]
