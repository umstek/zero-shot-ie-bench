"""Runner-protocol tests: the FastAPI app over a stubbed backend.

Ported in spirit from legacy test_von_judge_rate.py's ServeJudgeRateTests
(the demo's stdin/stdout JSON service became the shared /v1/<track>
protocol) and test_von_pipe.py (the non-ascii round trip — over HTTP
UTF-8 now instead of a cp1252 pipe). No weights: get_backend resolves
through load_von_backend, which every test stubs.
"""

import sys
from unittest.mock import patch

from conftest import choice_answer, noul_answer, sdk_stubs, stub_backend
from fastapi.testclient import TestClient

import von_runner.von as von_module
from von_runner.server import DEFAULT_PORT, SERVED_MODEL_ID, VON_MODELS
from von_runner.server import app as von_app


def install_stub_backend(backend=None):
    """Point the lazy loader at a stub backend and keep the SDK modules
    stubbed so the wrappers never import the real (torch-heavy) package.
    Returns (backend, patchers) — stop the patchers in a finally."""
    backend = backend if backend is not None else stub_backend()
    modules, _, _, _ = sdk_stubs(backend=backend)
    patchers = [
        patch.dict(sys.modules, modules),
        patch.object(von_module, "load_von_backend", return_value=backend),
    ]
    for patcher in patchers:
        patcher.start()
    von_module.reset_backend()
    return backend, patchers


def stop_patchers(patchers):
    for patcher in reversed(patchers):
        patcher.stop()


def client():
    return TestClient(von_app)


def choice_item(item_id: str = "mp-001", text: str = "I love this playlist."):
    return {
        "item_id": item_id,
        "input": {"text": text, "images": []},
        "question": {"type": "choice", "options": ["positive", "negative", "neutral"]},
    }


class TestInfoAndHealth:
    def test_healthz_is_up(self):
        assert client().get("/healthz").json() == {"ok": True}

    def test_info_reports_the_catalog_model_and_version(self):
        info = client().get("/info").json()
        assert info["runner"] == "von"
        assert info["protocol"] == "v1"
        assert info["models"] == [
            {
                "model_id": "wfzyx/von",
                "version": "von-1.0@fbc0196efa4fa1825557a8d5adad464fa1e872bb",
                "runtime": {"engine": "torch", "dtype": None, "quant": None, "device": "cpu"},
            }
        ]
        assert info["host"]["hostname"]

    def test_catalog_constants_stay_in_sync_with_the_served_model(self):
        assert VON_MODELS[0]["model_id"] == SERVED_MODEL_ID
        assert DEFAULT_PORT == 8030  # documented default, --port overrides


class TestDecisionTrack:
    def test_choice_question_answers_with_the_label_and_confidence(self):
        backend, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision", json={"model_id": "wfzyx/von", "items": [choice_item()]}
            )
        finally:
            stop_patchers(patchers)
        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == "wfzyx/von"
        assert body["compute_ms"] >= 0
        result = body["results"][0]
        assert result["item_id"] == "mp-001"
        assert result["prediction"] == "positive"
        assert result["prob"] == 0.9
        assert result["error"] is None
        # the exact call path into the pinned SDK (the Choice stub returns
        # its kwargs, so the criteria dict is visible on the mock)
        backend.evaluate_choice.assert_called_once_with(
            "decision",
            "I love this playlist.",
            {
                "instructions": "Choose the best option.",
                "criteria": {"positive": None, "negative": None, "neutral": None},
            },
            temperature=1.0,
        )

    def test_choice_question_accepts_described_options_and_instructions(self):
        backend, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "d-1",
                            "input": "plain string state",
                            "question": {
                                "type": "choice",
                                "instructions": "What is the overall sentiment?",
                                "choices": {"positive": "clearly positive"},
                            },
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        assert response.json()["results"][0]["prediction"] == "positive"
        backend.evaluate_choice.assert_called_once_with(
            "decision",
            "plain string state",
            {
                "instructions": "What is the overall sentiment?",
                "criteria": {"positive": "clearly positive"},
            },
            temperature=1.0,
        )

    def test_noul_question_answers_with_the_verdict_and_probability(self):
        backend, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "n-1",
                            "input": {"text": "ticket"},
                            "question": {
                                "type": "noul",
                                "instructions": "Is the customer asking for a refund?",
                            },
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        result = response.json()["results"][0]
        # 0.87 >= 0.5 -> yes (legacy judgment_row threshold)
        assert result["prediction"] == "yes"
        assert result["prob"] == 0.87
        backend.evaluate_noul.assert_called_once_with(
            "judgment",
            "ticket",
            {"instructions": "Is the customer asking for a refund?"},
            temperature=1.0,
        )

    def test_noul_at_exactly_half_counts_as_yes(self):
        backend = stub_backend()
        backend.evaluate_noul.return_value = noul_answer(0.5)
        _, patchers = install_stub_backend(backend)
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "n-2",
                            "input": "s",
                            "question": {"type": "noul", "instructions": "i"},
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        assert response.json()["results"][0]["prediction"] == "yes"

    def test_score_question_answers_with_the_expectation(self):
        backend, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/score",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "r-1",
                            "input": {"text": "ticket"},
                            "question": {
                                "type": "score",
                                "instructions": "How urgent?",
                                "rubric": ["wait", "soon", "now"],
                            },
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        result = response.json()["results"][0]
        assert result["prediction"] == 2.4
        assert result["prob"] == 0.4
        backend.evaluate_score.assert_called_once_with(
            "rating",
            "ticket",
            {"instructions": "How urgent?", "criteria": ["wait", "soon", "now"]},
            temperature=1.0,
        )

    def test_one_bad_item_does_not_kill_the_batch(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        choice_item("ok-1"),
                        choice_item("bad-1", text="   "),
                        choice_item("ok-2"),
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        results = response.json()["results"]
        assert [r["item_id"] for r in results] == ["ok-1", "bad-1", "ok-2"]
        assert results[0]["error"] is None
        assert results[2]["error"] is None
        assert results[1]["prediction"] is None
        assert results[1]["error"] == (
            "ValueError: payload needs a non-empty 'input' (or 'input.text')"
        )


class TestErrorMapping:
    def test_unknown_track_is_a_404(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/ner", json={"model_id": "wfzyx/von", "items": [choice_item()]}
            )
        finally:
            stop_patchers(patchers)
        assert response.status_code == 404

    def test_unknown_model_id_maps_to_an_error_row(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision", json={"model_id": "wfzyx/von-1.0", "items": [choice_item()]}
            )
        finally:
            stop_patchers(patchers)
        error = response.json()["results"][0]["error"]
        assert error is not None and "unknown model_id" in error

    def test_unsupported_question_type_names_the_type(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [{"item_id": "x", "input": "t", "question": {"type": "ner"}}],
                },
            )
        finally:
            stop_patchers(patchers)
        error = response.json()["results"][0]["error"]
        assert error is not None and "ner" in error

    def test_noul_without_instructions_names_the_field(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [{"item_id": "x", "input": "t", "question": {"type": "noul"}}],
                },
            )
        finally:
            stop_patchers(patchers)
        error = response.json()["results"][0]["error"]
        assert error is not None and "instructions" in error

    def test_score_without_a_two_level_rubric_names_the_field(self):
        _, patchers = install_stub_backend()
        try:
            response = client().post(
                "/v1/score",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "x",
                            "input": "t",
                            "question": {
                                "type": "score",
                                "instructions": "i",
                                "rubric": ["only one"],
                            },
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        error = response.json()["results"][0]["error"]
        assert error is not None and "rubric" in error


class TestPipeParity:
    def test_non_ascii_choice_labels_round_trip_intact(self):
        # port of legacy test_von_pipe.py: non-ascii labels used to fight
        # Windows code pages on the demo's stdin/stdout pipe; over HTTP
        # the JSON is UTF-8 end to end and the label must come back
        # byte-identical.
        label = "你好"
        backend = stub_backend()
        backend.evaluate_choice.return_value = choice_answer(label, {label: 1.0}, 1.0)
        _, patchers = install_stub_backend(backend)
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": "wfzyx/von",
                    "items": [
                        {
                            "item_id": "zh-1",
                            "input": {"text": "测试"},
                            "question": {"type": "choice", "options": [label, "හොඳයි"]},
                        }
                    ],
                },
            )
        finally:
            stop_patchers(patchers)
        result = response.json()["results"][0]
        assert result["prediction"] == label
        assert result["prob"] == 1.0
