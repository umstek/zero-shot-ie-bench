"""Runner-protocol tests: the FastAPI app over a stubbed engine.

No weights: engine.load_model is patched to return a tiny randomly-
initialized NanoDiff (conftest.tiny_nanodiff — same class, same forward,
a test-sized config), so the lazy per-checkpoint singleton, the payload
parsing and the full format/softmax path all run over stub weights. The
port of the legacy demo's stdin/stdout JSON service contract (test
naming follows runners/von's TestClient suite).
"""

from unittest.mock import Mock, patch

from conftest import NEGATIVE_TEXT, SENTIMENT_OPTIONS, choice_item, tiny_nanodiff
from fastapi.testclient import TestClient

import nanodiff_runner.engine as engine
from nanodiff_runner.server import DEFAULT_PORT, NANODIFF_MODELS
from nanodiff_runner.server import app as nanodiff_app

V1_ID = "pngwn/nanodiff-350m-typed-decisions-lam1"
V2_ID = "pngwn/nanodiff-350m-typed-decisions-v2"


def install_stub_model():
    """Point the lazy loader at a tiny seeded NanoDiff. Returns (loader,
    patcher) — the Mock the lazy singleton resolves through (call-count
    assertions use it), and the patcher to stop in a finally."""
    model = tiny_nanodiff()
    loader = Mock(return_value=(model, 0))
    patcher = patch.object(engine, "load_model", loader)
    patcher.start()
    return loader, patcher


def client():
    return TestClient(nanodiff_app)


class TestInfoAndHealth:
    def test_healthz_is_up(self):
        assert client().get("/healthz").json() == {"ok": True}

    def test_info_reports_both_checkpoints(self):
        info = client().get("/info").json()
        assert info["runner"] == "nanodiff"
        assert info["protocol"] == "v1"
        assert info["models"] == [
            {
                "model_id": V1_ID,
                "version": "typed-decisions-lam1@step-3000",
                "runtime": {
                    "engine": "torch",
                    "dtype": "bfloat16",
                    "quant": None,
                    "device": "cpu",
                },
            },
            {
                "model_id": V2_ID,
                "version": "typed-decisions-v2@step-3000",
                "runtime": {
                    "engine": "torch",
                    "dtype": "bfloat16",
                    "quant": None,
                    "device": "cpu",
                },
            },
        ]
        assert info["host"]["hostname"]

    def test_served_constants_stay_in_sync(self):
        assert [m["model_id"] for m in NANODIFF_MODELS] == [V1_ID, V2_ID]
        assert DEFAULT_PORT == 8032  # documented default, --port overrides

    def test_decision_is_the_only_track(self):
        _, patcher = install_stub_model()
        try:
            assert (
                client()
                .post("/v1/score", json={"model_id": V1_ID, "items": [choice_item()]})
                .status_code
                == 404
            )
        finally:
            patcher.stop()


class TestDecisionTrack:
    def test_choice_question_answers_with_an_option_and_its_probability(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision", json={"model_id": V1_ID, "items": [choice_item()]}
            )
        finally:
            patcher.stop()
        assert response.status_code == 200
        body = response.json()
        assert body["model_id"] == V1_ID
        assert body["compute_ms"] >= 0
        result = body["results"][0]
        assert result["item_id"] == "cls-001"
        assert result["prediction"] in SENTIMENT_OPTIONS
        assert result["error"] is None
        # prob is the chosen option's answer-slot softmax mass
        assert 0.0 <= result["prob"] <= 1.0

    def test_weights_load_lazily_and_only_once_per_checkpoint(self):
        loader, patcher = install_stub_model()
        try:
            first = (
                client()
                .post(
                    "/v1/decision",
                    json={
                        "model_id": V1_ID,
                        "items": [choice_item("a-1"), choice_item("a-2", text=NEGATIVE_TEXT)],
                    },
                )
                .json()
            )
            second = (
                client()
                .post("/v1/decision", json={"model_id": V1_ID, "items": [choice_item("b-1")]})
                .json()
            )
        finally:
            patcher.stop()
        assert [r["error"] for r in first["results"]] == [None, None]
        assert second["results"][0]["error"] is None
        # the lazy singleton cached the stub after the first predict
        assert loader.call_count == 1

    def test_v2_model_id_loads_its_own_checkpoint(self):
        loader, patcher = install_stub_model()
        try:
            v1 = (
                client()
                .post("/v1/decision", json={"model_id": V1_ID, "items": [choice_item("v1")]})
                .json()
            )
            v2 = (
                client()
                .post("/v1/decision", json={"model_id": V2_ID, "items": [choice_item("v2")]})
                .json()
            )
        finally:
            patcher.stop()
        assert v1["results"][0]["error"] is None
        assert v2["results"][0]["error"] is None
        # one load per checkpoint, keyed apart
        assert loader.call_count == 2

    def test_described_options_map_by_position(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "d-1",
                            "input": "plain string state",
                            "question": {
                                "type": "choice",
                                "text": "What is the overall sentiment of this text?",
                                "choices": {
                                    "positive": "clearly positive",
                                    "negative": "clearly negative",
                                },
                            },
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        result = response.json()["results"][0]
        assert result["prediction"] in ("positive", "negative")
        assert result["error"] is None

    def test_task_word_falls_back_to_the_bench_phrasing(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "t-1",
                            "input": {"text": "The striker scored twice."},
                            "question": {"type": "choice", "options": SENTIMENT_OPTIONS},
                            "params": {"task": "topic"},
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        # no question text on the item; params.task -> engine.QUESTION["topic"]
        assert response.json()["results"][0]["error"] is None


class TestMultiTrack:
    def test_multi_question_answers_one_pick_per_question(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "m-1",
                            "input": {
                                "text": "I was charged twice for my subscription this month."
                            },
                            "question": {
                                "type": "multi",
                                "questions": [
                                    {
                                        "text": "Which team should handle this ticket?",
                                        "options": ["billing", "technical", "account"],
                                    },
                                    {
                                        "text": "Is the customer asking for a refund?",
                                        "options": ["no", "yes"],
                                    },
                                    {
                                        "text": "How urgent is this issue?",
                                        "options": ["could wait", "soon", "immediately"],
                                    },
                                ],
                            },
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        result = response.json()["results"][0]
        assert result["error"] is None
        assert isinstance(result["prediction"], list)
        assert len(result["prediction"]) == 3
        assert result["prob"] is None

    def test_multi_without_questions_names_the_field(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [{"item_id": "m-2", "input": "t", "question": {"type": "multi"}}],
                },
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "question.questions" in error

    def test_multi_entry_without_options_names_the_entry(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "m-3",
                            "input": "t",
                            "question": {
                                "type": "multi",
                                "questions": [{"text": "Which one?"}],
                            },
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "options" in error


class TestErrorMapping:
    def test_unknown_model_id_maps_to_an_error_row(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={"model_id": "pngwn/nanodiff-350m-base", "items": [choice_item()]},
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "unknown model_id" in error

    def test_unsupported_question_type_names_the_type(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [{"item_id": "x", "input": "t", "question": {"type": "ner"}}],
                },
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "ner" in error

    def test_bare_choice_question_sends_the_generic_default(self):
        # the suites' decision items carry options only; the engine still
        # needs a question string, so the glue supplies the generic one
        # (a known params.task upgrades it to the bench phrasing — the
        # task-word test below). Mock engine.predict to see the call args.
        from unittest.mock import Mock

        predict_mock = Mock(
            return_value=("positive", {"positive": 1.0, "negative": 0.0, "neutral": 0.0})
        )
        model_patcher = patch.object(engine, "load_model", return_value=(tiny_nanodiff(), 0))
        predict_patcher = patch.object(engine, "predict", predict_mock)
        model_patcher.start()
        predict_patcher.start()
        try:
            response = client().post(
                "/v1/decision",
                json={"model_id": V1_ID, "items": [choice_item("bare-1")]},
            )
        finally:
            model_patcher.stop()
            predict_patcher.stop()
        result = response.json()["results"][0]
        assert result["error"] is None and result["prediction"] == "positive"
        call = predict_mock.call_args
        assert call.args[2] == "Choose the best option."  # the question string
        assert call.args[3] == SENTIMENT_OPTIONS  # the ordered options

    def test_choice_without_options_names_the_field(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "x",
                            "input": "t",
                            "question": {"type": "choice", "text": "Which?"},
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "question.options" in error

    def test_more_than_ten_options_rejects_single_token_letters(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "x",
                            "input": "t",
                            "question": {
                                "type": "choice",
                                "text": "Which?",
                                "options": [f"o{i}" for i in range(11)],
                            },
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        error = response.json()["results"][0]["error"]
        assert error is not None and "A-J" in error

    def test_one_bad_item_does_not_kill_the_batch(self):
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        choice_item("ok-1"),
                        choice_item("bad-1", text="   "),
                        choice_item("ok-2"),
                    ],
                },
            )
        finally:
            patcher.stop()
        results = response.json()["results"]
        assert [r["item_id"] for r in results] == ["ok-1", "bad-1", "ok-2"]
        assert results[0]["error"] is None
        assert results[2]["error"] is None
        assert results[1]["prediction"] is None
        assert results[1]["error"] == (
            "ValueError: payload needs a non-empty 'input' (or 'input.text')"
        )


class TestPipeParity:
    def test_non_ascii_option_labels_round_trip_intact(self):
        # the legacy demo escaped JSON to survive Windows code pages on its
        # stdin/stdout pipe; over HTTP the JSON is UTF-8 end to end and the
        # label must come back byte-identical.
        label = "positif"
        accented = "négatif"
        _, patcher = install_stub_model()
        try:
            response = client().post(
                "/v1/decision",
                json={
                    "model_id": V1_ID,
                    "items": [
                        {
                            "item_id": "fr-1",
                            "input": {"text": "J'adore ce clavier."},
                            "question": {
                                "type": "choice",
                                "text": "Quel est le sentiment général de ce texte ?",
                                "options": [label, accented],
                            },
                        }
                    ],
                },
            )
        finally:
            patcher.stop()
        result = response.json()["results"][0]
        assert result["error"] is None
        assert result["prediction"] in (label, accented)
