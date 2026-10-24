"""Predict dispatch tests over stubbed pipelines — no weights, no network.

The stub pipelines are seeded straight into the engine's cache
(relik_module._PIPELINES, keyed by HF repo), which get_pipeline treats as
populated; the relik output stand-ins carry the Span/triplet fields the
row mappings read (legacy demos/relik_demo.py shapes).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import relik_runner.relik as relik_module
from relik_runner.relik import (
    CATALOG_MODEL_ID,
    MODEL_ID,
    NER_MODEL_ID,
    predict,
    resolve_track,
)


def span(start: int, end: int, label: str, text: str) -> SimpleNamespace:
    """A relik Span stand-in (NamedTuple fields: start, end, label, text)."""
    return SimpleNamespace(start=start, end=end, label=label, text=text)


def triplet(subject, label: str, object_, confidence: float) -> SimpleNamespace:
    """A relik triplet stand-in (subject, label, object, confidence)."""
    return SimpleNamespace(subject=subject, label=label, object=object_,
                           confidence=confidence)


def stub_pipeline() -> Mock:
    """A Relik stand-in: (text, num_workers=0) in, one RelikOutput out (a
    bare object for a single string — relik only returns a list for a
    list)."""
    out = SimpleNamespace(
        spans=[span(0, 5, "--NME--", "Alice"), span(16, 20, "--NME--", "Acme")],
        triplets=[triplet(span(0, 5, "--NME--", "Alice"), "P108",
                          span(16, 20, "--NME--", "Acme"), 0.93)],
    )
    pipeline = Mock()
    pipeline.return_value = out
    return pipeline


def seed(model_id: str, pipeline) -> None:
    """Pre-populate the cache for a wire model (get_pipeline then never
    touches load_relik)."""
    repo, _ = relik_module.WIRE_MODELS[model_id]
    relik_module._PIPELINES[repo] = pipeline


class TestResolveTrack:
    def test_the_benched_deployment_defaults_to_relation(self):
        assert resolve_track(CATALOG_MODEL_ID, {}, {}) == "relation"

    def test_the_ner_sibling_defaults_to_ner(self):
        assert resolve_track(NER_MODEL_ID, {}, {}) == "ner"

    def test_relation_keys_ask_for_relation_on_either_model(self):
        for model_id in (CATALOG_MODEL_ID, NER_MODEL_ID):
            for key in ("relations", "relation_labels", "triplets"):
                assert resolve_track(model_id, {key: []}, {}) == "relation"

    def test_span_keys_ask_for_ner_on_either_model(self):
        for model_id in (CATALOG_MODEL_ID, NER_MODEL_ID):
            for key in ("entities", "entity_labels", "labels", "spans"):
                assert resolve_track(model_id, {key: []}, {}) == "ner"

    def test_an_explicit_track_key_wins_over_everything(self):
        assert resolve_track(CATALOG_MODEL_ID, {"track": "ner", "relations": []},
                             {}) == "ner"
        assert resolve_track(NER_MODEL_ID, {}, {"track": "relation"}) == "relation"


class TestPredictRelation:
    def test_triplets_come_back_as_legacy_rows(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        out = predict(CATALOG_MODEL_ID, {"input": "Alice works for Acme.",
                                         "question": {}})
        assert out["error"] is None and out["prob"] is None
        assert out["prediction"] == [{
            "subject": {"start": 0, "end": 5, "label": "--NME--", "text": "Alice"},
            "relation": "P108",
            "object": {"start": 16, "end": 20, "label": "--NME--", "text": "Acme"},
            "confidence": 0.93,
        }]
        pipeline = relik_module._PIPELINES[MODEL_ID]
        pipeline.assert_called_once_with("Alice works for Acme.", num_workers=0)

    def test_params_top_k_is_set_on_the_pipeline(self):
        pipeline = stub_pipeline()
        seed(CATALOG_MODEL_ID, pipeline)
        predict(CATALOG_MODEL_ID, {"input": "text", "question": {},
                                   "params": {"top_k": 5}})
        assert pipeline.top_k == 5

    def test_the_ner_sibling_serves_relations_on_an_explicit_track(self):
        seed(NER_MODEL_ID, stub_pipeline())
        out = predict(NER_MODEL_ID, {"input": "text", "question": {},
                                     "params": {"track": "relation"}})
        assert out["error"] is None
        assert out["prediction"][0]["relation"] == "P108"


class TestPredictNer:
    def test_spans_come_back_as_legacy_rows(self):
        seed(NER_MODEL_ID, stub_pipeline())
        out = predict(NER_MODEL_ID, {"input": "Alice works for Acme.",
                                     "question": {}})
        assert out["error"] is None and out["prob"] is None
        assert out["prediction"] == [
            {"start": 0, "end": 5, "label": "--NME--", "text": "Alice"},
            {"start": 16, "end": 20, "label": "--NME--", "text": "Acme"},
        ]

    def test_labels_in_the_question_are_not_shown_to_the_closed_vocabulary(self):
        # relik cannot take free-text labels at inference (its README's
        # whole point); whatever rides along must not reach the pipeline
        pipeline = stub_pipeline()
        seed(NER_MODEL_ID, pipeline)
        out = predict(NER_MODEL_ID, {"input": "text",
                                     "question": {"labels": ["person", "company"]}})
        assert out["error"] is None
        pipeline.assert_called_once_with("text", num_workers=0)


class TestPredictErrors:
    def test_unknown_model_id_maps_to_an_error_row(self):
        out = predict("relik-ie/nope", {"input": "t", "question": {}})
        assert out["prediction"] is None
        assert "unknown model_id" in out["error"]

    def test_empty_or_non_string_input_names_the_field(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        for bad in ("", "   ", None, {"text": "dict input"}):
            out = predict(CATALOG_MODEL_ID, {"input": bad, "question": {}})
            assert out["prediction"] is None
            assert "input" in out["error"]
        assert relik_module._PIPELINES[MODEL_ID].called is False

    def test_engine_failures_become_error_rows_not_exceptions(self):
        pipeline = Mock()
        pipeline.side_effect = RuntimeError("faiss index exploded")
        seed(CATALOG_MODEL_ID, pipeline)
        out = predict(CATALOG_MODEL_ID, {"input": "t", "question": {}})
        assert out == {"prediction": None, "prob": None,
                       "error": "RuntimeError: faiss index exploded"}

    def test_non_object_question_names_the_field(self):
        seed(CATALOG_MODEL_ID, stub_pipeline())
        out = predict(CATALOG_MODEL_ID, {"input": "t", "question": "ner"})
        assert "question" in out["error"]


def test_single_output_object_is_normalized():
    # relik returns a bare RelikOutput for a single string and a list for
    # a list; the engine normalizes to the single output (legacy demo did
    # the isinstance dance)
    pipeline = Mock()
    pipeline.return_value = [  # the list shape, e.g. from an internal batch
        SimpleNamespace(spans=[span(0, 1, "--NME--", "A")], triplets=[]),
    ]
    seed(CATALOG_MODEL_ID, pipeline)
    out = predict(CATALOG_MODEL_ID, {"input": "A", "question": {}})
    assert out["error"] is None
    assert out["prediction"] == []
