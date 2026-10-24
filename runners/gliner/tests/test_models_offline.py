"""Offline dispatch tests: every model module's call path on fakes.

No network, no weights: the loaders are stubbed through each module's
_LOADED cache (the same dicts the lazy loads populate), so the tests pin
the bench-ported call semantics — thresholds, label handling, answer
shapes and the per-item error contract — not the checkpoints.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from gliner_runner import models
from gliner_runner.models import gliformer, gliner_relex, glirel, gliclass, laya


class TestGliclassDispatch:
    def test_argmax_over_bare_label_keys(self, monkeypatch):
        seen = {}

        def pipe(text, labels, threshold=0.0):
            seen.update(text=text, labels=list(labels), threshold=threshold)
            return [[{"label": label, "score": score}
                     for label, score in zip(labels, [0.4, 0.9, 0.1])]]

        monkeypatch.setitem(gliclass._LOADED, "knowledgator/gliclass-base-v3.0", pipe)
        out = models.predict(
            "gliclass-base-v3.0.local",
            {"input": "great product",
             "question": {"labels": {"negative": "d", "positive": "d", "neutral": "d"}}})
        # bench semantics: bare keys (dict order), threshold 0.0, argmax
        assert seen == {"text": "great product",
                        "labels": ["negative", "positive", "neutral"],
                        "threshold": 0.0}
        assert out == {"prediction": "positive", "prob": 0.9, "error": None}

    def test_empty_output_answers_none(self, monkeypatch):
        monkeypatch.setitem(gliclass._LOADED, "knowledgator/gliclass-base-v3.0",
                            lambda text, labels, threshold=0.0: [[]])
        out = models.predict("gliclass-base-v3.0.local",
                             {"input": "x", "question": {"labels": ["a"]}})
        assert out["prediction"] is None and out["error"] is None

    def test_plain_list_labels_pass_through(self, monkeypatch):
        seen = {}

        def pipe(text, labels, threshold=0.0):
            seen["labels"] = list(labels)
            return [[{"label": "a", "score": 0.5}]]

        monkeypatch.setitem(gliclass._LOADED, "knowledgator/gliclass-base-v3.0", pipe)
        models.predict("gliclass-base-v3.0.local",
                       {"input": "x", "question": {"labels": ["a", "b"]}})
        assert seen["labels"] == ["a", "b"]


class TestGliformerDispatch:
    def test_classify_takes_the_top_class(self, monkeypatch):
        model = SimpleNamespace(
            classify=lambda text, labels, threshold: [
                {"class_name": "negative", "score": 0.8}])
        monkeypatch.setitem(gliformer._LOADED, "knowledgator/gliformer-base-v1", model)
        out = models.predict("gliformer-base-v1.local",
                             {"input": "terrible",
                              "question": {"labels": {"positive": "d", "negative": "d"}}})
        assert out["prediction"] == "negative"
        assert out["prob"] == 0.8

    def test_nothing_above_threshold_answers_none(self, monkeypatch):
        model = SimpleNamespace(classify=lambda text, labels, threshold: [])
        monkeypatch.setitem(gliformer._LOADED, "knowledgator/gliformer-base-v1", model)
        out = models.predict("gliformer-base-v1.local",
                             {"input": "x", "question": {"labels": ["a"]}})
        assert out["prediction"] is None

    def test_ner_sorts_and_dedupes_spans(self, monkeypatch):
        model = SimpleNamespace(predict_entities=lambda text, labels, threshold: [
            {"start": 0, "end": 5, "label": "person"},
            {"start": 0, "end": 5, "label": "person"},  # duplicate
            {"start": 7, "end": 12, "label": "company"},
        ])
        monkeypatch.setitem(gliformer._LOADED, "knowledgator/gliformer-base-v1", model)
        out = models.predict("gliformer-base-v1.local",
                             {"input": "Alice works at Acme",
                              "question": {"labels": ["person", "company"]}})
        assert out["prediction"] == [(0, 5, "person"), (7, 12, "company")]
        assert out["prob"] is None

    def test_non_string_input_is_a_per_item_error(self, monkeypatch):
        # the input check must fail before any weights are touched, so the
        # cache is stubbed and the loader is never reached
        monkeypatch.setitem(gliformer._LOADED, "knowledgator/gliformer-base-v1",
                            SimpleNamespace())
        out = models.predict("gliformer-base-v1.local",
                             {"input": ["not", "text"],
                              "question": {"labels": ["person"]}})
        assert out["prediction"] is None
        assert out["error"] is not None and "ValueError" in out["error"]


class TestRelexDispatch:
    def test_ner_reads_inference_entities(self, monkeypatch):
        def inference(**kwargs):
            assert kwargs["relations"] == []
            assert kwargs["return_relations"] is False
            # result[0] is the entities list (the app.py entities-tab shape)
            return [{"start": 3, "end": 8, "label": "person"}], None

        model = SimpleNamespace(inference=inference)
        monkeypatch.setitem(gliner_relex._LOADED,
                            "knowledgator/gliner-relex-multi-v1.0", model)
        out = models.predict("gliner-relex-multi-v1.0.local",
                             {"input": "Ms. Ada wrote it",
                              "question": {"labels": ["person"]}})
        assert out["prediction"] == [(3, 8, "person")]

    def test_relation_rows_are_sorted_and_constraint_filtered(self, monkeypatch):
        def inference(**kwargs):
            assert kwargs["relations"] == ["works for", "ceo of", "located in"]
            assert kwargs["flat_ner"] is False
            return (None, [[
                {"relation": "works for", "head": {"text": "Ada", "type": "person"},
                 "tail": {"text": "Acme", "type": "company"}, "score": 0.8},
                # wrong head type for the spec's constraint -> dropped
                {"relation": "ceo of", "head": {"text": "Acme", "type": "company"},
                 "tail": {"text": "Ada", "type": "person"}, "score": 0.9},
                # unconstrained relation -> kept
                {"relation": "located in", "head": {"text": "Acme", "type": "company"},
                 "tail": {"text": "Paris", "type": "location"}, "score": 0.7},
            ]])

        model = SimpleNamespace(inference=inference)
        monkeypatch.setitem(gliner_relex._LOADED,
                            "knowledgator/gliner-relex-multi-v1.0", model)
        out = models.predict(
            "gliner-relex-multi-v1.0.local",
            {"input": "Ada founded Acme in Paris",
             "question": {"entity_labels": ["person", "company", "location"],
                          "relations": [{"name": "works for", "head": "person",
                                         "tail": "company"},
                                        {"name": "ceo of", "head": "person"},
                                        "located in"]}})
        assert out["prediction"] == [
            {"relation": "works for", "head": "Ada", "head_type": "person",
             "tail": "Acme", "tail_type": "company", "score": 0.8},
            {"relation": "located in", "head": "Acme", "head_type": "company",
             "tail": "Paris", "tail_type": "location", "score": 0.7},
        ]

    def test_parse_relation_specs_variants(self):
        # plain strings, dicts, a single dict, and a JSON-encoded string
        parsed = gliner_relex.parse_relation_specs(
            [{"name": "a", "head": "person"}, "b"])
        assert parsed == (["a", "b"], {"a": ("person", None)})
        parsed = gliner_relex.parse_relation_specs({"name": "a"})
        assert parsed == (["a"], {})
        parsed = gliner_relex.parse_relation_specs('[{"name": "a", "tail": "co"}]')
        assert parsed == (["a"], {"a": (None, "co")})

    def test_parse_relation_specs_errors(self):
        for bad in ("{not json", [{}], {"name": ""}, [], 42):
            with pytest.raises(ValueError):
                gliner_relex.parse_relation_specs(bad)

    def test_census_classification_uses_the_vendor_multitask_mapping(self, monkeypatch):
        seen = {}

        def predict_entities(text, labels, threshold):
            seen.update(text=text, threshold=threshold)
            return [{"text": "positive", "score": 0.6}]

        model = SimpleNamespace(predict_entities=predict_entities)
        monkeypatch.setitem(gliner_relex._LOADED,
                            "knowledgator/gliner-relex-multi-v1.0", model)
        out = models.predict("gliner-relex-multi-v1.0.local",
                             {"input": "I love it",
                              "question": {"labels": {"positive": "d", "negative": "d"}}})
        # the gliner.multitask prompt shape, verbatim
        assert seen["text"] == ("Classify text into the following classes: "
                                "positive, negative \n I love it")
        assert seen["threshold"] == gliner_relex.CLASSIFICATION_THRESHOLD
        assert out["prediction"] == "positive"
        assert out["prob"] == 0.6


class TestGlirelDispatch:
    def test_tokenization_matches_the_gliner_regex(self):
        tokens, offsets = glirel.tokenize("Alice works for Acme Ltd.")
        assert tokens == ["Alice", "works", "for", "Acme", "Ltd", "."]
        assert offsets[3] == (16, 20)

    def test_char_spans_map_onto_inclusive_token_rows(self):
        tokens, ner = glirel.char_spans_to_ner(
            "Alice works for Acme Ltd.",
            [{"start": 0, "end": 5, "label": "person"},
             {"start": 16, "end": 24, "label": "company"}])
        assert tokens == ["Alice", "works", "for", "Acme", "Ltd", "."]
        # end 24 stops at "Ltd": the period token (24, 25) does not overlap
        # [16, 24), so the row stays [3, 4] inclusive
        assert ner == [[0, 0, "person", "Alice"], [3, 4, "company", "Acme Ltd"]]

    def test_relations_are_sorted_and_typed_through_the_plus_one_lookup(self, monkeypatch):
        calls = {}

        def predict_relations(tokens, labels, threshold, ner, top_k):
            calls.update(threshold=threshold, top_k=top_k, ner=ner)
            # head_pos/tail_pos are EXCLUSIVE-end token spans
            return [{"label": "works_for", "head_text": ["Alice"], "head_pos": [0, 1],
                     "tail_text": ["Acme", "Ltd"], "tail_pos": [3, 5], "score": 0.9},
                    {"label": "located_in", "head_text": ["Acme", "Ltd"],
                     "head_pos": [3, 5],
                     "tail_text": ["Paris"], "tail_pos": [6, 7], "score": 0.5}]

        model = SimpleNamespace(predict_relations=predict_relations)
        monkeypatch.setitem(glirel._LOADED, "jackboyla/glirel-large-v0", model)
        out = models.predict(
            "glirel-large-v0.local",
            {"input": "Alice works for Acme Ltd in Paris",
             "question": {"relation_labels": ["works_for", "located_in"],
                          "ner": [[0, 0, "person", "Alice"],
                                  [3, 4, "company", "Acme Ltd"],
                                  [6, 6, "location", "Paris"]]}})
        assert calls == {"threshold": 0.3, "top_k": 2,
                         "ner": [[0, 0, "person", "Alice"],
                                 [3, 4, "company", "Acme Ltd"],
                                 [6, 6, "location", "Paris"]]}
        # the (ent[0], ent[1] + 1) lookup types each row through the
        # exclusive-end positions GLiREL reports
        assert out["prediction"] == [
            {"relation": "works_for", "head": "Alice", "head_type": "person",
             "tail": "Acme Ltd", "tail_type": "company", "score": 0.9},
            {"relation": "located_in", "head": "Acme Ltd", "head_type": "company",
             "tail": "Paris", "tail_type": "location", "score": 0.5},
        ]

    def test_fewer_than_two_entities_answers_no_pairs(self, monkeypatch):
        model = SimpleNamespace(
            predict_relations=lambda *a, **k: pytest.fail("must not be called"))
        monkeypatch.setitem(glirel._LOADED, "jackboyla/glirel-large-v0", model)
        out = models.predict("glirel-large-v0.local",
                             {"input": "one entity only: Alice",
                              "question": {"relation_labels": ["works_for"],
                                           "entities": [{"start": 17, "end": 22,
                                                         "label": "person"}]}})
        assert out["prediction"] == [] and out["error"] is None

    def test_missing_relation_labels_is_a_per_item_error(self, monkeypatch):
        monkeypatch.setitem(glirel._LOADED, "jackboyla/glirel-large-v0",
                            SimpleNamespace())
        out = models.predict("glirel-large-v0.local",
                             {"input": "x", "question": {}})
        assert out["error"] is not None and "ValueError" in out["error"]


class TestLayaDispatch:
    def test_choice_answers_the_label_with_its_confidence(self, monkeypatch):
        states: list = []
        agent = SimpleNamespace(
            predict=lambda state, questions: (
                states.append(state) or
                {"answers": {"q": {"choice": "positive", "confidence": 0.77}}}))
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya", agent)
        out = models.predict("laya.local",
                             {"input": "The food was cold.",
                              "question": {"type": "choice",
                                           "instructions": "What is the overall "
                                                           "sentiment of this text?",
                                           "criteria": {"positive": None,
                                                        "negative": None}}})
        # a string input becomes the {"text": ...} state run_laya_typed used
        assert states == [{"text": "The food was cold."}]
        assert out == {"prediction": "positive", "prob": 0.77, "error": None}

    def test_score_and_noul_answer_their_numbers(self, monkeypatch):
        agent = SimpleNamespace(predict=lambda state, questions: {
            "answers": {"q": {"score": 1.5}}})
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya", agent)
        out = models.predict("laya.local",
                             {"input": {"excuses": ["my cat walked on the keyboard"]},
                              "question": {"type": "score", "instructions": "i",
                                           "criteria": ["a", "b"]}})
        assert out["prediction"] == 1.5 and out["prob"] is None

        agent = SimpleNamespace(predict=lambda state, questions: {
            "answers": {"q": {"noul": 0.12}}})
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya", agent)
        out = models.predict("laya.local",
                             {"input": "x", "question": {"type": "noul",
                                                         "instructions": "i"}})
        assert out["prediction"] == 0.12 and out["prob"] is None

    def test_dict_state_passes_through_untouched(self, monkeypatch):
        seen = {}
        agent = SimpleNamespace(predict=lambda state, questions: (
            seen.update(state=state) or {"answers": {"q": {"choice": "cat",
                                                           "confidence": 0.5}}}))
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya", agent)
        models.predict("laya.local",
                       {"input": {"excuses": ["my cat walked on the keyboard"]},
                        "question": {"type": "choice", "instructions": "i",
                                     "criteria": {}}})
        assert seen["state"] == {"excuses": ["my cat walked on the keyboard"]}

    def test_untyped_question_is_a_per_item_error(self, monkeypatch):
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya",
                            SimpleNamespace(predict=lambda *a, **k: {}))
        out = models.predict("laya.local",
                             {"input": "x", "question": {"instructions": "i"}})
        assert out["error"] is not None and "ValueError" in out["error"]

    def test_router_loader_builds_the_process_singleton(self, monkeypatch):
        built = []

        class FakeRouter:
            def __init__(self, device):
                built.append(device)

        fake_laya = SimpleNamespace(Router=FakeRouter)
        monkeypatch.setitem(sys.modules, "laya", fake_laya)
        monkeypatch.setattr(laya, "_ROUTER", [])
        first = laya.load_router("convaiinnovations/laya")
        second = laya.load_router("convaiinnovations/laya")
        assert built == ["cpu"]  # built once, cached afterwards
        assert first is second


class TestPredictContract:
    def test_unknown_model_is_a_per_item_error_not_an_exception(self):
        out = models.predict("not-a-model.local", {"input": "x", "question": {}})
        assert out["prediction"] is None
        assert "unknown model_id" in out["error"]

    def test_track_mismatch_is_a_per_item_error(self, monkeypatch):
        # laya has no NER implementation: a list-labels question falls back
        # to decision, but force the mismatch by asking through the module
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya",
                            SimpleNamespace(predict=lambda *a, **k: {}))
        out = models.predict("laya.local", {"input": "x", "question": {"labels": ["a"]}})
        # list labels fall back to laya's first capability (decision) — the
        # question type check then rejects it
        assert out["error"] is not None

    def test_load_failure_is_a_per_item_error(self, monkeypatch):
        def boom(repo):
            raise RuntimeError("weights not cached")

        monkeypatch.setattr(gliclass, "load", boom)
        out = models.predict("gliclass-base-v3.0.local",
                             {"input": "x", "question": {"labels": ["a"]}})
        assert "RuntimeError: weights not cached" in out["error"]

    def test_missing_question_falls_back_to_the_first_capability(self, monkeypatch):
        # no question at all: laya's first capability (decision) is reached,
        # which then rejects the untyped question as a per-item error
        monkeypatch.setitem(laya._LOADED, "convaiinnovations/laya",
                            SimpleNamespace(predict=lambda *a, **k: {}))
        out = models.predict("laya.local", {"input": "x"})
        assert out["prediction"] is None
        assert out["error"] is not None and "ValueError" in out["error"]
