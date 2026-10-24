"""Per-family smoke tests against real checkpoints — HEAVY, weight-gated.

These load actual weights (several GB, fp32 CPU) and are skipped unless the
checkpoint is already in the local HF hub cache: CI never downloads weights
(docs/plan/02 — "CI never downloads model weights"). On a machine that has
the weights (e.g. seoul after a bench run), they double as load-path
regression checks. Assertions pin shapes and callability, not score values —
prediction parity vs legacy is the later integration step.

The laya.router mode is not smoke-tested here: the Router pulls its mmBERT
routing checkpoint on top of the two Laya checkpoints.
"""

from __future__ import annotations

import pytest
from hf_weights import weights_cached  # pyrefly: ignore[missing-import]

from gliner_runner.models import gliformer, gliclass, glirel, gliner_relex, laya

GLIFORMER = "knowledgator/gliformer-base-v1"
GLIREL = "jackboyla/glirel-large-v0"
RELEX = "knowledgator/gliner-relex-multi-v1.0"
LICLASS = "knowledgator/gliclass-modern-base-v3.0"  # the smallest gliclass
LAYA = "convaiinnovations/laya"

needs = pytest.mark.skipif


@needs(not weights_cached(GLIFORMER), reason=f"{GLIFORMER} not in the local HF cache")
class TestGliformerWeights:
    def test_classify_and_ner_answer_the_bench_shapes(self):
        model = gliformer.load(GLIFORMER)
        label, prob = gliformer.run_classify(
            model, "This laptop has amazing performance but terrible battery life!",
            {"labels": {"positive": "d", "negative": "d", "neutral": "d"}}, {})
        assert label in ("positive", "negative", "neutral") or label is None
        assert prob is None or 0.0 <= prob <= 1.0
        spans, _ = gliformer.run_ner(
            model, "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
            {"labels": ["company", "person", "product", "location"]}, {})
        assert all(isinstance(s, tuple) and len(s) == 3 for s in spans)


@needs(not weights_cached(RELEX), reason=f"{RELEX} not in the local HF cache")
class TestRelexWeights:
    def test_ner_and_relations_in_one_pass(self):
        model = gliner_relex.load(RELEX)
        spans, _ = gliner_relex.run_ner(
            model, "Alice works for Acme in Paris.",
            {"labels": ["person", "company", "location"]}, {})
        assert isinstance(spans, list)
        rows, _ = gliner_relex.run_relation(
            model, "Alice works for Acme in Paris.",
            {"entity_labels": ["person", "company", "location"],
             "relations": [{"name": "works for", "head": "person",
                            "tail": "company"}]}, {})
        assert isinstance(rows, list)
        assert all(row["head_type"] == "person" and row["tail_type"] == "company"
                   for row in rows)


@needs(not weights_cached(GLIREL), reason=f"{GLIREL} not in the local HF cache")
class TestGlirelWeights:
    def test_relations_over_handed_entities(self):
        model = glirel.load(GLIREL)
        text = "Alice works for Acme Ltd in Paris."
        rows, _ = glirel.run_relation(
            model, text,
            {"relation_labels": ["works_for", "located_in"],
             "entities": [{"start": 0, "end": 5, "label": "person"},
                          {"start": 16, "end": 25, "label": "company"}]}, {})
        assert all(set(row) == {"relation", "head", "head_type", "tail",
                                "tail_type", "score"} for row in rows)


@needs(not weights_cached(LICLASS), reason=f"{LICLASS} not in the local HF cache")
class TestGliclassWeights:
    def test_argmax_classification(self):
        pipe = gliclass.load(LICLASS)
        label, prob = gliclass.run_classify(
            pipe, "The food was cold and the waiter was rude.",
            {"labels": {"positive": "d", "negative": "d", "neutral": "d"}}, {})
        assert label in ("positive", "negative", "neutral", None)
        assert prob is None or 0.0 <= prob <= 1.0


@needs(not weights_cached(LAYA), reason=f"{LAYA} not in the local HF cache")
class TestLayaWeights:
    def test_choice_question_answers_a_label(self):
        agent = laya.load(LAYA)
        prediction, prob = laya.run_decision(
            agent, "The food was cold and the waiter was rude.",
            {"type": "choice",
             "instructions": 'What is the overall sentiment of this text: "..."',
             "criteria": {"positive": None, "negative": None, "neutral": None}}, {})
        assert prediction in ("positive", "negative", "neutral", None)
        assert prob is None or 0.0 <= prob <= 1.0
