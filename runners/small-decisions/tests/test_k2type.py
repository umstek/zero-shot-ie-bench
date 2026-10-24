"""Offline regression tests for models/k2type.py (port of the client-side
classes of legacy/tests/test_k2type_demo.py; the bench-registry checks
became the /info test — the registry is the catalog now).

No model, no downloads: the module must import without jev/torch (heavy
imports are lazy, and to_record() only touches the snapshot when an
instructions/criteria payload carries a dict that needs jev.encode.render),
the pure request/response mapping functions are exercised, and the
predict() mapping runs against a stub engine seeded into the module
cache."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

import small_decisions.models.k2type as k2type
from conftest import CHOICE, NOUL, SCORE


class TestModelHome:
    def test_default_is_the_c_drive_snapshot(self):
        # D: is full; the snapshot follows the Intern-Decision precedent
        # at C:\src (see the client docstring)
        assert k2type.model_home().endswith("K2-Type-0.9B")
        assert "K2-Type-0.9B" in k2type.model_home()

    def test_k2type_home_env_overrides(self):
        with patch.dict(os.environ, {"K2TYPE_HOME": "D:/models/K2-Type"}):
            assert k2type.model_home() == "D:/models/K2-Type"

    def test_registry_carries_the_catalog_identity(self):
        assert k2type.MODEL_ID == "ifm/k2-type"
        assert k2type.VERSION == "k2-type-0.9b@unversioned"


class TestToRecord:
    def test_choice_takes_string_instructions_and_criteria(self):
        rec = k2type.to_record({
            "state": "a ticket",
            "questions": {"q": {
                "type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": None}}}})
        assert rec["state"] == "a ticket"
        assert rec["questions"]["q"] == {
            "type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": None},
            # label is a placeholder the forward never reads
            "label": "billing"}

    def test_score_and_noul_rows_map_with_placeholders(self):
        rec = k2type.to_record({
            "state": None,
            "questions": {
                "s": {"type": "score", "instructions": "How urgent?",
                      "criteria": ["low", "high"]},
                "n": {"type": "noul", "instructions": "Refund?"}}})
        assert rec["questions"]["s"]["criteria"] == ["low", "high"]
        assert rec["questions"]["s"]["label"] == 0
        assert not rec["questions"]["n"]["label"]
        assert "criteria" not in rec["questions"]["n"]

    def test_option_counts_are_validated_like_the_server(self):
        with pytest.raises(ValueError, match="1..255"):
            k2type.to_record({
                "state": "s",
                "questions": {"q": {"type": "choice", "instructions": "?", "criteria": {}}}})
        with pytest.raises(ValueError, match="2..255"):
            k2type.to_record({
                "state": "s",
                "questions": {"q": {"type": "score", "instructions": "?", "criteria": ["only"]}}})

    def test_unknown_type_names_the_question(self):
        with pytest.raises(ValueError, match="weird"):
            k2type.to_record({
                "state": "s",
                "questions": {"weird": {"type": "bool", "instructions": "?"}}})


class TestAnswer:
    def test_choice_picks_argmax_with_the_margin_confidence(self):
        row = k2type.answer(
            {"type": "choice", "criteria": {"a": None, "b": None, "c": None, "d": None}},
            [0.1, 0.7, 0.1, 0.1])
        assert row["choice"] == "b"
        # (pmax - 1/K) / (1 - 1/K) with K=4
        assert row["confidence"] == pytest.approx((0.7 - 0.25) / 0.75, abs=1e-4)
        assert row["probabilities"]["b"] == 0.7

    def test_single_option_choice_is_fully_confident(self):
        row = k2type.answer({"type": "choice", "criteria": {"x": 1}}, [1.0])
        assert row["confidence"] == 1.0

    def test_noul_reports_p_true(self):
        row = k2type.answer({"type": "noul", "label": False}, [0.25, 0.75])
        assert row["noul"] == 0.75

    def test_score_returns_the_expected_level_and_legend(self):
        row = k2type.answer(
            {"type": "score", "criteria": ["low", "mid", "high"], "label": 0},
            [0.2, 0.5, 0.3])
        assert row["score"] == pytest.approx(1.1, abs=1e-4)
        assert row["legend"] == {"0": "low", "1": "mid", "2": "high"}
        assert row["probabilities"] == {"0": 0.2, "1": 0.5, "2": 0.3}


class _StubEngine:
    """Records predict() requests, answers every question alike."""

    def __init__(self):
        self.answer: dict[str, Any] = {
            "type": "choice", "choice": "positive", "confidence": 0.8,
            "probabilities": {"positive": 0.9, "negative": 0.1}}
        self.calls: list[Any] = []

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answer) for name in questions}, "input_tokens": 42}


class TestPredict:
    def test_text_is_the_state_and_the_row_maps_choice_probabilities(self):
        stub = _StubEngine()
        k2type._ENGINE.clear()
        k2type._ENGINE.append(stub)
        try:
            result = k2type.predict(CHOICE)
        finally:
            k2type._ENGINE.clear()
        state, questions = stub.calls[0]
        assert state == CHOICE["input"]["text"]
        question = questions["q"]
        assert question["type"] == "choice"
        assert question["instructions"]
        # confidence is the model's own margin, not pmax — answer_to_result
        # reports the chosen label's probability either way
        assert result == {"prediction": "positive", "prob": 0.9}

    def test_noul_reads_the_p_true_side(self):
        stub = _StubEngine()
        stub.answer = {"type": "noul", "noul": 0.25}
        k2type._ENGINE.append(stub)
        try:
            result = k2type.predict(NOUL)
        finally:
            k2type._ENGINE.clear()
        assert result == {"prediction": "false", "prob": 0.75}

    def test_score_reads_the_expected_level(self):
        stub = _StubEngine()
        stub.answer = {"type": "score", "score": 1.1,
                       "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}
        k2type._ENGINE.append(stub)
        try:
            result = k2type.predict(SCORE)
        finally:
            k2type._ENGINE.clear()
        assert result == {"prediction": 1.1, "prob": 0.5}


def test_import_pulls_no_heavy_dependencies():
    # the delta matters: another test (or venv) may already hold torch
    before = set(sys.modules)
    importlib.reload(k2type)
    added = set(sys.modules) - before
    assert "jev" not in added
    assert "torch" not in added
    assert "transformers" not in added
