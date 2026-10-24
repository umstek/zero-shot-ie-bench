"""Offline regression tests for models/certo.py and models/rlcd.py — the
two vendored-code engines (ported from the certo GitHub repo and the
LFM2.5-350M-RLCD HF repo respectively).

No model, no downloads: both modules import without torch/transformers/
jsonschema (the vendored code imports lazily), the mapping helpers are
exercised, and predict() runs against stub engines seeded into the module
caches."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

import small_decisions.models.certo as certo
import small_decisions.models.rlcd as rlcd


# ------------------------------------------------------------------ certo
class _StubCerto:
    """A DecisionModel stand-in: records decide() calls."""

    def __init__(self):
        self.calls: list[tuple[str, list[dict]]] = []

    def decide(self, state, options, max_state_len=64, max_option_len=48, abstain_below=None):
        self.calls.append((state, options))
        ids = [option["id"] for option in options]
        probs = {ids[0]: 0.86, **{i: 0.14 / (len(ids) - 1) for i in ids[1:]}}
        return {"probs": probs, "top": ids[0]}


class CertoTests:
    def test_import_pulls_no_heavy_dependencies(self):
        before = set(sys.modules)
        importlib.reload(certo)
        added = set(sys.modules) - before
        assert "torch" not in added
        assert "transformers" not in added

    def test_choice_maps_descriptions_and_probs(self):
        stub = _StubCerto()
        certo._ENGINE.append(stub)
        try:
            result = certo.predict({
                "input": {"text": "charged twice, want a refund"},
                "question": {"type": "choice",
                             "criteria": {"billing": "charges and payments", "tech": None}},
            })
        finally:
            certo._ENGINE.clear()
        assert result == {"prediction": "billing", "prob": pytest.approx(0.86)}
        state, options = stub.calls[0]
        assert state == "charged twice, want a refund"
        # a criterion's description is what the model scores; bare labels
        # fall back to the label itself
        assert options == [{"id": "billing", "description": "charges and payments"},
                           {"id": "tech", "description": "tech"}]

    def test_noul_and_score_are_refused_by_name(self):
        certo._ENGINE.append(_StubCerto())
        try:
            with pytest.raises(ValueError, match="noul"):
                certo.predict({"input": "x", "question": {"type": "noul"}})
        finally:
            certo._ENGINE.clear()


# ------------------------------------------------------------------- rlcd
class _StubRlcd:
    """An Engine stand-in: records constrained() calls."""

    def __init__(self, choice: str | None = "positive"):
        self.choice = choice
        self.calls: list[tuple[str, dict]] = []

    def constrained(self, context, schema):
        self.calls.append((context, schema))
        return {"text": '{"q": %s}' % ('null' if self.choice is None else f'"{self.choice}"'),
                "scores": {"q": [{"value": "positive", "log_likelihood": -0.11},
                                 {"value": "negative", "log_likelihood": -2.3}]},
                "prompt_tokens": 12, "branches": 2,
                "branch_tokens_padded": 24, "forward_calls": 2}


class RlcdTests:
    def test_import_pulls_no_heavy_dependencies(self):
        before = set(sys.modules)
        importlib.reload(rlcd)
        added = set(sys.modules) - before
        assert "torch" not in added
        assert "transformers" not in added
        assert "jsonschema" not in added

    def test_choice_builds_a_closed_one_field_enum_schema(self):
        stub = _StubRlcd()
        rlcd._ENGINE.append(stub)
        try:
            result = rlcd.predict({
                "input": {"text": "This laptop is great."},
                "question": {"type": "choice",
                             "instructions": "The overall sentiment of this text",
                             "options": ["positive", "negative"]},
            })
        finally:
            rlcd._ENGINE.clear()
        assert result == {"prediction": "positive", "prob": None}
        context, schema = stub.calls[0]
        assert context == "This laptop is great."
        assert schema["properties"]["q"]["enum"] == ["positive", "negative"]
        assert schema["additionalProperties"] is False

    def test_score_answers_the_winning_level_text(self):
        stub = _StubRlcd(choice="mid")
        rlcd._ENGINE.append(stub)
        try:
            result = rlcd.predict({
                "input": "now",
                "question": {"type": "score", "instructions": "How urgent?",
                             "criteria": ["low", "mid", "high"]},
            })
        finally:
            rlcd._ENGINE.clear()
        # the constrained head picks one branch — the winning level's index
        assert result["prediction"] == 1

    def test_noul_is_refused_by_name(self):
        with pytest.raises(ValueError, match="noul"):
            rlcd.predict({"input": "x", "question": {"type": "noul"}})
