"""Offline regression tests for models/decision2.py and models/d3.py
(ports of legacy/tests/test_decision2_demo.py's client-side classes; the
bench-registry checks became the /info test — the registry is the catalog
now).

No model, no downloads: the modules must import without decision2/d3/
torch (heavy imports are lazy), the pure size/home helpers are exercised,
a missing snapshot fails fast naming the pull command, and the predict()
mapping runs against stub engines seeded into the module caches."""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import patch

import pytest

import small_decisions.models.decision2 as decision2
import small_decisions.models.d3 as d3
from conftest import CHOICE, NOUL, SCORE


class _StubEngine:
    """Records system_one() requests, answers every question alike."""

    def __init__(self, answer: dict[str, Any]):
        self.answer = answer
        self.calls: list[Any] = []

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answer) for name in questions}, "model": "stub",
                "usage": {"input_tokens": 42}}


# ------------------------------------------------------------ Decision 2.0
class TestDecision2Sizes:
    def test_sizes_cover_the_three_benched_checkpoints(self):
        assert decision2.SIZES == {
            "kai-0.6b": "Decision-2.0-Kai-0.6B",
            "eos-0.8b": "Decision-2.0-Eos-0.8B",
            "sol-2b": "Decision-2.0-Sol-2B",
        }
        assert decision2.DEFAULT_SIZE == "kai-0.6b"

    def test_normalize_takes_shorthand_full_repo_or_none(self):
        assert decision2.normalize_size(None) == "kai-0.6b"
        assert decision2.normalize_size("Kai-0.6B") == "kai-0.6b"
        assert decision2.normalize_size("vllm-sr/Decision-2.0-Sol-2B") == "sol-2b"
        with pytest.raises(ValueError, match="nox-4b"):
            decision2.normalize_size("nox-4b")

    def test_catalog_ids_map_to_their_sizes(self):
        assert decision2.MODEL_IDS == {
            "vllm-sr/decision-2.0-kai": "kai-0.6b",
            "vllm-sr/decision-2.0-eos": "eos-0.8b",
            "vllm-sr/decision-2.0-sol": "sol-2b",
        }


class TestDecision2Homes:
    def test_default_homes_sit_under_the_c_drive_base(self):
        # D: is full; the snapshots follow the K2-Type / Intern-Decision
        # precedent at C:\src (see the client docstring)
        assert decision2.model_dir("kai-0.6b") == os.path.join(r"C:\src", "Decision-2.0-Kai-0.6B")
        assert decision2.model_dir("sol-2b").endswith("Decision-2.0-Sol-2B")

    def test_decision2_home_env_overrides_the_prefix(self):
        with patch.dict(os.environ, {"DECISION2_HOME": "D:/models"}):
            assert decision2.model_dir("eos-0.8b") == os.path.join("D:/models", "Decision-2.0-Eos-0.8B")

    def test_missing_runtime_names_the_download_command(self):
        with patch.dict(os.environ, {"DECISION2_HOME": os.path.dirname(__file__)}):
            with pytest.raises(FileNotFoundError, match="hf download vllm-sr/Decision-2.0-Kai-0.6B"):
                decision2._ensure_runtime_on_path("kai-0.6b")


class TestDecision2Predict:
    def test_text_is_the_state_and_the_answer_row_maps(self):
        stub = _StubEngine({
            "type": "choice", "choice": "positive", "confidence": 0.8,
            "probabilities": {"positive": 0.9, "negative": 0.1},
        })
        decision2._ENGINE["kai-0.6b"] = stub
        result = decision2.predict("vllm-sr/decision-2.0-kai", CHOICE)
        assert result == {"prediction": "positive", "prob": 0.9}
        state, questions = stub.calls[0]
        assert state == CHOICE["input"]["text"]
        question = questions["q"]
        assert question["type"] == "choice"
        assert question["criteria"] == {"positive": None, "negative": None}

    def test_in_place_error_rows_map_to_the_error_shape(self):
        decision2._ENGINE["eos-0.8b"] = _StubEngine({"type": "choice", "error": "invalid_question"})
        result = decision2.predict("vllm-sr/decision-2.0-eos", CHOICE)
        assert result["prediction"] is None
        assert "invalid_question" in result["error"]

    def test_noul_and_score_rows_map(self):
        decision2._ENGINE["sol-2b"] = _StubEngine({"type": "noul", "noul": 0.87})
        result = decision2.predict("vllm-sr/decision-2.0-sol", NOUL)
        assert result == {"prediction": "true", "prob": 0.87}

        decision2._ENGINE["sol-2b"] = _StubEngine({
            "type": "score", "score": 1.1,
            "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3},
        })
        result = decision2.predict("vllm-sr/decision-2.0-sol", SCORE)
        assert result == {"prediction": 1.1, "prob": 0.5}

    def test_unknown_model_id_raises(self):
        with pytest.raises(KeyError, match="no-such"):
            decision2.predict("no-such", CHOICE)


# ------------------------------------------------------------ Decision 3.0
class TestD3SizesAndHomes:
    def test_sizes_cover_the_three_benched_checkpoints(self):
        assert d3.SIZES == {"edge": "d3-edge", "lite": "d3-lite", "nano": "d3-nano"}
        assert d3.DEFAULT_SIZE == "edge"
        assert d3.MODEL_IDS == {
            "vllm-sr/d3-edge": "edge",
            "vllm-sr/d3-lite": "lite",
            "vllm-sr/d3-nano": "nano",
        }

    def test_normalize_takes_shorthand_repo_or_none(self):
        assert d3.normalize_size(None) == "edge"
        assert d3.normalize_size("D3-Nano") == "nano"
        assert d3.normalize_size("vllm-sr/d3-lite") == "lite"
        with pytest.raises(ValueError, match="mini-4b"):
            d3.normalize_size("mini-4b")

    def test_default_homes_sit_under_the_c_drive_base(self):
        assert d3.model_dir("edge") == os.path.join(r"C:\src", "d3-edge")
        with patch.dict(os.environ, {"D3_HOME": "D:/models"}):
            assert d3.model_dir("nano") == os.path.join("D:/models", "d3-nano")

    def test_missing_runtime_names_the_download_command(self):
        with patch.dict(os.environ, {"D3_HOME": os.path.dirname(__file__)}):
            with pytest.raises(FileNotFoundError, match="hf download vllm-sr/d3-edge"):
                d3._ensure_runtime_on_path("edge")


class TestD3Predict:
    def test_choice_mapping_and_state_pass_through(self):
        stub = _StubEngine({
            "type": "choice", "choice": "negative", "confidence": 0.6,
            "probabilities": {"positive": 0.4, "negative": 0.6},
        })
        d3._ENGINE["edge"] = stub
        result = d3.predict("vllm-sr/d3-edge", CHOICE)
        assert result == {"prediction": "negative", "prob": 0.6}
        state, questions = stub.calls[0]
        assert state == CHOICE["input"]["text"]
        assert questions["q"]["instructions"]

    def test_error_rows_map_to_the_error_shape(self):
        d3._ENGINE["lite"] = _StubEngine({"type": "score", "error": "max_length_exceeded"})
        result = d3.predict("vllm-sr/d3-lite", SCORE)
        assert result["prediction"] is None
        assert "max_length_exceeded" in result["error"]


# ----------------------------------------------------- shared import rules
@pytest.mark.parametrize("module", [decision2, d3])
def test_import_pulls_no_heavy_dependencies(module):
    # the delta matters: another test (or venv) may already hold torch
    before = set(sys.modules)
    import importlib

    importlib.reload(module)
    added = set(sys.modules) - before
    assert "torch" not in added
    assert "transformers" not in added
    if module is decision2:
        assert "decision2" not in added
    else:
        assert "d3_runtime" not in added
