"""Offline regression tests for models/d1.py (port of the client-side
substance of legacy/tests/test_d1_wiring.py).

The legacy test checked the bench registries + recorded results rows; its
replacement here is the module contract (the registry is the catalog now —
/info is policed in test_server.py, and the catalog's runner assignment is
asserted so a reassignment flips this module into the served set). No
model, no downloads: the module imports without torch/transformers, the
pure helpers are exercised, and a missing snapshot fails fast naming the
pull command."""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import patch

import pytest

import small_decisions.models.d1 as d1


class CatalogAssignment:
    """The reason d1 is ported but not registered: the catalog keeps the
    local d1 deployments on the von island (transformers >=5.14 there, but
    this island's 5.17 satisfies it too — the routing is catalog data)."""

    def test_the_ported_catalog_ids_match_the_d1_yamls(self):
        assert {entry["model_id"] for entry in d1.CATALOG_ENTRIES} == {
            "liquid/d1-3b", "liquid/d1-omni-600m"}
        assert d1.MODEL_IDS["liquid/d1-3b"] == "d1-3b"


class ModelTests:
    def test_normalize_takes_shorthand_repo_or_none(self):
        assert d1.normalize_model(None) == "d1-3b"
        assert d1.normalize_model("D1-3B") == "d1-3b"
        assert d1.normalize_model("LiquidAI/d1-omni-600M") == "d1-omni-600m"
        with pytest.raises(ValueError, match="d1-ultra"):
            d1.normalize_model("d1-ultra")

    def test_default_homes_sit_under_the_c_drive_base(self):
        assert d1.model_dir("d1-3b") == os.path.join(r"C:\src", "d1-3B")
        with patch.dict(os.environ, {"D1_HOME": "D:/models"}):
            assert d1.model_dir("d1-omni-600m") == os.path.join("D:/models", "d1-omni-600M")

    def test_missing_snapshot_names_the_download_command(self):
        with patch.dict(os.environ, {"D1_HOME": os.path.dirname(__file__)}):
            with pytest.raises(FileNotFoundError, match="hf download LiquidAI/d1-3B"):
                d1._require_snapshot("d1-3b")


class _StubEngine:
    """Records predict() requests, answers every question alike."""

    def __init__(self):
        self.answer = {"type": "choice", "choice": "billing", "confidence": 0.9,
                       "probabilities": {"billing": 0.9, "tech": 0.1}}
        self.calls: list[tuple[str, dict]] = []

    def predict(self, state, questions, images=None, audio=None):
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answer) for name in questions},
                "model": "d1-3B", "input_tokens": 42}


class PredictTests:
    def test_text_is_the_state_and_the_answer_row_maps(self):
        stub = _StubEngine()
        d1._ENGINE["d1-3b"] = stub
        try:
            result = d1.predict("liquid/d1-3b", {
                "input": {"text": "Charged twice, want a refund."},
                "question": {"type": "choice", "options": ["billing", "tech"]},
            })
        finally:
            d1._ENGINE.clear()
        assert result == {"prediction": "billing", "prob": 0.9}
        state, questions = stub.calls[0]
        assert state == "Charged twice, want a refund."
        assert questions["q"]["criteria"] == {"billing": None, "tech": None}


def test_import_pulls_no_heavy_dependencies():
    # the delta matters: another test (or venv) may already hold torch
    before = set(sys.modules)
    importlib.reload(d1)
    added = set(sys.modules) - before
    assert "torch" not in added
    assert "transformers" not in added
