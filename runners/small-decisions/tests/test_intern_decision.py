"""Offline regression tests for models/intern_decision.py (the loader-side
substance of legacy/tests/test_intern_decision_demo.py's client checks).

No model, no downloads: the module imports without torch/transformers, the
size/home helpers are exercised, and the per-home runtime caching + the
predict() mapping run against a stubbed snapshot (a tmp home whose
inference.py stands in for the checkpoint's standalone runtime — the
real per-home sys.path/cache mechanics, not a mock of them)."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import small_decisions.models.intern_decision as intern_decision

STUB_INFERENCE = '''
"""A stand-in for the snapshot's standalone inference.py runtime."""

DEFAULT_TEMPERATURE = 1.23


class DecisionEngine:
    def __init__(self, checkpoint=None, device="cpu", dtype="float32"):
        self.checkpoint = checkpoint
        self.device = device
        self.dtype = dtype

    def predict(self, request):
        answers = {
            name: {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.9, "tech": 0.1}}
            for name in request["questions"]
        }
        return {"answers": answers, "model": "stub", "input_tokens": 42}
'''


@pytest.fixture()
def intern_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A tmp INTERN_DECISION_HOME holding one stubbed snapshot."""
    (tmp_path / "Intern-Decision-0.8B").mkdir()
    (tmp_path / "Intern-Decision-0.8B" / "inference.py").write_text(STUB_INFERENCE, encoding="utf-8")
    monkeypatch.setenv("INTERN_DECISION_HOME", str(tmp_path))
    yield tmp_path


class SizeTests:
    def test_sizes_cover_the_three_benched_checkpoints(self):
        assert intern_decision.SIZES == {
            "0.8B": "internlm/Intern-Decision-0.8B",
            "2B": "internlm/Intern-Decision-2B",
            "4B": "internlm/Intern-Decision-4B",
        }
        assert intern_decision.DEFAULT_SIZE == "0.8B"

    def test_normalize_takes_shorthand_or_none(self):
        assert intern_decision.normalize_size(None) == "0.8B"
        assert intern_decision.normalize_size("0.8b") == "0.8B"
        assert intern_decision.normalize_size("internlm/Intern-Decision-2B") == "2B"
        with pytest.raises(ValueError, match="8B"):
            intern_decision.normalize_size("8B")

    def test_default_homes_sit_under_the_c_drive_base(self):
        assert intern_decision.size_home("0.8B") == os.path.abspath(
            os.path.join(r"C:\src", "Intern-Decision-0.8B"))
        with patch.dict(os.environ, {"INTERN_DECISION_HOME": "D:/models"}):
            assert intern_decision.size_home("2B") == os.path.abspath(
                os.path.join("D:/models", "Intern-Decision-2B"))

    def test_missing_snapshot_names_the_env_var(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        monkeypatch.setenv("INTERN_DECISION_HOME", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="INTERN_DECISION_HOME"):
            intern_decision.load_engine("0.8B")


class RuntimeCachingTests:
    def test_one_runtime_module_per_home(self, intern_home: Path):
        first = intern_decision._load_runtime(str(intern_home / "Intern-Decision-0.8B"))
        again = intern_decision._load_runtime(str(intern_home / "Intern-Decision-0.8B"))
        assert first is again

    def test_the_runtime_import_never_leaks_inference(self, intern_home: Path):
        intern_decision._load_runtime(str(intern_home / "Intern-Decision-0.8B"))
        assert "inference" not in sys.modules  # hygiene after the import


class PredictTests:
    def test_request_dict_and_answer_mapping(self, monkeypatch: pytest.MonkeyPatch, intern_home: Path):
        intern_decision._ENGINE_KEYED.clear()
        intern_decision._RUNTIMES.clear()
        try:
            result = intern_decision.predict("internlm/intern-decision-0.8b", {
                "input": {"text": "Charged twice."},
                "question": {"type": "choice", "options": ["billing", "tech"]},
            })
        finally:
            intern_decision._ENGINE_KEYED.clear()
            intern_decision._RUNTIMES.clear()
            home = str(intern_home)
            while home in sys.path:
                sys.path.remove(home)
        assert result == {"prediction": "billing", "prob": 0.9}

    def test_unknown_model_id_raises(self):
        with pytest.raises(KeyError, match="no-such"):
            intern_decision.predict("no-such", {"input": "x", "question": {"type": "choice"}})


def test_import_pulls_no_heavy_dependencies():
    # the delta matters: another test (or venv) may already hold torch
    before = set(sys.modules)
    importlib.reload(intern_decision)
    added = set(sys.modules) - before
    assert "torch" not in added
    assert "transformers" not in added
    assert "inference" not in added
