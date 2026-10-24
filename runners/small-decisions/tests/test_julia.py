"""Offline regression tests for models/julia.py (port of the client-side
classes of legacy/tests/test_julia_demo.py; the bench-registry checks
became the /info test — the registry is the catalog now).

No model, no downloads: the module must import without julia/torch (heavy
imports are lazy), the home handling is exercised, and the transformers
5.17 compat guard + the predict() mapping run against a stubbed julia
package patched into sys.modules (the module's stale-module pop is
stubbed out so the seeds survive load_engine's import)."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

import small_decisions.models.julia as julia


class ModelHomeTests:
    def test_default_is_the_snapshot_next_to_the_repo(self):
        assert julia.model_home().endswith("Julia-1")

    def test_julia_home_env_overrides(self):
        with patch.dict(os.environ, {"JULIA_HOME": "D:/models/Julia-1"}):
            assert julia.model_home() == "D:/models/Julia-1"

    def test_registry_carries_the_hf_model_id(self):
        assert julia.MODEL_ID == "SupersonicLabs/Julia-1"


def test_import_pulls_no_heavy_dependencies():
    # the delta matters: another test (or venv) may already hold torch
    before = set(sys.modules)
    importlib.reload(julia)
    added = set(sys.modules) - before
    assert "julia" not in added
    assert "torch" not in added
    assert "transformers" not in added


def test_load_refuses_to_run_without_julia(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    # the heavy import sits inside load(), not at module level
    monkeypatch.setenv("JULIA_HOME", str(tmp_path))
    monkeypatch.setattr(julia, "_pop_stale_modules", lambda: None)
    with pytest.raises((ImportError, FileNotFoundError)):
        julia.load_engine()


def _stub_module(name: str) -> Any:
    """types.ModuleType, but attribute-assignable without checker complaints
    (the stubs must be real module objects for sys.modules)."""
    return cast(Any, types.ModuleType(name))


def _seed_julia_stubs(monkeypatch: pytest.MonkeyPatch):
    """sys.modules stubs for the snapshot's runtime: load_model captures
    its kwargs, predict() answers one canned choice question."""
    calls: list[dict] = []

    def load_model(home, device="cpu", strict_encoding=False, head_length=None, **kwargs):
        calls.append({"home": home, "device": device, "strict_encoding": strict_encoding,
                      "head_length": head_length})

        class _Engine:
            @staticmethod
            def predict(state, questions):
                return {"answers": {
                    name: {"type": "choice", "choice": "positive",
                           "max_probability": 0.9,
                           "probabilities": {"positive": 0.9, "negative": 0.1}}
                    for name in questions}}

        return _Engine()

    encoder_mod = _stub_module("julia.router.encoder")
    encoder_mod.specialize_decision_encoder = lambda model: True
    router_mod = _stub_module("julia.router")
    router_mod.encoder = encoder_mod
    julia_mod = _stub_module("julia")
    julia_mod.load_model = load_model
    julia_mod.router = router_mod
    monkeypatch.setitem(sys.modules, "julia", julia_mod)
    monkeypatch.setitem(sys.modules, "julia.router", router_mod)
    monkeypatch.setitem(sys.modules, "julia.router.encoder", encoder_mod)
    return calls, encoder_mod


@pytest.fixture()
def julia_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A minimal snapshot shape (julia/ package dir) so load()'s presence
    check passes without the real 550 MiB clone."""
    (tmp_path / "julia").mkdir()
    monkeypatch.setenv("JULIA_HOME", str(tmp_path))
    monkeypatch.setattr(julia, "_pop_stale_modules", lambda: None)
    yield tmp_path


class LoadTests:
    def test_load_applies_the_517_guard_and_the_card_kwargs(
        self, monkeypatch: pytest.MonkeyPatch, julia_home: Path
    ):
        calls, encoder_mod = _seed_julia_stubs(monkeypatch)
        engine = julia.load_engine(device="cpu")
        assert engine is not None
        # the card's kwargs reached load_model
        assert calls[-1]["strict_encoding"] is True
        assert calls[-1]["head_length"] == 512
        # the compat guard: the monkeypatch over the removed 5.x API is
        # disabled so the stock ModernBERT forward runs
        assert encoder_mod.specialize_decision_encoder("m") is False

    def test_home_stays_first_on_sys_path(self, monkeypatch: pytest.MonkeyPatch, julia_home: Path):
        calls, _ = _seed_julia_stubs(monkeypatch)
        julia.load_engine()
        assert calls[-1]["home"] == str(julia_home)
        sys.path.remove(str(julia_home))  # undo load()'s insert


class PredictTests:
    def test_max_probability_row_maps_to_the_runner_contract(
        self, monkeypatch: pytest.MonkeyPatch, julia_home: Path
    ):
        _seed_julia_stubs(monkeypatch)
        julia._ENGINE.clear()
        try:
            result = julia.predict({
                "input": {"text": "The food was cold."},
                "question": {"type": "choice", "options": ["positive", "negative"]},
            })
        finally:
            julia._ENGINE.clear()
            home = julia.model_home()
            while home in sys.path:
                sys.path.remove(home)
        # confidence is Julia's max_probability (full softmax, no separate
        # confidence field); answer_to_result reads the chosen label's own
        # probability, the same number
        assert result == {"prediction": "positive", "prob": 0.9}
