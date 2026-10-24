"""Shared stubs for the von runner tests.

The pinned von-sdk only really runs in the cache-gated slow test
(test_real_weights.py); everything else stubs the modules the engine
imports lazily (huggingface_hub.snapshot_download, von.backends.
option_marker_backend, von.types) through sys.modules — the same patching
legacy tests/test_von_loading.py used, and offline by construction.
"""

import types
from typing import Any
from unittest.mock import Mock

import pytest

import von_runner.von as von_module

# The von.types dataclasses the wrappers instantiate; kwargs-in, dict-out
# makes every call site assertable.
VON_TYPES_STUB = types.SimpleNamespace(
    Choice=lambda **kwargs: kwargs,
    Noul=lambda **kwargs: kwargs,
    Score=lambda **kwargs: kwargs,
)


def sdk_stubs(backend: Any = None, checkpoint: Any = None):
    """sys.modules patch set faking the SDK + hub modules and capturing
    the factory calls. Returns (modules, backend, backend_factory, snapshot)."""
    backend = backend if backend is not None else Mock(_default_temp=1.0)
    backend_factory = Mock(return_value=backend)
    snapshot = Mock(return_value=checkpoint)
    modules = {
        "huggingface_hub": types.SimpleNamespace(snapshot_download=snapshot),
        "von": types.ModuleType("von"),
        "von.backends": types.ModuleType("von.backends"),
        "von.backends.option_marker_backend": types.SimpleNamespace(
            OptionMarkerBackend=backend_factory
        ),
        "von.types": VON_TYPES_STUB,
    }
    return modules, backend, backend_factory, snapshot


def choice_answer(
    choice: str, probabilities: dict | None = None, confidence: float = 1.0
) -> types.SimpleNamespace:
    """A ChoiceAnswer stand-in (legacy row(): choice/probabilities/confidence)."""
    return types.SimpleNamespace(
        choice=choice,
        probabilities=probabilities if probabilities is not None else {choice: 1.0},
        confidence=confidence,
    )


def noul_answer(noul: float) -> types.SimpleNamespace:
    """A NoulAnswer stand-in (judgment_row(): noul -> derived verdict)."""
    return types.SimpleNamespace(noul=noul)


def score_answer(
    score: float,
    confidence: float = 0.4,
    legend: dict | None = None,
    probabilities: dict | None = None,
) -> types.SimpleNamespace:
    """A ScoreAnswer stand-in (rating_row(): score/confidence/legend/
    probabilities keyed by level index)."""
    return types.SimpleNamespace(
        score=score,
        confidence=confidence,
        legend=legend if legend is not None else {},
        probabilities=probabilities if probabilities is not None else {},
    )


def stub_backend() -> Mock:
    """A von backend stand-in with the three evaluate_* methods wired to
    the answer stand-ins at the temperatures the calibration would set."""
    backend = Mock(_default_temp=1.0)
    backend.evaluate_choice.return_value = choice_answer(
        "positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}, 0.9
    )
    backend.evaluate_noul.return_value = noul_answer(0.87)
    backend.evaluate_score.return_value = score_answer(
        2.4, 0.4, {"0": "wait", "1": "soon", "2": "now"}, {"0": 0.1, "1": 0.2, "2": 0.7}
    )
    return backend


@pytest.fixture(autouse=True)
def _no_shared_backend():
    """Keep the module-level backend cache empty around every test: the
    server tests install their stub via load_von_backend; nothing may
    leak a loaded (or stubbed) backend into another test."""
    von_module._BACKEND = None
    yield
    von_module._BACKEND = None
