"""Loading tests, ported from legacy/tests/test_von_loading.py.

The engine (von_runner/von.py, port of legacy engines/von_client.py) must
keep: no random-scorer fallback when option_marker.pt is missing, the
eager strict _get_model() load before the first answer, the pinned
checkpoint revision on every snapshot_download, and a module that imports
without torch / transformers / the SDK.
"""

import importlib
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from conftest import sdk_stubs

import von_runner.von as von_module
from von_runner.von import MODEL_REVISION, get_backend, load_von_decider, provenance, reset_backend


def test_missing_trained_weights_never_fall_back():
    with tempfile.TemporaryDirectory() as checkpoint:
        modules, _, factory, _ = sdk_stubs(checkpoint=checkpoint)
        with patch.dict(sys.modules, modules):
            try:
                load_von_decider()
                raised = None
            except RuntimeError as exc:
                raised = exc
        assert raised is not None and "weights are missing" in str(raised)
        factory.assert_not_called()


def test_complete_model_is_loaded_before_first_decision():
    with tempfile.TemporaryDirectory() as checkpoint:
        Path(checkpoint, "option_marker.pt").touch()
        modules, backend, factory, snapshot = sdk_stubs(checkpoint=checkpoint)
        with patch.dict(sys.modules, modules):
            decide = load_von_decider()
            backend._get_model.assert_called_once_with()
            backend.evaluate_choice.assert_not_called()
            factory.assert_called_once_with(checkpoint_dir=checkpoint, device="cpu")
            assert snapshot.call_args.kwargs["revision"] == MODEL_REVISION
            decide(state="text", choices={"positive": None}, instructions="sentiment")
            backend.evaluate_choice.assert_called_once()


def test_incomplete_state_dict_prevents_inference():
    with tempfile.TemporaryDirectory() as checkpoint:
        Path(checkpoint, "option_marker.pt").touch()
        modules, backend, _, _ = sdk_stubs(checkpoint=checkpoint)
        backend._get_model.side_effect = RuntimeError("Missing key: scorer.dense.weight")
        with patch.dict(sys.modules, modules):
            try:
                load_von_decider()
                raised = None
            except RuntimeError as exc:
                raised = exc
        assert raised is not None and "Missing key" in str(raised)
        backend.evaluate_choice.assert_not_called()


def test_provenance_carries_the_identity_tuple():
    assert provenance() == {
        "model": "wfzyx/von-1.0",
        "model_revision": "fbc0196efa4fa1825557a8d5adad464fa1e872bb",
        "sdk_revision": "a94aa368ff4dcc58cdfd569cf0b4b1ae7d592d54",
        "backend": "option-marker",
    }


def test_get_backend_loads_once_and_reset_forces_a_reload():
    loads = Mock()
    with patch.object(von_module, "load_von_backend", loads):
        first = get_backend()
        assert get_backend() is first
        loads.assert_called_once()
        reset_backend()
        get_backend()
        assert loads.call_count == 2


def test_engine_module_never_imports_heavy_dependencies():
    # port of legacy ImportTests: another test (or venv) may already hold
    # torch, so the delta around a fresh import of the engine is what
    # matters. The real SDK lives in this venv — importing the engine must
    # not touch it (weights load lazily inside load_von_backend).
    saved = {name: module for name, module in sys.modules.items() if name == "von_runner.von"}
    for name in saved:
        del sys.modules[name]
    before = set(sys.modules)
    try:
        module = importlib.import_module("von_runner.von")
        added = set(sys.modules) - before
        assert module is not None
        for heavy in ("torch", "transformers", "huggingface_hub"):
            assert heavy not in added, heavy
        sdk = [name for name in added if name == "von" or name.startswith("von.")]
        assert not sdk, sdk
    finally:
        sys.modules.update(saved)
