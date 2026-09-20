from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from von_client import MODEL_REVISION, load_von_decider


class VonLoadingTests(unittest.TestCase):
    def fake_dependencies(self, checkpoint):
        backend = Mock()
        backend._default_temp = 1.0
        backend_factory = Mock(return_value=backend)
        snapshot = Mock(return_value=checkpoint)
        modules = {
            "huggingface_hub": types.SimpleNamespace(snapshot_download=snapshot),
            "von": types.ModuleType("von"),
            "von.backends": types.ModuleType("von.backends"),
            "von.backends.option_marker_backend": types.SimpleNamespace(
                OptionMarkerBackend=backend_factory),
            "von.types": types.SimpleNamespace(Choice=lambda **kwargs: kwargs),
        }
        return modules, backend, backend_factory, snapshot

    def test_missing_trained_weights_never_fall_back(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            modules, _, factory, _ = self.fake_dependencies(checkpoint)
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "weights are missing"):
                    load_von_decider()
            factory.assert_not_called()

    def test_complete_model_is_loaded_before_first_decision(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            Path(checkpoint, "option_marker.pt").touch()
            modules, backend, factory, snapshot = self.fake_dependencies(checkpoint)
            with patch.dict(sys.modules, modules):
                decide = load_von_decider()
                backend._get_model.assert_called_once_with()
                backend.evaluate_choice.assert_not_called()
                factory.assert_called_once_with(checkpoint_dir=checkpoint, device="cpu")
                self.assertEqual(snapshot.call_args.kwargs["revision"], MODEL_REVISION)
                decide(state="text", choices={"positive": None}, instructions="sentiment")
                backend.evaluate_choice.assert_called_once()

    def test_incomplete_state_dict_prevents_inference(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            Path(checkpoint, "option_marker.pt").touch()
            modules, backend, _, _ = self.fake_dependencies(checkpoint)
            backend._get_model.side_effect = RuntimeError("Missing key: scorer.dense.weight")
            with patch.dict(sys.modules, modules):
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    load_von_decider()
            backend.evaluate_choice.assert_not_called()


if __name__ == "__main__":
    unittest.main()
