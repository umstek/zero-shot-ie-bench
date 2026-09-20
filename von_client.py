"""Load von's complete, trained option-marker checkpoint on CPU.

The PyPI SDK's default NLI loader invents a random classifier for this
checkpoint. The pinned Git SDK has the correct architecture, but falls back
to a random scorer if option_marker.pt is absent. Never allow that fallback.
"""

from pathlib import Path

MODEL_ID = "wfzyx/von-1.0"
MODEL_REVISION = "fbc0196efa4fa1825557a8d5adad464fa1e872bb"
SDK_REVISION = "a94aa368ff4dcc58cdfd569cf0b4b1ae7d592d54"
CHECKPOINT_FILES = [
    "config.json", "model.safetensors", "option_marker.pt",
    "tokenizer.json", "tokenizer_config.json", "marker_calibration.json",
]


def provenance():
    return {"model": MODEL_ID, "model_revision": MODEL_REVISION,
            "sdk_revision": SDK_REVISION, "backend": "option-marker"}


def load_von_decider():
    """Download and strictly load all weights before returning decide()."""
    from huggingface_hub import snapshot_download
    try:
        from von.backends.option_marker_backend import OptionMarkerBackend
        from von.types import Choice
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-von.txt in .venv-von; the PyPI SDK does "
            "not include the trained option-marker loader.") from exc

    checkpoint = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, allow_patterns=CHECKPOINT_FILES))
    if not (checkpoint / "option_marker.pt").is_file():
        raise RuntimeError("Refusing von inference: trained option_marker.pt "
                           "weights are missing.")
    backend = OptionMarkerBackend(checkpoint_dir=str(checkpoint), device="cpu")
    # Pinned SDK uses weights_only=True and strict load_state_dict here.
    # Eager loading both validates the trained head and excludes setup from timing.
    backend._get_model()

    def decide(*, state, choices, instructions):
        return backend.evaluate_choice(
            "decision", state, Choice(instructions=instructions, criteria=choices),
            temperature=backend._default_temp)

    return decide
