"""AgentJev-0.6B — malevrigns' candidate-head decision model, in-process.

The odd one out on this island: it does NOT speak the TypeSafe System One
contract. Its server (the agent-jev clone's jev_service, historical port
8149) takes POST /api/evaluate with its own boolean/choice/score rows
(engines/agentjev_client.py is the legacy client); this island imports
jev_service.engine.DecisionEngine directly and converts the runner's
System One question to the clone's row shape (_systemone.agentjev_row)
and its answer back (_systemone.agentjev_answer_row).

  - AGENTJEV_HOME (default: the `agent-jev` clone next to this repo) goes
    on sys.path so `import jev_service` / `import agentjev` resolve.
  - AGENTJEV_CHECKPOINT (default HOME/agentjev_v1.pt) — the checkpoint the
    model card's conversion produces once.
  - AGENTJEV_MODEL_PATH (default HOME/models/Qwen3-0.6B-Base, else the
    Qwen/Qwen3-0.6B HF-cache snapshot) — the backbone the head rides on.
  - AGENTJEV_TEMPERATURES (default HOME/temperatures.json when present) —
    the fitted primitive temperatures; the engine raises on a bad one.
  - AGENTJEV_DEVICE (default cpu; fp32 — autocast is CUDA-only, so no
    dtype flag on CPU, matching the catalog's runtime).

The clone is not pip-installable and keeps its own requirements
(torch>=2.0, transformers>=4.40, pyarrow, pillow, requests). None of
those clash with this island's pins (torch 2.8.0, transformers 5.17.0),
but the clone stays optional: this module imports clean without it and
predict() fails with a named fix only when the engine is actually asked
for — the registry keeps serving the other three.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from . import _systemone
from ._base import DEFAULT_AGENTJEV_HOME, env_home, hf_snapshot

MODEL_ID = "agentjev-v1"
VERSION = "agentjev-v1@unversioned"  # catalog/models/malevrigns/agentjev.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

_BACKBONE_HUB = "Qwen/Qwen3-0.6B"

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def home() -> Path:
    """The agent-jev clone: AGENTJEV_HOME, defaulting to the sibling of this repo."""
    return env_home("AGENTJEV_HOME", DEFAULT_AGENTJEV_HOME)


def checkpoint() -> Path:
    """The converted decision checkpoint (agentjev_v1.pt)."""
    return Path(os.environ.get("AGENTJEV_CHECKPOINT", home() / "agentjev_v1.pt"))


def model_path() -> Path | None:
    """The Qwen3-0.6B backbone snapshot: explicit env, clone-local, HF cache."""
    explicit = os.environ.get("AGENTJEV_MODEL_PATH")
    if explicit:
        return Path(explicit)
    clone_local = home() / "models" / "Qwen3-0.6B-Base"
    if (clone_local / "config.json").is_file():
        return clone_local
    snapshot = hf_snapshot(_BACKBONE_HUB)
    if snapshot:
        return snapshot
    return None


def temperatures() -> str | None:
    """The fitted temperatures file when one exists (AGENTJEV_TEMPERATURES)."""
    explicit = os.environ.get("AGENTJEV_TEMPERATURES")
    if explicit:
        return explicit
    fitted = home() / "temperatures.json"
    return str(fitted) if fitted.is_file() else None


def available() -> bool:
    """The clone, its checkpoint and the backbone snapshot are all present."""
    if not (home() / "jev_service" / "engine.py").is_file():
        return False
    if not checkpoint().is_file():
        return False
    path = model_path()
    return path is not None and path.is_dir()


def _load() -> Any:
    """The lazy weight load (the clone's server loads at start; here it is
    paid on the first predict instead)."""
    with _LOCK:
        if _ENGINE:
            return _ENGINE[0]
        clone = home()
        if not (clone / "jev_service" / "engine.py").is_file():
            raise RuntimeError(
                f"agentjev-v1: no agent-jev clone at {clone} — clone "
                "https://github.com/malevrigns/agent-jev next to this repo or point "
                "AGENTJEV_HOME at it"
            )
        if str(clone) not in sys.path:
            sys.path.insert(0, str(clone))
        from jev_service.engine import DecisionEngine  # noqa: PLC0415 # pyrefly: ignore[missing-import] — the clone, on sys.path at runtime

        path = model_path()
        if path is None:
            raise RuntimeError(
                "agentjev-v1: no Qwen3-0.6B backbone — set AGENTJEV_MODEL_PATH to a "
                f"local snapshot (or point {checkpoint()} at a converted run)"
            )
        device = os.environ.get("AGENTJEV_DEVICE", "cpu")  # catalog requires: accel cpu
        _ENGINE.append(
            DecisionEngine(
                str(checkpoint()),
                str(path),
                device=device,
                temperatures=temperatures(),
            )
        )
        return _ENGINE[0]


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    One System One question becomes one /api/evaluate-shaped row; the id
    ("q") doubles as the answer key, and the clone's contract wants ids
    unique per call — one question per item, so that always holds.
    """
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    engine = _load()
    row = _systemone.agentjev_row(question)
    result = engine.evaluate({"state": state, "questions": [row]})
    answers = result["results"][0]["answers"]
    return _systemone.answer_to_result(
        question["type"], _systemone.agentjev_answer_row(answers[0])
    )
