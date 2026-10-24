"""decider-0.8b — Mapika's decider, loaded in-process from PyPI decider-ai.

The catalog serves it through the package's own System One server
(`uvicorn decider.serve:app` with DECIDER_MODEL=Mapika/decider-0.8b,
historical port 8018, lazy weight load on the first request); this island
skips the HTTP hop and calls decider.infer.Decider.system_one() directly
— the same engine, question rendering and temperatures, in-process.

  - DECIDER_MODEL (default Mapika/decider-0.8b) — the hub repo id.
  - DECIDER_WEIGHTS (optional) — a local snapshot directory (the
    D1_HOME-style override); otherwise the HF cache is used, and a cache
    miss downloads on first predict.
  - DECIDER_DEVICE (default cpu — the catalog requires accel cpu).

decider-ai needs transformers>=5 and an unpinned torch; both are pinned
by this island to the kev clone's validated stack.
"""

from __future__ import annotations

import os
import threading
from importlib.util import find_spec
from typing import Any

from . import _systemone
from ._base import hf_snapshot, pick_path

MODEL_ID = "decider-0.8b"
VERSION = "decider-0.8b@unversioned"  # catalog/models/mapika/decider.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

DEFAULT_REPO = "Mapika/decider-0.8b"

_ENGINE: list[Any] = []
_LOCK = threading.Lock()


def repo_id() -> str:
    """The weights repo: DECIDER_MODEL or the catalog's Mapika/decider-0.8b."""
    return os.environ.get("DECIDER_MODEL", DEFAULT_REPO)


def available() -> bool:
    """The decider package is installed and the weights are locally present.

    The historical serve pulled the hub snapshot into the HF cache before
    answering; a miss would download at load time, which available() does
    not count as available.
    """
    if find_spec("decider") is None:
        return False
    if os.environ.get("DECIDER_WEIGHTS"):
        return True  # explicit snapshot; its presence is checked at load
    snapshot = hf_snapshot(repo_id())
    return snapshot is not None


def _load() -> Any:
    """The lazy weight load (decider lazy-loads on the first request too)."""
    with _LOCK:
        if _ENGINE:
            return _ENGINE[0]
        from decider.infer import Decider  # noqa: PLC0415 — the heavy stack, lazy

        path = pick_path(repo_id(), "DECIDER_WEIGHTS")
        device = os.environ.get("DECIDER_DEVICE", "cpu")  # catalog requires: accel cpu
        _ENGINE.append(Decider(path, device=device, use_graphs=False))
        return _ENGINE[0]


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    system_one() scores each question in its own row (independent=True —
    answers cannot see each other), exactly the server default.
    """
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    engine = _load()
    answers = engine.system_one(state, {"q": question})["answers"]
    return _systemone.answer_to_result(question["type"], answers["q"])
