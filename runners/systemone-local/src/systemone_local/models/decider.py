"""decider — Mapika's decider checkpoints, loaded in-process from PyPI decider-ai.

The catalog serves them through the package's own System One server
(`uvicorn decider.serve:app` with DECIDER_MODEL=Mapika/decider-0.8b,
historical port 8018, lazy weight load on the first request); this island
skips the HTTP hop and calls decider.infer.Decider.system_one() directly
— the same engine, question rendering and temperatures, in-process.

Decider is checkpoint-agnostic (the decider-2b card loads it exactly the
same way: `Decider("Mapika/decider-2b")`), so this module is both the 0.8b
entry (MODEL_ID decider-0.8b) and the family's shared machinery home:
checkpoint_available / checkpoint_predict take the hub repo id and the
env-var names, and decider2.py binds them to Mapika/decider-2b.

  - DECIDER_MODEL (default Mapika/decider-0.8b) — the 0.8b hub repo id.
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
WEIGHTS_ENV = "DECIDER_WEIGHTS"
DEVICE_ENV = "DECIDER_DEVICE"

# One engine slot per checkpoint repo (each loads lazily on its first
# predict and stays); the load lock is shared across checkpoints — CPU
# loads run minutes, so serialize them instead of racing two multi-GB
# weight reads into RAM at once.
_ENGINES: dict[str, list[Any]] = {}
_LOAD_LOCK = threading.Lock()


def checkpoint_available(repo: str, weights_env: str) -> bool:
    """The decider package is installed and `repo`'s weights are locally present.

    The historical serve pulled the hub snapshot into the HF cache before
    answering; a miss would download at load time, which available() does
    not count as available.
    """
    if find_spec("decider") is None:
        return False
    if os.environ.get(weights_env):
        return True  # explicit snapshot; its presence is checked at load
    return hf_snapshot(repo) is not None


def checkpoint_load(repo: str, weights_env: str, device_env: str) -> Any:
    """The lazy weight load for one checkpoint (decider lazy-loads on the
    first request too)."""
    slot = _ENGINES.get(repo)
    if slot:
        return slot[0]
    with _LOAD_LOCK:
        slot = _ENGINES.get(repo)
        if slot:
            return slot[0]
        from decider.infer import Decider  # noqa: PLC0415 — the heavy stack, lazy

        path = pick_path(repo, weights_env)
        device = os.environ.get(device_env, "cpu")  # catalog requires: accel cpu
        _ENGINES[repo] = [Decider(path, device=device, use_graphs=False)]
        return _ENGINES[repo][0]


def checkpoint_predict(
    payload: dict[str, Any], repo: str, weights_env: str, device_env: str
) -> dict[str, Any]:
    """One decision item for `repo`: {input, question} -> {prediction, prob}.

    system_one() scores each question in its own row (independent=True —
    answers cannot see each other), exactly the server default.
    """
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    engine = checkpoint_load(repo, weights_env, device_env)
    answers = engine.system_one(state, {"q": question})["answers"]
    return _systemone.answer_to_result(question["type"], answers["q"])


def repo_id() -> str:
    """The weights repo: DECIDER_MODEL or the catalog's Mapika/decider-0.8b."""
    return os.environ.get("DECIDER_MODEL", DEFAULT_REPO)


def available() -> bool:
    """The decider package is installed and the 0.8b weights are present."""
    return checkpoint_available(repo_id(), WEIGHTS_ENV)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob} (0.8b)."""
    return checkpoint_predict(payload, repo_id(), WEIGHTS_ENV, DEVICE_ENV)
