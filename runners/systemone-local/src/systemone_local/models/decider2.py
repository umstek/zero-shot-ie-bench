"""decider-2b — Mapika's larger decider checkpoint, same PyPI engine as decider-0.8b.

decider.infer.Decider is checkpoint-agnostic (the 2b model card loads it
exactly like the 0.8b: `Decider("Mapika/decider-2b")`, text inference from
decider-ai 1.4.0+ — this island already pins >=1.10), so this module is a
thin twin of decider.py: the same Decider.system_one() call through
decider's shared per-checkpoint machinery, bound to the 2b hub repo and
its own env-var names (DECIDER2_*, so the historical single-checkpoint
DECIDER_MODEL override keeps steering only the 0.8b entry).

  - DECIDER2_MODEL (default Mapika/decider-2b) — the hub repo id.
  - DECIDER2_WEIGHTS (optional) — a local snapshot directory (the
    D1_HOME-style override); otherwise the HF cache is used, and a cache
    miss downloads on first predict.
  - DECIDER2_DEVICE (default cpu — the catalog requires accel cpu).
"""

from __future__ import annotations

import os
from typing import Any

from . import decider

MODEL_ID = "decider-2b"
VERSION = "decider-2b@unversioned"  # catalog/models/mapika/decider.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

DEFAULT_REPO = "Mapika/decider-2b"
WEIGHTS_ENV = "DECIDER2_WEIGHTS"
DEVICE_ENV = "DECIDER2_DEVICE"


def repo_id() -> str:
    """The weights repo: DECIDER2_MODEL or the catalog's Mapika/decider-2b."""
    return os.environ.get("DECIDER2_MODEL", DEFAULT_REPO)


def available() -> bool:
    """The decider package is installed and the 2b weights are present."""
    return decider.checkpoint_available(repo_id(), WEIGHTS_ENV)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob} (2b)."""
    return decider.checkpoint_predict(payload, repo_id(), WEIGHTS_ENV, DEVICE_ENV)
