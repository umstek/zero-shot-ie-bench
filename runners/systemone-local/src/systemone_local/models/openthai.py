"""OpenThai-SystemOne 0.8B — iApp's Thai/English decision model, in-process.

The catalog serves it through the package's own server (`uvicorn
openthai_systemone.server:app` with OPENTHAI_SYSTEMONE_MODEL=
iapp/OpenThai-SystemOne, historical port 8029); this island calls
openthai_systemone.client.SystemOneClient.system_one() directly — the
same 256-way slot head, formatting and abstain slot, in-process.

  - OPENTHAI_MODEL (default iapp/OpenThai-SystemOne) — the hub repo id.
  - OPENTHAI_WEIGHTS (optional) — a local snapshot directory (the
    D1_HOME-style override); otherwise the HF cache is used. The weights
    are GATED on HF: a cache miss downloads only with an HF token.
  - OPENTHAI_DEVICE (default cpu — the catalog requires accel cpu).

The cold load runs minutes (the catalog's warm-up timeout is 600 s for
the same reason); the client picks fp32 on CPU like the server does.
"""

from __future__ import annotations

import os
import threading
from importlib.util import find_spec
from typing import Any

from . import _systemone
from ._base import hf_snapshot, pick_path

MODEL_ID = "openthai-systemone"
VERSION = "openthai-systemone@unversioned"  # catalog/models/iapp/openthai.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

DEFAULT_REPO = "iapp/OpenThai-SystemOne"

_CLIENT: list[Any] = []
_LOCK = threading.Lock()


def repo_id() -> str:
    """The weights repo: OPENTHAI_MODEL or the catalog's iapp/OpenThai-SystemOne."""
    return os.environ.get("OPENTHAI_MODEL", DEFAULT_REPO)


def available() -> bool:
    """The openthai package is installed and the (gated) weights are local.

    A gated cache miss needs an HF token to download; available() does not
    count it as available, so tests and cold CI stay offline. find_spec,
    not import: the package's __init__ pulls torch/transformers.
    """
    if find_spec("openthai_systemone") is None:
        return False
    if os.environ.get("OPENTHAI_WEIGHTS"):
        return True  # explicit snapshot; its presence is checked at load
    snapshot = hf_snapshot(repo_id())
    return snapshot is not None


def _load() -> Any:
    """The lazy weight load (minutes cold; the reason the catalog's warm-up
    timeout is 600 s)."""
    with _LOCK:
        if _CLIENT:
            return _CLIENT[0]
        from openthai_systemone.client import SystemOneClient  # noqa: PLC0415 — heavy, lazy

        path = pick_path(repo_id(), "OPENTHAI_WEIGHTS")
        device = os.environ.get("OPENTHAI_DEVICE", "cpu")  # catalog requires: accel cpu
        _CLIENT.append(SystemOneClient(path, device=device, model_name=MODEL_ID))
        return _CLIENT[0]


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}."""
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    client = _load()
    response = client.system_one(state, {"q": question})
    row = response.answers["q"]
    return _systemone.answer_to_result(
        question["type"], row.model_dump() if hasattr(row, "model_dump") else dict(row)
    )
