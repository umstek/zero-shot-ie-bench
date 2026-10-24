"""kev-0.8b — jaredpalmer's open Jev lookalike, loaded in-process.

The engine is not a package: it lives in the jaredpalmer/kev clone
(https://github.com/jaredpalmer/kev — a Qwen3.5-0.8B backbone + LoRA +
pointer head serving the System One contract). Legacy ran it as its own
uvicorn server on 127.0.0.1:8009 (`uv run --extra serve --project ../kev
python -m kev.serve --run jaredpalmer/kev-0.8b`); this island imports the
clone's kev package directly and answers in-process:

  - KEV_HOME (default: the `kev` clone next to this repo) goes on
    sys.path so `import kev` resolves.
  - KEV_WEIGHTS (optional) is a run directory (head.pt + adapter);
    otherwise KEV_RUN (default jaredpalmer/kev-0.8b) resolves through
    the HF cache, or downloads from the hub on first use.

The clone needs transformers >=5.17 + torch 2.6..2.9 + peft — the anchor
stack of this island (its serving venv validated 5.17.0 / 2.8.0 /
0.21.0), so kev, decider and OpenThai share one env here.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

from . import _systemone
from ._base import DEFAULT_KEV_HOME, env_home, hf_snapshot

MODEL_ID = "kev-0.8b"
VERSION = "kev-0.8b@2026-09"  # catalog/models/jaredpalmer/kev-0.8b.yaml
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu", "batch": None}

DEFAULT_RUN = "jaredpalmer/kev-0.8b"
# kev.serve's inference ceilings (training used 384/640; per-branch cap
# mirrors Jev's ~32k, bounded by the base model window)
MAX_STATE_TOKENS = 8192
MAX_BRANCH_TOKENS = 8192

_STATE: dict[str, Any] = {"tok": None, "model": None, "device": None}
_LOCK = threading.Lock()


def home() -> Path:
    """The kev clone: KEV_HOME, defaulting to the sibling of this repo."""
    return env_home("KEV_HOME", DEFAULT_KEV_HOME)


def run_id() -> str:
    """The run to serve: KEV_WEIGHTS (a run dir) or KEV_RUN (default hub id)."""
    if os.environ.get("KEV_WEIGHTS"):
        return os.environ["KEV_WEIGHTS"]
    return os.environ.get("KEV_RUN", DEFAULT_RUN)


def _resolved_run() -> str:
    """A local run dir, else the HF-cache snapshot, else the hub id (kev.evaluate.resolve_run then downloads on first use — live serving only)."""
    run = run_id()
    if os.path.isdir(run):
        return run
    snapshot = hf_snapshot(run.partition("@")[0])
    return str(snapshot) if snapshot else run


def available() -> bool:
    """The kev clone is present and the run's weights are locally resolvable.

    A cache miss would download at load time; available() does not count
    that as available, so tests and cold CI stay offline.
    """
    if not (home() / "kev" / "model.py").is_file():
        return False
    run = run_id()
    if os.path.isdir(run):
        return (Path(run) / "head.pt").is_file()
    snapshot = hf_snapshot(run.partition("@")[0])
    return snapshot is not None and (snapshot / "head.pt").is_file()


def _load() -> None:
    """The lazy weight load (minutes cold on CPU); kev.serve's main() path."""
    with _LOCK:
        if _STATE["model"] is not None:
            return
        clone = home()
        if not (clone / "kev" / "model.py").is_file():
            raise RuntimeError(
                f"kev-0.8b: no kev clone at {clone} — clone https://github.com/jaredpalmer/kev "
                "next to this repo or point KEV_HOME at it"
            )
        if str(clone) not in sys.path:
            sys.path.insert(0, str(clone))
        from kev.evaluate import load  # noqa: PLC0415 # pyrefly: ignore[missing-import] — the clone, on sys.path at runtime

        device = os.environ.get("KEV_DEVICE", "cpu")  # catalog requires: accel cpu
        _STATE["tok"], _STATE["model"] = load(_resolved_run(), device)
        _STATE["device"] = device


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    The same one-pass path kev.serve's POST /v1/systemone takes (packed
    questions behind a shared state prefix) with a single question per
    item; probabilities are taken unrounded (kev.api.to_answers rounds to
    2 decimals for the wire format).
    """
    from kev.api import SystemOneRequest, to_record  # noqa: PLC0415 # pyrefly: ignore[missing-import] — the clone, on sys.path at runtime

    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    _load()
    request = SystemOneRequest(state=state, model=MODEL_ID, questions={"q": question})
    record, meta = to_record(request)
    model = _STATE["model"]
    encoded = model.encode(
        _STATE["tok"], record, max_state=MAX_STATE_TOKENS, max_branch=MAX_BRANCH_TOKENS
    )
    probs = [p.tolist() for p in model.probs(encoded)]
    entry = meta[0]
    if entry["type"] == "choice":
        row = _systemone.choice_row(entry["keys"], probs[0])
    elif entry["type"] == "noul":
        row = _systemone.noul_row(probs[0][1])  # to_record orders the options [no, yes]
    else:
        row = _systemone.score_row(probs[0])
    return _systemone.answer_to_result(question["type"], row)
