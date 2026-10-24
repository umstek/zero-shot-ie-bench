"""internlm's Intern-Decision family (0.8B / 2B / 4B), loaded in-process
from local HF snapshots.

Ported from legacy/engines/intern_decision_client.py: fine-tuned Qwen3.5
(whose upstream weights Alibaba Cloud ship under a second Apache-2.0
LICENSE-QWEN file). The mechanism: one CAUSAL forward pass over the state,
the decision schema and a complete assistant JSON skeleton whose every
field holds a masked `<decision>` placeholder; logits are read at the
position immediately BEFORE each placeholder and softmaxed over that
field's candidate symbols (A,B,...,Z,a,...,z,0,...,9 — up to 62 options,
1-16 questions), then a per-checkpoint probability-calibration temperature
rescales the softmax (argmax preserved). No generate() call.

The runtime ships INSIDE each HF snapshot (a standalone inference.py +
requirements.txt, like Julia's): there is nothing to pip-install, the
loader sys.path-inserts the snapshot directory and imports its `inference`
module. Each snapshot's inference.py carries that checkpoint's own
DEFAULT_TEMPERATURE calibration constant, so every size must load with ITS
OWN snapshot dir — the loader caches one runtime module per resolved home
(all three are named inference.py, so a naive sys.path insert would
silently reuse the first one).

The snapshots' requirements.txt pins torch==2.9.1/transformers==5.14.1;
those pins are ignored — this island's torch 2.14.1+cpu and transformers
5.17.0 satisfy it (Qwen3_5ForConditionalGeneration and the processor load
and predict soundly on 5.17, verified on the 0.8B; no compat guard needed).
transformers falls back to reference PyTorch implementations for the
Qwen3.5 linear-attention kernels on CPU (causal_conv1d and
flash-linear-attention are GPU packages) — correct, just slower.

Dtype: float32 on CPU, family-wide (the bf16 forward runs ~7x slower with
identical predictions — measured on the 0.8B).

Homes: INTERN_DECISION_HOME (a prefix directory; each size's snapshot sits
at <prefix>/Intern-Decision-<size>) or the default C:\src base.

This engine takes ONE request dict: predict({"state": ..., "questions":
{...}}) — the state may be any JSON value and is rendered into the prompt
(the snapshot's runtime handles that; images ride the same request but the
benches exercise the text path).

Catalog: internlm/intern-decision-{0.8b,2b,4b}, versions
intern-decision-{0.8b,2b,4b}@unversioned, deployments *.local
(runtime torch/fp32/cpu).
r"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from . import _systemone

# catalog model id -> benchmark/demo shorthand (legacy SIZES keys)
MODEL_IDS = {
    "internlm/intern-decision-0.8b": "0.8B",
    "internlm/intern-decision-2b": "2B",
    "internlm/intern-decision-4b": "4B",
}

# benchmark/demo shorthand -> HF checkpoint (README: Family maps)
SIZES = {
    "0.8B": "internlm/Intern-Decision-0.8B",
    "2B": "internlm/Intern-Decision-2B",
    "4B": "internlm/Intern-Decision-4B",
}
DEFAULT_SIZE = "0.8B"

# Transcribed from catalog/models/internlm/*.yaml, deployment *.local
# (all three share the runtime shape).
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

ENTRIES: list[dict[str, Any]] = [
    {"model_id": model_id, "version": f"intern-decision-{short.lower()}@unversioned", "runtime": dict(RUNTIME)}
    for model_id, short in (
        ("internlm/intern-decision-0.8b", "0.8B"),
        ("internlm/intern-decision-2b", "2B"),
        ("internlm/intern-decision-4b", "4B"),
    )
]

# one runtime module per resolved snapshot home (all ship inference.py)
_RUNTIMES: dict[str, Any] = {}
# one loaded DecisionEngine per size (its snapshot owns the calibration)
_ENGINE_KEYED: dict[str, Any] = {}
_LOCK = threading.Lock()


def normalize_size(size: Any) -> str:
    """Registry key for a size ('0.8b', '0.8B', full HF id or None).
    ValueError names the problem so callers can surface it."""
    if size is None or not str(size).strip():
        return DEFAULT_SIZE
    key = str(size).strip().lower()
    for canonical in SIZES:
        if key == canonical.lower():
            return canonical
    for canonical, hf_id in SIZES.items():
        if key == hf_id.lower():
            return canonical
    raise ValueError(f"unknown Intern-Decision size {size!r}; known: {', '.join(SIZES)} or the HF id")


def size_home(size: Any) -> str:
    """Snapshot directory for one size: <INTERN_DECISION_HOME or
    C:\\src>/Intern-Decision-<size>, resolved absolute."""
    size = normalize_size(size)
    base = os.environ.get("INTERN_DECISION_HOME", r"C:\src")
    return os.path.abspath(os.path.join(base, f"Intern-Decision-{size}"))


def available(size: Any = DEFAULT_SIZE) -> bool:
    """The size's snapshot directory is locally present."""
    return os.path.isdir(size_home(size))


def _load_runtime(home: str) -> Any:
    """Import one snapshot's standalone inference.py, cached per resolved
    home so each size runs its own DEFAULT_TEMPERATURE constant."""
    runtime = _RUNTIMES.get(home)
    if runtime is None:
        # Unconditional insert: a FAILED load of one size leaves its home
        # on sys.path, and a "not already present" guard would let a
        # retried import of another size resolve THAT snapshot's
        # inference.py instead. Insertion at index 0 always wins.
        sys.path.insert(0, home)
        stale = sys.modules.pop("inference", None)  # a sibling snapshot's
        try:
            import inference as runtime  # pyrefly: ignore[missing-import] — the snapshot's standalone runtime
        finally:
            sys.modules.pop("inference", None)  # sys.modules hygiene only
            if stale is not None:  # restore whatever "inference" was
                sys.modules["inference"] = stale
        _RUNTIMES[home] = runtime
    return runtime


def load_engine(size: Any = None, device: str = "cpu", dtype: str = "float32") -> Any:
    """DecisionEngine over the local snapshot of one size.

    Heavy imports stay inside (offline tests import this module freely).
    checkpoint= is passed explicitly so each size loads with its OWN
    snapshot — its inference.py's DEFAULT_TEMPERATURE calibration
    constant therefore applies per checkpoint."""
    size = normalize_size(size)
    home = size_home(size)
    if not os.path.isdir(home):
        raise FileNotFoundError(
            f"no Intern-Decision-{size} snapshot at {home} — point "
            "INTERN_DECISION_HOME at the directory holding the "
            "Intern-Decision-0.8B/-2B/-4B snapshots"
        )
    runtime = _load_runtime(home)
    return runtime.DecisionEngine(checkpoint=home, device=device, dtype=dtype)


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item for one of the three catalog sizes:
    {input, question} -> {prediction, prob}."""
    size = MODEL_IDS.get(model_id)
    if size is None:
        raise KeyError(f"unknown model_id: {model_id!r} (served: {sorted(MODEL_IDS)})")
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        engine = _ENGINE_KEYED.get(size)
        if engine is None:
            engine = load_engine(size)
            _ENGINE_KEYED[size] = engine
    out = engine.predict({"state": state, "questions": {"q": question}})
    return _systemone.systemone_result(question["type"], out["answers"]["q"])
