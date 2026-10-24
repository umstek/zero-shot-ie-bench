"""vLLM Semantic Router's Decision 3.0 family (d3-edge / d3-lite / d3-nano),
loaded in-process from local snapshots.

Ported from legacy/engines/d3_client.py: nothing is pip-installed — the
runtime ships inside each HF snapshot (d3_runtime.py + d3_format/d3_fast/
d3_kernels, backbone weights and readout.safetensors) and load()
sys.path-inserts the snapshot and imports d3_runtime directly. The load
verifies the package against MODEL_MANIFEST.json first ("fast": SHA-256
for every file up to 64 MiB, size for the weight shards). One runtime per
process: loading a second size pops the first snapshot's d3_* modules
first (all of them ship the same module names). The manifest keys are
POSIX-relative names — no Windows separator wart to patch (unlike
Decision 2.0).

The runtime imports transformers.models.qwen3_5 and
masking_utils.create_recurrent_attention_mask — 5.x-only code, and the
Gated DeltaNet layers run transformers' PyTorch reference on the CPU (the
fla kernels are GPU-only; the runtime refuses to load on a CPU when
transformers binds them — .venv-von's transformers 5.17.0 binds the
reference even with fla installed, measured). This island pins 5.17.0, the
validated stack. Backbone loads bf16 with SDPA (the runtime's hard-coded
choice), fp32 readout + softmax.

Homes: D3_HOME (a prefix directory; each size's snapshot sits at
<prefix>/d3-<size>) or the default C:\src base.

Catalog: vllm-sr/d3-{edge,lite,nano}, versions d3-*@unversioned,
deployments d3-*.local (runtime torch/bf16/cpu).
r"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from . import _systemone

# catalog model id -> benchmark/demo shorthand (legacy SIZES keys)
MODEL_IDS = {
    "vllm-sr/d3-edge": "edge",
    "vllm-sr/d3-lite": "lite",
    "vllm-sr/d3-nano": "nano",
}

# benchmark/demo shorthand -> snapshot directory name under the home prefix
SIZES = {
    "edge": "d3-edge",
    "lite": "d3-lite",
    "nano": "d3-nano",
}
DEFAULT_SIZE = "edge"

# Transcribed from catalog/models/vllm-sr/d3-*.yaml, deployment d3-*.local
# (all three share the runtime shape).
RUNTIME = {"engine": "torch", "dtype": "bf16", "quant": None, "device": "cpu"}

ENTRIES: list[dict[str, Any]] = [
    {"model_id": model_id, "version": f"d3-{short}@unversioned", "runtime": dict(RUNTIME)}
    for model_id, short in (
        ("vllm-sr/d3-edge", "edge"),
        ("vllm-sr/d3-lite", "lite"),
        ("vllm-sr/d3-nano", "nano"),
    )
]

# the flat runtime modules a snapshot ships; popped from sys.modules
# before a load so a second size imports its own packaged code
_SNAPSHOT_MODULES = ("d3_engine", "d3_fast", "d3_format", "d3_kernels", "d3_runtime", "d3_server")

_ENGINE: dict[str, Any] = {}
_LOCK = threading.Lock()


def normalize_size(size: Any) -> str:
    """Registry key for a size ('edge', 'd3-edge', full HF id or None).
    ValueError names the problem so callers can surface it."""
    if size is None or not str(size).strip():
        return DEFAULT_SIZE
    key = str(size).strip().lower()
    for canonical in SIZES:
        if key == canonical.lower():
            return canonical
    for canonical, repo in SIZES.items():
        if key == repo.lower() or key == f"vllm-sr/{repo}".lower():
            return canonical
    raise ValueError(f"unknown Decision 3.0 size {size!r} (one of {', '.join(SIZES)})")


def model_dir(size: Any = DEFAULT_SIZE) -> str:
    """Snapshot directory: D3_HOME prefix, default C:\\src."""
    base = os.environ.get("D3_HOME", r"C:\src")
    return os.path.join(base, SIZES[normalize_size(size)])


def _ensure_runtime_on_path(size: str) -> str:
    """Put the snapshot on sys.path so `import d3_runtime` resolves;
    returns the snapshot path (idempotent). Unconditional insert: a
    FAILED earlier load could leave a stale path entry, and a
    "not already present" guard would then import the wrong snapshot's
    modules — remove every existing occurrence, then index 0 always
    wins."""
    size = normalize_size(size)
    home = model_dir(size)
    if not os.path.isfile(os.path.join(home, "d3_runtime.py")):
        raise FileNotFoundError(
            f"no {SIZES[size]} snapshot at {home} — pull it with "
            f"`hf download vllm-sr/{SIZES[size]} --local-dir {home}` "
            "(or point D3_HOME at the prefix directory)"
        )
    while home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)
    return home


def available(size: Any = DEFAULT_SIZE) -> bool:
    """The snapshot (its d3_runtime.py) is locally present."""
    return os.path.isfile(os.path.join(model_dir(size), "d3_runtime.py"))


def _load(size: str, verify: str = "fast") -> Any:
    """D3 runtime handle over the local snapshot (bf16 backbone, fp32
    readout on the CPU). Heavy imports stay inside (offline tests import
    this module freely). The load verifies the package against
    MODEL_MANIFEST.json first ("fast": SHA-256 for every file up to
    64 MiB, size for the weight shards; "full" hashes every byte), then
    loads the backbone — expect the manifest pass plus the weight read
    (edge ~1.2 GB, nano ~4.4 GB) before the first answer."""
    size = normalize_size(size)
    home = _ensure_runtime_on_path(size)
    for stale in [name for name in sys.modules if name in _SNAPSHOT_MODULES]:
        del sys.modules[stale]
    from d3_runtime import D3  # noqa: PLC0415 — the heavy stack, lazy  # pyrefly: ignore[missing-import] — snapshot-resolved

    return D3.from_pretrained(home, device="cpu", verify=verify)


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item for one of the three catalog sizes:
    {input, question} -> {prediction, prob}."""
    size = MODEL_IDS.get(model_id)
    if size is None:
        raise KeyError(f"unknown model_id: {model_id!r} (served: {sorted(MODEL_IDS)})")
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    with _LOCK:
        engine = _ENGINE.get(size)
        if engine is None:
            engine = _load(size)
            _ENGINE[size] = engine
    out = engine.system_one(state=state, questions={"q": question})
    return _systemone.systemone_result(question["type"], out["answers"]["q"])
