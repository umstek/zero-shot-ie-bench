"""Loader for internlm's Intern-Decision family (local HF snapshots).

Intern-Decision (Apache 2.0; fine-tuned from Qwen3.5, whose upstream
weights Alibaba Cloud ship under a second Apache-2.0 LICENSE-QWEN file)
is a multimodal structured-decision line in three checkpoints:
Intern-Decision-0.8B (853M), -2B (2.2B) and -4B (4.5B). The mechanism:
one CAUSAL forward pass over the state, the decision schema and a
complete assistant JSON skeleton whose every field holds a masked
`<decision>` placeholder; logits are read at the position immediately
BEFORE each placeholder and softmaxed over that field's candidate
symbols (A,B,...,Z,a,...,z,0,...,9 — up to 62 options, 1-16 questions),
then a per-checkpoint probability-calibration temperature rescales the
softmax (argmax preserved). No generate() call.

The runtime ships INSIDE each HF snapshot (a standalone inference.py +
requirements.txt, like Julia's): there is nothing to pip-install, the
loader sys.path-inserts the snapshot directory and imports its
`inference` module. Each snapshot's inference.py carries that
checkpoint's own DEFAULT_TEMPERATURE calibration constant, so every
size must load with ITS OWN snapshot dir — the loader caches one
runtime module per resolved home (all three are named inference.py, so
a naive sys.path insert would silently reuse the first one).

The snapshots' requirements.txt pins torch==2.9.1/transformers==5.14.1;
those pins are ignored — .venv-von's torch 2.14.0+cpu and transformers
5.17.0 satisfy it (Qwen3_5ForConditionalGeneration and the processor
load and predict soundly on 5.17, verified on the 0.8B; no compat guard
needed). transformers falls back to reference PyTorch implementations
for the Qwen3.5 linear-attention kernels on CPU (causal_conv1d and
flash-linear-attention are GPU packages) — correct, just slower.

Dtype: float32 on CPU, family-wide. Measured on the 0.8B over the eight
easy-tier sentiment texts (deterministic across repeats): the bf16
forward runs ~7x slower (9.2 s vs 1.3 s per predict) with identical
8/8 predictions, so bfloat16 buys nothing on this CPU — the loader
defaults to float32 and the demo/benches pass nothing.

Homes: INTERN_DECISION_HOME (a prefix directory; each size's snapshot
sits at <prefix>/Intern-Decision-<size>) or the default C:\\src base —
engines resolve to C:\\src\\Intern-Decision-0.8B/-2B/-4B.

Unlike Lumma's decide(state, questions) and Julia's predict(state,
questions), this engine takes ONE request dict:
engine.predict({"state": ..., "questions": {...}}) — the state may be
any JSON value and is rendered into the prompt; images are supported by
the checkpoints but this repo is text-only.
"""

import os
import sys

# benchmark/demo shorthand -> HF checkpoint (README: Family maps)
SIZES = {
    "0.8B": "internlm/Intern-Decision-0.8B",
    "2B": "internlm/Intern-Decision-2B",
    "4B": "internlm/Intern-Decision-4B",
}
DEFAULT_SIZE = "0.8B"

# one runtime module per resolved snapshot home (all ship inference.py)
_RUNTIMES: dict[str, object] = {}


def normalize_size(size) -> str:
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
    raise ValueError(f"unknown Intern-Decision size {size!r}; known: "
                     f"{', '.join(SIZES)} or the HF id")


def size_home(size) -> str:
    """Snapshot directory for one size: <INTERN_DECISION_HOME or
    C:\\src>/Intern-Decision-<size>, resolved absolute."""
    size = normalize_size(size)
    base = os.environ.get("INTERN_DECISION_HOME", r"C:\src")
    return os.path.abspath(os.path.join(base, f"Intern-Decision-{size}"))


def load_engine(size=None, device: str = "cpu", dtype: str = "float32"):
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
            "Intern-Decision-0.8B/-2B/-4B snapshots")

    runtime = _RUNTIMES.get(home)
    if runtime is None:
        if home not in sys.path:
            sys.path.insert(0, home)
        stale = sys.modules.pop("inference", None)  # a sibling snapshot's
        try:
            import inference as runtime  # the snapshot's standalone runtime
        finally:
            sys.modules.pop("inference", None)  # keep sys.path hygiene:
            if stale is not None:  # restore whatever "inference" was
                sys.modules["inference"] = stale
        _RUNTIMES[home] = runtime

    return runtime.DecisionEngine(checkpoint=home, device=device, dtype=dtype)
