"""In-process loader for vLLM Semantic Router's Decision 3.0 family
(local snapshots).

Decision 3.0 (Apache 2.0, "Towards Open Multimodal Foundation Decision
Models", released 2026-10-10) is the third generation of the line behind
the Ollaya-served `decision` model: multimodal (text + images + videos)
Qwen3.5-backbone checkpoints that answer typed System One questions —
choice (2-255 keyed, described options), noul (a true/false probability)
and score (2-10 ordered levels) — with a code-readout head: the prompt
lists every option under a single-token answer code and a 255-way
readout scores those codes at the last prompt position, one question per
forward pass, batched `batch_size` prompts to a padded batch. No
generation, no truncation. Benched sizes: d3-edge (594M, Qwen3.5-0.8B
backbone + a 0.10B vision encoder), d3-lite (853M) and d3-nano (2.2B).
The collection's d3-mini 4.5B / d3-flash 8.4B / d3 26.1B stay out —
download size and per-question cost on this 32 GB CPU box (README, Noted
not benchmarked), the same line the Decision 2.0 Nox/Lux/Vega trio drew.

The packages are self-verifying: D3.from_pretrained checks every file
against MODEL_MANIFEST.json (verify="fast", the runtime's default,
SHA-256-hashes every packaged file up to 64 MiB — all of the runtime
code, configs and readout — and size-checks the weight shards;
verify="full" hashes every byte). The manifest keys are POSIX-relative
names, so unlike Decision 2.0 there is no Windows separator wart to
patch. The backbone loads bf16 with SDPA (the runtime's hard-coded
choice; fp32 readout + softmax), the Gated DeltaNet layers run
transformers' PyTorch reference on the CPU (the fla kernels are GPU-only
— the runtime refuses to load on a CPU when transformers binds them;
.venv-von's transformers 5.17.0 binds the reference even with fla
installed, measured), and the ROCm HIP-graph fast path disables itself
with a reason.

Nothing is pip-installed: the runtime ships inside each HF snapshot
(d3_runtime.py + d3_format/d3_fast/d3_kernels + modeling/pipeline for
the trust_remote_code route, backbone weights and readout.safetensors).
load_engine() sys.path-inserts the snapshot and imports d3_runtime
directly — the same house pattern as decision2/k2type/intern-decision,
and the exact code path the card's own Decision Index engine
(d3_engine.py) drives. One runtime per process: loading a second size
pops the first snapshot's d3_* modules first, so each size runs its own
packaged code (all of them ship the same module names).

Homes: D3_HOME (a prefix directory; each size's snapshot sits at
<prefix>/d3-<size>) or the default C:\\src base — engines resolve to
C:\\src\\d3-edge / d3-lite / d3-nano (the repo's own D: drive is full,
the same reason Decision 2.0, K2-Type and Intern-Decision live on C:).

predict(state, questions) is the card's system_one contract minus HTTP
plumbing; state is text or JSON, questions map ids to
{"type", "instructions", "criteria"}:

    engine = d3_client.load_engine("edge")
    engine.predict("some text", {"q": {"type": "choice",
                                       "instructions": "...",
                                       "criteria": {label: desc}}})
    -> {"answers": {"q": {"choice": ..., "probabilities": {...},
                          "confidence": ...}},
        "model": "d3-edge", "input_tokens": 123}

Answer shapes come from the vendored product_answer: choice ->
{choice, probabilities, confidence (a normalized-entropy measure)};
noul -> {noul: p_true} (no confidence field); score -> {score:
expected level, probabilities, legend, confidence}. Malformed or
over-budget questions do not raise — they answer {"type": ...,
"error": "invalid_question" | "max_length_exceeded"} in place, so
callers check for "error" before reading fields.
"""

from __future__ import annotations

import os
import sys

# benchmark/demo shorthand -> snapshot directory name under the home prefix
SIZES = {
    "edge": "d3-edge",
    "lite": "d3-lite",
    "nano": "d3-nano",
}
DEFAULT_SIZE = "edge"

# the flat runtime modules a snapshot ships; popped from sys.modules
# before a load so a second size imports its own packaged code
_SNAPSHOT_MODULES = ("d3_engine", "d3_fast", "d3_format", "d3_kernels",
                     "d3_runtime", "d3_server")


def normalize_size(size) -> str:
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
    raise ValueError(f"unknown Decision 3.0 size {size!r} "
                     f"(one of {', '.join(SIZES)})")


def model_dir(size=DEFAULT_SIZE) -> str:
    """Snapshot directory: D3_HOME prefix, default C:\\src."""
    base = os.environ.get("D3_HOME", r"C:\src")
    return os.path.join(base, SIZES[normalize_size(size)])


def _ensure_runtime_on_path(size=DEFAULT_SIZE) -> str:
    """Put the snapshot on sys.path so `import d3_runtime` resolves;
    returns the snapshot path (idempotent). Unconditional insert: a
    FAILED earlier load could leave a stale path entry, and a "not
    already present" guard would then import the wrong snapshot's
    modules — remove every existing occurrence, then index 0 always
    wins."""
    size = normalize_size(size)
    home = model_dir(size)
    if not os.path.isfile(os.path.join(home, "d3_runtime.py")):
        raise FileNotFoundError(
            f"no {SIZES[size]} snapshot at {home} — pull it with "
            f"`hf download vllm-sr/{SIZES[size]} --local-dir {home}` "
            "(or point D3_HOME at the prefix directory)")
    while home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)
    return home


class D3Engine:
    """One loaded package; predict() answers every question of a request
    in padded batches over the shared state, one forward per question."""

    def __init__(self, handle, name: str, max_input_tokens):
        self._handle = handle
        self.name = name
        self.max_input_tokens = max_input_tokens

    def predict(self, state, questions: dict) -> dict:
        out = self._handle.system_one(state=state, questions=questions)
        return {"answers": out["answers"], "model": out["model"],
                "input_tokens": out["usage"]["input_tokens"]}


def load_engine(size=DEFAULT_SIZE, device: str = "cpu",
                verify: str = "fast") -> D3Engine:
    """D3Engine over the local snapshot (bf16 backbone, fp32 readout on
    the CPU). Heavy imports stay inside (offline tests import this
    module freely). The load verifies the package against
    MODEL_MANIFEST.json first ("fast": SHA-256 for every file up to
    64 MiB, size for the weight shards; "full" hashes every byte), then
    loads the backbone — expect the manifest pass plus the weight read
    (edge ~1.2 GB, nano ~4.4 GB) before the first answer."""
    size = normalize_size(size)
    home = _ensure_runtime_on_path(size)
    # a load after another size (or a different D3_HOME) must not reuse
    # the first snapshot's modules — pop the runtime set so the fresh
    # path entry wins
    for stale in [name for name in sys.modules
                  if name in _SNAPSHOT_MODULES]:
        del sys.modules[stale]
    from d3_runtime import D3

    handle = D3.from_pretrained(home, device=device, verify=verify)
    return D3Engine(handle, handle.model_name, handle.max_length)
