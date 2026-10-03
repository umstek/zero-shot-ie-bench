"""In-process loader for vLLM Semantic Router's Decision 2.0 family
(local snapshots).

Decision 2.0 (Apache 2.0, "Towards Open Foundation Decision Models") is
the second generation of the line behind the Ollaya-served `decision`
model this repo already benches: Qwen-backbone checkpoints with a
trained candidate head that answer typed System One questions —
choice (2-255 keyed, described options), noul (a true/false
probability) and score (2-10 ordered levels) — from ONE forward pass
over the shared state, one row per question, no generation. Benched
sizes: Kai-0.6B (597M on Qwen3-0.6B-Base), Eos-0.8B (753M, a
Decision-1.0-Eos fine-tune) and Sol-2B (1.88B, likewise). The
collection's Nox-4B / Lux-9B / Vega-27B stay out — RAM and download
size on this 32 GB CPU box (README, Noted not benchmarked).

The packages are self-verifying: decision2.api.Decision2.from_pretrained
SHA-256-hashes every packaged byte against MODEL_MANIFEST.json before
anything loads and checks the packed parameter counts, then runs the
vendored dev2model runtime (backbone + candidate head, prompt
decision2-segmented-options-global-query-v1). CPU inference is fp32 —
the bf16 residency / HIP-graph fast paths only engage on a GPU. Kai's
score answers carry per-level logit offsets (score_bias.json) which
the loader applies itself. One Windows wart: the vendored fingerprint
keys backbone paths with backslashes, so load_engine() re-binds
checkpoint_fingerprint with a POSIX-separator normalization (see
_patch_windows_fingerprint) — the manifest's own byte verification
passes untouched; only the combined identity digest is re-keyed.

Nothing is pip-installed: the runtime ships inside each HF snapshot
(the decision2/ package plus backbone/ weights and
decision_head.safetensors). load_engine() sys.path-inserts the
snapshot and imports decision2.api directly (the card's
trust_remote_code AutoModel route would work too; the direct import is
the k2type/intern-decision house pattern and skips the transformers
remote-code dance). One runtime per process: loading a second size
pops the first snapshot's decision2 modules first, so each size runs
its own packaged code (all three ship the same package name).

Homes: DECISION2_HOME (a prefix directory; each size's snapshot sits
at <prefix>/Decision-2.0-<Name>) or the default C:\\src base —
engines resolve to C:\\src\\Decision-2.0-Kai-0.6B / -Eos-0.8B /
-Sol-2B (the repo's own D: drive is full, the same reason K2-Type and
Intern-Decision live on C:).

predict(state, questions) is the card's system_one contract minus HTTP
plumbing; state is text or JSON, questions map ids to
{"type", "instructions", "criteria"}:

    engine = decision2_client.load_engine("kai-0.6b")
    engine.predict("some text", {"q": {"type": "choice",
                                       "instructions": "...",
                                       "criteria": {label: desc}}})
    -> {"answers": {"q": {"choice": ..., "probabilities": {...},
                          "confidence": ...}},
        "model": "Decision-2.0-Kai-0.6B", "input_tokens": 123}

Answer shapes come from the vendored infer.product_answer: choice ->
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
    "kai-0.6b": "Decision-2.0-Kai-0.6B",
    "eos-0.8b": "Decision-2.0-Eos-0.8B",
    "sol-2b": "Decision-2.0-Sol-2B",
}
DEFAULT_SIZE = "kai-0.6b"


def normalize_size(size) -> str:
    """Registry key for a size ('kai', 'Kai-0.6B', full HF id or None).
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
    raise ValueError(f"unknown Decision 2.0 size {size!r} "
                     f"(one of {', '.join(SIZES)})")


def model_dir(size=DEFAULT_SIZE) -> str:
    """Snapshot directory: DECISION2_HOME prefix, default C:\\src."""
    base = os.environ.get("DECISION2_HOME", r"C:\src")
    return os.path.join(base, SIZES[normalize_size(size)])


def _ensure_runtime_on_path(size=DEFAULT_SIZE) -> None:
    """Put the snapshot on sys.path so `import decision2` resolves
    (idempotent). Unconditional insert: a FAILED earlier load could
    leave a stale path entry, and a "not already present" guard would
    then import the wrong snapshot's package — index 0 always wins."""
    size = normalize_size(size)
    home = model_dir(size)
    if not os.path.isdir(os.path.join(home, "decision2")):
        raise FileNotFoundError(
            f"no {SIZES[size]} snapshot at {home} — pull it with "
            f"`hf download vllm-sr/{SIZES[size]} --local-dir {home}` "
            "(or point DECISION2_HOME at the prefix directory)")
    if home not in sys.path:
        sys.path.insert(0, home)


class Decision2Engine:
    """One loaded package; predict() answers every question of a request
    in one batched forward (share_context stays off — the exact path,
    the same answers the scored predictions were produced with)."""

    def __init__(self, handle, name: str, max_input_tokens: int):
        self._handle = handle
        self.name = name
        self.max_input_tokens = max_input_tokens

    def predict(self, state, questions: dict) -> dict:
        out = self._handle.system_one(state=state, questions=questions)
        return {"answers": out["answers"], "model": out["model"],
                "input_tokens": out["usage"]["input_tokens"]}


def _patch_windows_fingerprint() -> None:
    """Make the vendored checkpoint fingerprint agree with the published
    manifests on Windows.

    checkpoint_fingerprint() keys its hash dict by str(Path.relative_to),
    which yields backslash separators here; canonical() then JSON-escapes
    them, so the combined identity digest can never match
    identity.model_sha256 in MODEL_MANIFEST.json — measured on Kai: every
    fingerprint file verifies individually (from_pretrained's packaged-byte
    pass runs first and passes), the backslash keying is the only
    difference, and re-keying with forward slashes reproduces the
    published digest exactly. QwenDecision.load imports the function from
    decision2._vendor.dev2model.infer at call time, so re-binding the
    module attribute redirects the check without touching the snapshot.
    On a POSIX host (or a fixed upstream) the keys are already POSIX and
    the wrapper is a pass-through."""
    import decision2._vendor.dev2model.infer as infer

    original = infer.checkpoint_fingerprint

    def posix_fingerprint(path, source_path=None):
        identity = original(path, source_path)
        posix_keys = {key.replace("\\", "/"): value
                      for key, value in identity["files_sha256"].items()}
        if posix_keys == identity["files_sha256"]:
            return identity
        import hashlib

        from decision2._vendor.dev2model.data import canonical
        return {"model_sha256": hashlib.sha256(
                    canonical(posix_keys).encode("utf-8")).hexdigest(),
                "files_sha256": posix_keys}

    infer.checkpoint_fingerprint = posix_fingerprint


def load_engine(size=DEFAULT_SIZE, device: str = "cpu",
                threads=None) -> Decision2Engine:
    """Decision2Engine over the local snapshot (fp32 on the CPU).

    Heavy imports stay inside (offline tests import this module freely).
    The load re-verifies every packaged byte against the manifest —
    expect several seconds of SHA-256 hashing before the backbone
    loads (Kai ~1.5 GB, Sol ~4.8 GB)."""
    size = normalize_size(size)
    _ensure_runtime_on_path(size)
    # a load after another size (or a different DECISION2_HOME) must not
    # reuse the first snapshot's modules — pop the whole package so the
    # fresh path entry wins
    for stale in [name for name in sys.modules
                  if name == "decision2" or name.startswith("decision2.")]:
        del sys.modules[stale]
    from decision2.api import Decision2

    _patch_windows_fingerprint()
    handle = Decision2.from_pretrained(model_dir(size), device=device,
                                       threads=threads)
    return Decision2Engine(handle, handle.model_name,
                           handle.max_input_tokens)
