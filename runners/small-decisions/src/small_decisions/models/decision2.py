"""vLLM Semantic Router's Decision 2.0 family (Kai 0.6B / Eos 0.8B / Sol 2B),
loaded in-process from local snapshots.

Ported from legacy/engines/decision2_client.py: nothing is pip-installed —
the runtime ships inside each HF snapshot (the decision2/ package plus
backbone/ weights and decision_head.safetensors) and load() sys.path-inserts
the snapshot and imports decision2.api directly. from_pretrained
SHA-256-verifies every packaged byte against MODEL_MANIFEST.json before
anything loads. One runtime per process: loading a second size pops the
first snapshot's decision2 modules first (all three ship the same package
name). The vendored checkpoint fingerprint keys backbone paths with
backslashes on Windows, so load() re-binds checkpoint_fingerprint with
POSIX separators (_patch_windows_fingerprint — the manifest's own byte
verification passes untouched).

The vendored runtime imports Qwen3_5ForConditionalGeneration and
transformers.integrations.sdpa_attention — transformers 5.x code, which is
why this island pins 5.17.0 (the stack the catalog deployment notes and the
legacy client validated: ".venv-von").

Homes: DECISION2_HOME (a prefix directory; each size's snapshot sits at
<prefix>/Decision-2.0-<Name>) or the default C:\src base.

Catalog: vllm-sr/decision-2.0-{kai,eos,sol}, versions
decision-2.0-{kai-0.6b,eos-0.8b,sol-2b}@unversioned, deployments
*.local (runtime torch/fp32/cpu).
r"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from . import _systemone

# catalog model id -> benchmark/demo shorthand (legacy SIZES keys)
MODEL_IDS = {
    "vllm-sr/decision-2.0-kai": "kai-0.6b",
    "vllm-sr/decision-2.0-eos": "eos-0.8b",
    "vllm-sr/decision-2.0-sol": "sol-2b",
}

# benchmark/demo shorthand -> snapshot directory name under the home prefix
SIZES = {
    "kai-0.6b": "Decision-2.0-Kai-0.6B",
    "eos-0.8b": "Decision-2.0-Eos-0.8B",
    "sol-2b": "Decision-2.0-Sol-2B",
}
DEFAULT_SIZE = "kai-0.6b"

# Transcribed from catalog/models/vllm-sr/decision-2.0-*.yaml, deployment
# *.local (all three share the runtime shape).
RUNTIME = {"engine": "torch", "dtype": "fp32", "quant": None, "device": "cpu"}

ENTRIES: list[dict[str, Any]] = [
    {"model_id": model_id, "version": f"decision-2.0-{short}@unversioned", "runtime": dict(RUNTIME)}
    for model_id, short in (
        ("vllm-sr/decision-2.0-kai", "kai-0.6b"),
        ("vllm-sr/decision-2.0-eos", "eos-0.8b"),
        ("vllm-sr/decision-2.0-sol", "sol-2b"),
    )
]

_ENGINE: dict[str, Any] = {}
_LOCK = threading.Lock()


def normalize_size(size: Any) -> str:
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
    raise ValueError(f"unknown Decision 2.0 size {size!r} (one of {', '.join(SIZES)})")


def model_dir(size: Any = DEFAULT_SIZE) -> str:
    """Snapshot directory: DECISION2_HOME prefix, default C:\\src."""
    base = os.environ.get("DECISION2_HOME", r"C:\src")
    return os.path.join(base, SIZES[normalize_size(size)])


def _ensure_runtime_on_path(size: str) -> None:
    """Put the snapshot on sys.path so `import decision2` resolves
    (idempotent). Unconditional insert: a FAILED earlier load could
    leave a stale path entry, and a "not already present" guard would
    then import the wrong snapshot's package — remove every existing
    occurrence, then index 0 always wins."""
    size = normalize_size(size)
    home = model_dir(size)
    if not os.path.isdir(os.path.join(home, "decision2")):
        raise FileNotFoundError(
            f"no {SIZES[size]} snapshot at {home} — pull it with "
            f"`hf download vllm-sr/{SIZES[size]} --local-dir {home}` "
            "(or point DECISION2_HOME at the prefix directory)"
        )
    while home in sys.path:
        sys.path.remove(home)
    sys.path.insert(0, home)


def available(size: Any = DEFAULT_SIZE) -> bool:
    """The snapshot (its decision2/ runtime package) is locally present."""
    return os.path.isdir(os.path.join(model_dir(size), "decision2"))


def _patch_windows_fingerprint() -> None:
    """Make the vendored checkpoint fingerprint agree with the published
    manifests on Windows.

    checkpoint_fingerprint() keys its hash dict by str(Path.relative_to),
    which yields backslash separators here; canonical() then JSON-escapes
    them, so the combined identity digest can never match
    identity.model_sha256 in MODEL_MANIFEST.json — measured on Kai: every
    fingerprint file verifies individually (from_pretrained's packaged-byte
    pass runs first and passes), the backslash keying is the only
    difference, and re-keying with forward slashes reproduces the published
    digest exactly. QwenDecision.load imports the function from
    decision2._vendor.dev2model.infer at call time, so re-binding the
    module attribute redirects the check without touching the snapshot.
    On a POSIX host (or a fixed upstream) the keys are already POSIX and
    the wrapper is a pass-through."""
    import decision2._vendor.dev2model.infer as infer  # pyrefly: ignore[missing-import] — snapshot-resolved

    original = infer.checkpoint_fingerprint

    def posix_fingerprint(path: Any, source_path: Any = None) -> dict[str, Any]:
        identity = original(path, source_path)
        posix_keys = {key.replace("\\", "/"): value for key, value in identity["files_sha256"].items()}
        if posix_keys == identity["files_sha256"]:
            return identity
        import hashlib

        from decision2._vendor.dev2model.data import canonical  # pyrefly: ignore[missing-import]

        return {
            "model_sha256": hashlib.sha256(canonical(posix_keys).encode("utf-8")).hexdigest(),
            "files_sha256": posix_keys,
        }

    infer.checkpoint_fingerprint = posix_fingerprint


def _load(size: str) -> Any:
    """Decision2Engine over the local snapshot (fp32 on the CPU).

    Heavy imports stay inside (offline tests import this module freely).
    The load re-verifies every packaged byte against the manifest —
    expect several seconds of SHA-256 hashing before the backbone loads
    (Kai ~1.5 GB, Sol ~4.8 GB). A load after another size (or a different
    DECISION2_HOME) must not reuse the first snapshot's modules — pop the
    whole package so the fresh path entry wins."""
    size = normalize_size(size)
    _ensure_runtime_on_path(size)
    for stale in [name for name in sys.modules if name == "decision2" or name.startswith("decision2.")]:
        del sys.modules[stale]
    # pyrefly: ignore[missing-import] — the snapshot's vendored runtime, sys.path-resolved
    from decision2.api import Decision2  # noqa: PLC0415 — the heavy stack, lazy

    _patch_windows_fingerprint()
    return Decision2.from_pretrained(model_dir(size), device="cpu")


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
