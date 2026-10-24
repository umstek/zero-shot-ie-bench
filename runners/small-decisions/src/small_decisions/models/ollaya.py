"""The Ollaya daemon's decision models, proxied over HTTP.

Ported from legacy/engines/ollaya_client.py (which pointed a JevClient at
the daemon): Ollaya (<https://ollaya.dev>, runtime Apache-2.0) serves open
decision models over TypeSafe's System One wire format at
<host>/v1/systemone (the contract engines/jev_client.py speaks), with the
`model` field selecting the checkpoint. This module speaks it directly
with urllib — no API key, no third-party deps. The daemon serves the small
ONNX/llama.cpp runner pieces; the weights are the authors' own HF files,
pinned to a commit and sha256-verified by Ollaya.

The server must be running before any call: `ollaya serve` (CLI from
<https://ollaya.dev/download>; default port 11435; this repo runs 0.12.1).
OLLAYA_BASE_URL overrides the host:port (a full .../v1/systemone URL wins
as-is, so tests can point it anywhere — the test suite does exactly that
against a stubbed HTTP server).

Served here (tag -> catalog model):

  ``decision``              vllm-sr/decision — Decision 1.0 by the vLLM
                            Semantic Router contributors: fully fine-tuned
                            Qwen3.5-0.8B backbone plus an endpoint head that
                            scores each option's last token against the end
                            of the question, one forward pass per question.
  ``nli`` / ``nli:modernbert-large``
                            MoritzLaurer zero-shot NLI classifiers
                            (DeBERTa-v3-large 435M / ModernBERT-large 396M):
                            every option becomes a hypothesis scored for
                            entailment — one sequence pair per option.
  ``jevk5``                 alibiserikbay JevK5 v0.3, the full 4B Qwen3.5
                            fine-tune (Q8_0 GGUF on llama.cpp).
  ``winnow:e4b``            EldanRing Winnow, fine-tuned from Gemma 4 E4B IT.
  ``decima`` / ``decima:small``
                            A. M. Madani's Decima: multilingual
                            late-interaction scorers (needs the 0.11+
                            daemon). base 2.0 = 321M mmBERT;
                            small 1.1 = 122M multilingual-e5-small.
  ``snap``                  logitlab's snap1-2b (MiniCPM5-2B, Q8_0 GGUF;
                            needs the 0.10+ daemon).
  ``arbiter``               Codekins' Arbiter (Zyot Lab): Gemma 3 4B IT +
                            LoRA with a fixed 24-slot head.
  ``credence``              Txoka's Credence v1: MiCA refinements of
                            Winnow-E4B (7.5B Q8_0).

Request-shape gotcha (verified against 0.7.2): the decision layers build
their premise from the ``state``, so the text to classify must BE the state
and ``instructions`` stays the bare question. Embedding the text in the
instructions is silently dropped by some models. Choice criteria should
carry label descriptions: the NLI pair encoder builds its hypothesis from
them.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from . import _systemone

# catalog model id -> Ollaya model tag (the daemon's served checkpoints)
TAGS = {
    "vllm-sr/decision": "decision",
    "moritzlaurer/nli-deberta-v3-large": "nli",
    "moritzlaurer/nli-modernbert-large": "nli:modernbert-large",
    "alibiserikbay/jevk5-4b": "jevk5",
    "eldanring/winnow-e4b": "winnow:e4b",
    "madani/decima": "decima",
    "madani/decima-small": "decima:small",
    "logitlab/snap": "snap",
    "codekins/arbiter": "arbiter",
    "txoka/credence": "credence",
}

# Transcribed from the catalog yamls' deployments (*.ollaya): the runtime
# dict records what the daemon runs (engine null = Ollaya's ONNX runners;
# llama.cpp = Q8_0 GGUF), not what this process loads.
ENTRIES: list[dict[str, Any]] = [
    {"model_id": "vllm-sr/decision", "version": "decision-1.0@unversioned",
     "runtime": {"engine": None, "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "moritzlaurer/nli-deberta-v3-large", "version": "deberta-v3-large@unversioned",
     "runtime": {"engine": None, "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "moritzlaurer/nli-modernbert-large", "version": "modernbert-large@unversioned",
     "runtime": {"engine": None, "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "alibiserikbay/jevk5-4b", "version": "jevk5-4b@unversioned",
     "runtime": {"engine": "llama.cpp", "dtype": None, "quant": "Q8_0", "device": "cpu"}},
    {"model_id": "eldanring/winnow-e4b", "version": "winnow-e4b@unversioned",
     "runtime": {"engine": "llama.cpp", "dtype": None, "quant": "Q8_0", "device": "cpu"}},
    {"model_id": "madani/decima", "version": "decima-base-2.0@unversioned",
     "runtime": {"engine": None, "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "madani/decima-small", "version": "decima-small-1.1@unversioned",
     "runtime": {"engine": None, "dtype": None, "quant": None, "device": "cpu"}},
    {"model_id": "logitlab/snap", "version": "snap1-2b@unversioned",
     "runtime": {"engine": "llama.cpp", "dtype": None, "quant": "Q8_0", "device": "cpu"}},
    {"model_id": "codekins/arbiter", "version": "arbiter-4b@unversioned",
     "runtime": {"engine": "llama.cpp", "dtype": None, "quant": "Q8_0", "device": "cpu"}},
    {"model_id": "txoka/credence", "version": "credence-v1@unversioned",
     "runtime": {"engine": "llama.cpp", "dtype": None, "quant": "Q8_0", "device": "cpu"}},
]


def base_url() -> str:
    """The local Ollaya /v1/systemone endpoint."""
    raw = os.environ.get("OLLAYA_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
    return raw if raw.endswith("/v1/systemone") else raw + "/v1/systemone"


def systemone(model_tag: str, state: Any, questions: dict[str, Any],
              timeout: int = 120) -> dict[str, Any]:
    """One POST /v1/systemone against the daemon (JevClient.ask, minus the
    key/danger-zone/assets plumbing a local Ollaya never needs)."""
    body = json.dumps({"model": model_tag, "state": state, "questions": questions}).encode("utf-8")
    request = urllib.request.Request(
        base_url(),
        data=body,
        method="POST",
        headers={"content-type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"Ollaya HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict) or "answers" not in payload:
        raise RuntimeError(f"Ollaya response missing answers: {payload!r:.300}")
    payload["_latency_s"] = round(time.perf_counter() - t0, 3)
    return payload


def predict(model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """One decision item: {input, question} -> {prediction, prob}.

    The question goes to the daemon verbatim in System One shape (the
    state/instructions split is load-bearing — see the module docstring's
    request-shape gotcha)."""
    tag = TAGS.get(model_id)
    if tag is None:
        raise KeyError(f"unknown model_id: {model_id!r} (served: {sorted(TAGS)})")
    question = _systemone.normalize_question(payload.get("question"))
    state = _systemone.state_text(payload.get("input"))
    out = systemone(tag, state, {"q": question})
    return _systemone.systemone_result(question["type"], out["answers"]["q"])
