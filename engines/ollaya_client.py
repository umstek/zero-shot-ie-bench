"""Minimal Python client for Ollaya, the local decision-model server.

Ollaya (https://ollaya.dev, runtime Apache-2.0) serves open decision models
over TypeSafe's System One wire format at http://127.0.0.1:11435/v1/systemone
(the same contract engines/jev_client.py implements), with the ``model``
field selecting the checkpoint. The official TypeSafe SDK works against it
unchanged; here a plain JevClient is pointed at the local URL - no API key,
no third-party deps (the JevClient is urllib-only, and Ollaya bills nothing).

Models benchmarked through this client (weights are the authors' own Hugging
Face files, pinned to a commit and sha256-verified by Ollaya, which only
hosts the small ONNX/llama.cpp runner pieces):

  ``nli`` / ``nli:modernbert-large``
      MoritzLaurer zero-shot NLI classifiers (DeBERTa-v3-large 435M /
      ModernBERT-large 396M): every option becomes a hypothesis scored for
      entailment - one sequence pair per option, batched per request.
  ``decision``
      Decision 1.0 by the vLLM Semantic Router contributors: fully
      fine-tuned Qwen3.5-0.8B backbone plus an endpoint head that scores
      each option's last token against the end of the question, one forward
      pass per question, never generates text.
  ``jevk5``
      alibiserikbay JevK5 v0.3, the full 4B Qwen3.5 fine-tune (Q8_0 GGUF on
      llama.cpp; labels A-P read from next-token probabilities). The lite
      encoder build ("JevK5-Lite") runs in-process elsewhere in this repo.
  ``winnow:e4b``
      EldanRing Winnow, fine-tuned from Gemma 4 E4B IT (Q8_0 GGUF on
      llama.cpp); Ollaya builds the author's prompt and reads the option
      letters' logits.

The server must be running before any call: ``ollaya serve`` (CLI from
https://ollaya.dev/download; default port 11435). OLLAYA_BASE_URL overrides
the host:port (a full .../v1/systemone URL wins as-is, so tests can point it
anywhere).

Request-shape gotcha (verified against 0.7.2): the decision layers build
their premise from the ``state``, so the text to classify must BE the state
and ``instructions`` stays the bare question. Embedding the text in the
instructions - the shape the local kev/decider/OpenThai servers take - is
silently dropped by some models (nli:modernbert-large then returns
text-independent probabilities). Choice criteria should carry label
descriptions: the NLI pair encoder builds its hypothesis from them, and
bare-label criteria measurably hurt it.
"""

from __future__ import annotations

import os

# benchmark system name -> Ollaya model tag (served checkpoints)
MODELS = {
    "nli deberta-v3-large (Ollaya)": "nli",
    "nli modernbert-large (Ollaya)": "nli:modernbert-large",
    "decision 0.75B (Ollaya)": "decision",
    "jevk5 4B (Ollaya)": "jevk5",
    "winnow e4b (Ollaya)": "winnow:e4b",
}


def base_url() -> str:
    """The local Ollaya /v1/systemone endpoint."""
    raw = os.environ.get("OLLAYA_BASE_URL",
                         "http://127.0.0.1:11435").rstrip("/")
    return raw if raw.endswith("/v1/systemone") else raw + "/v1/systemone"


def systemone(model: str):
    """A JevClient pointed at the local Ollaya server (no key, free)."""
    from engines.jev_client import JevClient

    return JevClient(base_url=base_url(), model=model)
