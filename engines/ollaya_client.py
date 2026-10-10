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
  ``decima`` / ``decima:small``
      A. M. Madani's Decima (2026-10-05): multilingual late-interaction
      scorers - the state and each option encode separately, every option
      reads the state, so option order never changes the answer; ordinal
      head for scores. base 2.0 = 321M mmBERT (100+ languages, 512 ctx);
      small 1.1 = 122M multilingual-e5-small, the fastest model Ollaya
      runs on a CPU. Needs the 0.11+ daemon (this repo runs 0.12.1).
  ``snap``
      logitlab's snap1-2b: MiniCPM5-2B fine-tuned on emnlmn's snap-engine
      prompt (Q8_0 GGUF on llama.cpp, 8k ctx, English+Italian); one option
      letter per option read from next-token probabilities, no
      calibration. Needs the 0.10+ daemon.
  ``arbiter``
      Codekins' Arbiter (Zyot Lab): Gemma 3 4B IT + LoRA with a fixed
      24-slot head (4.3B Q8_0); noul, choice of up to 16 options and
      scores of exactly 6 levels, multilingual.
  ``credence``
      Txoka's Credence v1: MiCA refinements of Winnow-E4B (7.5B Q8_0) -
      one checkpoint trades public accuracy, one calibration; the vision
      tags add Winnow's unchanged projector. Neither beats the original
      Winnow on every metric; benched for the family tree.

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
    "decima 321M (Ollaya)": "decima",
    "decima small 122M (Ollaya)": "decima:small",
    "snap 2B (Ollaya)": "snap",
    "arbiter 4B (Ollaya)": "arbiter",
    "credence 7.5B (Ollaya)": "credence",
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
