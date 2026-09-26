"""Certo 421M demo - runs in the MAIN venv.

altslate/certo-decision-model (421M) is a calibrated non-generative decision
model: a ModernBERT-large backbone with a per-option query/scoring head that
encodes the state once and scores every runtime option from its own text
description in one forward pass - no text generation, nothing to parse.
Inference goes through the vendored engines/certo_engine/ (MIT; card documents no
PyPI package), so only huggingface_hub + transformers are needed.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv/Scripts/python demos/certo_demo.py
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import sys
import time

SHARED_TEXTS = [
    "The food was cold and the waiter was rude.",
    "This is the best laptop I have ever owned.",
    "The meeting is scheduled for 3 PM in the main conference room.",
    "The flight was delayed for six hours with no explanation.",
    "She was thrilled with her exam results.",
    "Water boils at 100 degrees Celsius at sea level.",
]
SENTIMENT_OPTIONS = [
    {"id": "positive",
     "description": "Text expresses a clearly positive attitude"},
    {"id": "negative",
     "description": "Text expresses a clearly negative attitude"},
    {"id": "neutral",
     "description": "Factual or mixed text without a clear attitude"},
]
TOPIC_OPTIONS = [
    {"id": "technology",
     "description": "Software, hardware, AI, gadgets, engineering"},
    {"id": "business",
     "description": "Companies, markets, revenue, deals, management"},
    {"id": "sports", "description": "Athletes, matches, teams, tournaments"},
    {"id": "politics",
     "description": "Government, elections, policy, legislation"},
]


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    import json

    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from huggingface_hub import snapshot_download

    from engines.certo_engine import DecisionModel

    print("Loading altslate/certo-decision-model (421M; first run downloads "
          "the checkpoint)...")
    t0 = time.perf_counter()
    model = DecisionModel.load(
        snapshot_download("altslate/certo-decision-model"), device="cpu")
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    # ---------------------------------------------------------------- 1
    banner("1. One decide() call - calibrated probabilities over runtime "
           "options")
    text = "I was charged twice for the same order, please refund one of " \
           "them."
    options = [
        {"id": "billing", "description": "charges and payments"},
        {"id": "tech", "description": "app problems"},
        {"id": "shipping", "description": "delivery and logistics"},
    ]
    print(f'  state: "{text}"')
    print("  options: " + ", ".join(o["id"] for o in options))
    res = timed(model.decide, text, options)
    show(res)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, description-scored "
           "options")
    for text in SHARED_TEXTS:
        res = timed(model.decide, text, SENTIMENT_OPTIONS)
        probs = " ".join(f"{k}={v:.3f}" for k, v in res["probs"].items())
        print(f'  {res["top"]:<8} {probs}  "{text[:52]}"')

    # ---------------------------------------------------------------- 3
    banner("3. Topic - same mechanism, four options")
    for text in ("The striker scored twice in the final minutes of the "
                 "match.",
                 "The senate passed the budget bill after a late-night "
                 "session."):
        res = timed(model.decide, text, TOPIC_OPTIONS)
        probs = " ".join(f"{k}={v:.3f}" for k, v in res["probs"].items())
        print(f'  {res["top"]:<10} {probs}  "{text[:44]}"')

    print("\nDone. Same texts through the rerankers and the packed scorer:  "
          ".venv/Scripts/python demos/reranker_demo.py / "
          ".venv-von/Scripts/python demos/mojev_demo.py\n")


if __name__ == "__main__":
    main()
