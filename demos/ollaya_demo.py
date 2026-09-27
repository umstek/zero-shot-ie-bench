"""Ollaya demo: the Ollaya-served decision models, one tour.

Ollaya (https://ollaya.dev, Apache-2.0 runtime) serves open decision models
behind TypeSafe's System One wire format on a local daemon (default
127.0.0.1:11435). This repo benchmarks five of them through
engines/ollaya_client.py:

  nli / nli:modernbert-large  MoritzLaurer zero-shot NLI classifiers
                              (DeBERTa-v3-large 435M / ModernBERT-large
                              396M): every option becomes a hypothesis
                              scored for entailment.
  decision                    Decision 1.0 by the vLLM Semantic Router
                              contributors: Qwen3.5-0.8B backbone + endpoint
                              head, one forward pass per question.
  jevk5                       the full JevK5 4B (alibiserikbay, Q8_0 GGUF on
                              llama.cpp; the lite encoder build has its own
                              demo_jevk5 in .venv-von).
  winnow:e4b                  EldanRing Winnow, fine-tuned from Gemma 4 E4B
                              (Q8_0 GGUF on llama.cpp).

Prerequisite: the daemon running (`ollaya serve`, CLI from
https://ollaya.dev/download) and each model pulled once (`ollaya pull nli`,
`ollaya pull decision`, ...). Missing models are skipped with a note.

Sample texts are shared with the other demos so outputs compare directly.

Tour:
    .venv/Scripts/python demos/ollaya_demo.py            # all five models
    .venv/Scripts/python demos/ollaya_demo.py --models nli,decision
"""

from __future__ import annotations

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.abspath(__file__))))

import json
import sys
import time

from engines.jev_client import choice, noul, score
from engines.ollaya_client import MODELS, systemone

SHARED_TEXTS = [
    "The food was cold and the waiter was rude.",
    "This is the best laptop I have ever owned.",
    "The meeting is scheduled for 3 PM in the main conference room.",
    "The flight was delayed for six hours with no explanation.",
    "She was thrilled with her exam results.",
    "Water boils at 100 degrees Celsius at sea level.",
]
SENTIMENT = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual text without a clear attitude",
}

# the tour's model order (subset via --models)
TOUR = ["nli deberta-v3-large (Ollaya)", "nli modernbert-large (Ollaya)",
        "decision 0.75B (Ollaya)", "jevk5 4B (Ollaya)",
        "winnow e4b (Ollaya)"]


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"  ({time.perf_counter() - t0:.2f}s on this machine)")
    return result


def ask(client, state, questions, timeout=600):
    """One System One request; the state IS the text (Ollaya's decision
    layers build their premise from the state - see the client module)."""
    return client.ask(state, questions, timeout=timeout)


def tour(names: list[str]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    clients = {}
    for name in names:
        client = systemone(MODELS[name])
        try:  # a missing pull surfaces on first use - skip, don't die
            ask(client, "good", {"w": choice(
                "What is the sentiment of this text?", SENTIMENT)})
            clients[name] = client
        except Exception as exc:
            print(f"skipping {name} ({MODELS[name]}): {exc}")

    if not clients:
        raise SystemExit("No Ollaya models reachable - is `ollaya serve` "
                         "running, and are the models pulled?")

    # ---------------------------------------------------------------- 1
    banner("1. Typed questions about one state, in a single request")
    ticket = ("I was charged twice for my subscription this month and "
              "want a refund.")
    questions = {
        "department": choice("Which team should handle this?", {
            "billing": "Payments, invoices and refunds",
            "technical": "Bugs, errors and outages",
            "account": "Login, profile and settings"}),
        "refund": noul("Is the customer asking for a refund?"),
    }
    print(f'  state: "{ticket}"')
    for name, client in clients.items():
        print(f"\n  --- {name} ({MODELS[name]}) ---")
        out = timed(ask, client, ticket, questions)
        show(out["answers"])

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one request per text")
    for name, client in clients.items():
        print(f"\n  --- {name} ---")
        for text in SHARED_TEXTS:
            out = ask(client, text, {"q": choice(
                "What is the overall sentiment of this text?", SENTIMENT)})
            a = out["answers"]["q"]
            print(f"  {a.get('choice', '?'):<8} "
                  f"conf {a.get('confidence', 0):<7} {text}")

    # ---------------------------------------------------------------- 3
    banner("3. Score rubric - urgency of the ticket on three levels")
    for name, client in clients.items():
        print(f"\n  --- {name} ---")
        try:
            out = ask(client, ticket, {"urgency": score(
                "How urgent is this issue?", [
                    "could wait a few days",
                    "should be fixed soon",
                    "needs immediate action"])})
            show(out["answers"]["urgency"])
        except Exception as exc:
            print(f"  (score questions rejected: {exc})")

    print("\nDone. Same texts through Jev/Laya/kev:  "
          "python demos/demo_jev.py / demos/demo_laya.py; local kev "
          "servers are covered by the benchmark tabs.\n")


def main() -> None:
    names = TOUR
    for arg in sys.argv[1:]:
        if arg.startswith("--models="):
            wanted = arg.split("=", 1)[1].split(",")
            names = [n for n in TOUR if any(w in n for w in wanted)]
        elif arg == "--models" and sys.argv.index(arg) + 1 < len(sys.argv):
            wanted = sys.argv[sys.argv.index(arg) + 1].split(",")
            names = [n for n in TOUR if any(w in n for w in wanted)]
    if not names:
        raise SystemExit(f"no tour models matched; known: {TOUR}")
    tour(names)


if __name__ == "__main__":
    main()
