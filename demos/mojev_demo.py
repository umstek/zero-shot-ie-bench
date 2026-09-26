"""MoJev 0.85B demo + one-shot runner for the web UI - runs inside .venv-von.

MoJev (MoLeMo-Lab/mojev, 0.85B) is a packed one-pass typed-decision scorer:
state, question and every candidate value are packed under one tree
attention mask (Qwen3.5 encoder with fla linear-attention kernels) and one
forward pass returns a calibrated probability per candidate. The scorer
class ships inside the HF repo and loads via trust_remote_code; the request
path is adapted in the vendored engines/mojev_engine/ (MIT). It needs transformers
5.17 for the Qwen3.5 encoder, so it lives in .venv-von, not the main venv.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/mojev_demo.py

Web-UI runner (the app's MoJev tab spawns this like jevk5_demo.py and
talks JSON over stdin/stdout):
    .venv-von/Scripts/python demos/mojev_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...]}
    stdout: {"results": [{"choice": str | None,
                          "probabilities": {label: float},
                          "confidence": float}, ...]}
            or {"error": "Type: message"}
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
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
SENTIMENT_LABELS = ["positive", "negative", "neutral"]
TOPIC_LABELS = ["technology", "business", "sports", "politics"]
QUESTION = {"sentiment": "What is the overall sentiment of this text?",
            "topic": "Which topic category does this text belong to?"}


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


def decide_one(score, text: str, task: str, labels: list[str]) -> dict:
    """One text, one choice field: packed candidates, one pass."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to?")
    choice, probs = score(text, task, question, labels)
    return {
        "choice": choice,
        "probabilities": {label: round(prob, 4)
                          for label, prob in zip(labels, probs)},
        "confidence": round(max(probs), 4) if probs else None,
    }


# ------------------------------------------------------------------ runner
def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        from engines.mojev_engine import load_engine

        score, _ = load_engine("cpu")
        results = [decide_one(score, text, payload["task"],
                              payload["labels"])
                   for text in payload["texts"]]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading MoLeMo-Lab/mojev (0.85B Qwen3.5 encoder via "
          "trust_remote_code; first run downloads the checkpoint)...")
    from engines.mojev_engine import load_engine

    t0 = time.perf_counter()
    score, _ = load_engine("cpu")
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    # ---------------------------------------------------------------- 1
    banner("1. One packed pass - every candidate scored at once")
    text = "This laptop has amazing performance but terrible battery life!"
    print(f'  text: "{text}"')
    row = timed(decide_one, score, text, "sentiment", SENTIMENT_LABELS)
    show(row)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one call each")
    for text in SHARED_TEXTS:
        row = timed(decide_one, score, text, "sentiment", SENTIMENT_LABELS)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    # ---------------------------------------------------------------- 3
    banner("3. Topic - four packed candidates")
    text = "The striker scored twice in the final minutes of the match."
    print(f'  text: "{text}"')
    row = timed(decide_one, score, text, "topic", TOPIC_LABELS)
    show(row["probabilities"])

    print("\nDone. Same texts through the other decision engines:  "
          ".venv/Scripts/python demos/certo_demo.py / "
          ".venv-von/Scripts/python demos/jevk5_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
    else:
        tour()


if __name__ == "__main__":
    main()
