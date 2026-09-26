"""nanodiff 350M demo + one-shot runner for the web UI - runs in the MAIN
venv (tiktoken).

pngwn/nanodiff-350m-typed-decisions-lam1 (350M) is a bidirectional diffusion
LM - the only non-autoregressive system in this repo: the answer-letter slot
is the only [MASK]ed token and ONE bidirectional forward scores every
option (softmax restricted to the option-letter token ids); no decoding
loop, no generation. Inference goes through the vendored engines/nanodiff_engine/
(MIT; NanoDiff model class from BY571/nanoDiff plus the release's typed-
decision format). Slow on CPU (~10 s/question) - kept as the diffusion
datapoint.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv/Scripts/python demos/nanodiff_demo.py

Web-UI runner (the app's nanodiff tab spawns this like von_demo.py and
talks JSON over stdin/stdout):
    .venv/Scripts/python demos/nanodiff_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...]}
    stdout: {"results": [{"choice": str | None,
                          "probabilities": {label: float}}, ...]}
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


def _alias_vendored_nanodiff() -> None:
    """The checkpoint pickle stores its config objects under the upstream
    module name 'nanodiff' (BY571/nanoDiff); the vendored copy lives in
    engines.nanodiff_engine.nanodiff. Register it under the expected name so
    torch.load resolves the classes (same objects, no import hack inside
    the vendored engine)."""
    import engines.nanodiff_engine.nanodiff as vendored
    import engines.nanodiff_engine.nanodiff.config as vendored_config

    sys.modules.setdefault("nanodiff", vendored)
    sys.modules.setdefault("nanodiff.config", vendored_config)


def load():
    from engines.nanodiff_engine.runner import load_model

    _alias_vendored_nanodiff()
    t0 = time.perf_counter()
    model, _ = load_model("cpu")
    # stderr: in --serve mode stdout must stay pure JSON for the web UI
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return model


def decide_one(model, text: str, question: str, labels: list[str]) -> dict:
    """One masked bidirectional forward; softmax over option letters."""
    from engines.nanodiff_engine.runner import predict

    choice, probs = predict(model, text, question, labels, "cpu")
    return {"choice": choice, "probabilities": probs}


# ------------------------------------------------------------------ runner
def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        # one question per task word, same phrasing the benchmarks use
        q = ("What is the overall sentiment of this text?"
             if payload["task"] == "sentiment"
             else f"Which {payload['task']} category does this text "
                  "belong to?")
        model = load()
        results = [decide_one(model, text, q, payload["labels"])
                   for text in payload["texts"]]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading pngwn/nanodiff-350m-typed-decisions-lam1 (350M diffusion "
          "LM; first run downloads the checkpoint)...")
    model = load()

    # ---------------------------------------------------------------- 1
    banner("1. One bidirectional forward - every option scored at once")
    text = "This laptop has amazing performance but terrible battery life!"
    print(f'  text: "{text}"')
    row = timed(decide_one, model, text,
                "What is the overall sentiment of this text?",
                SENTIMENT_LABELS)
    show(row)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - a few shared texts (~10 s/question on CPU)")
    for text in SHARED_TEXTS[:3]:
        row = timed(decide_one, model, text,
                    "What is the overall sentiment of this text?",
                    SENTIMENT_LABELS)
        probs = " ".join(f"{k}={v:.2f}" for k, v in row["probabilities"]
                         .items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    # ---------------------------------------------------------------- 3
    banner("3. Topic - four option letters")
    text = "The striker scored twice in the final minutes of the match."
    print(f'  text: "{text}"')
    row = timed(decide_one, model, text,
                "Which topic category does this text belong to?",
                TOPIC_LABELS)
    show(row["probabilities"])

    print("\nDone. Same texts through the autoregressive decision engines:  "
          ".venv/Scripts/python demos/certo_demo.py / "
          ".venv-von/Scripts/python demos/mojev_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
    else:
        tour()


if __name__ == "__main__":
    main()
