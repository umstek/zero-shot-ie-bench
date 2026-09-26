"""LFM2.5-RLCD 350M demo + one-shot runner for the web UI - .venv-von.

notnotsamuel/LFM2.5-350M-RLCD keeps the LiquidAI/LFM2.5-350M weights and
adds RLCD training; inference runs through its constrained-decoding engine,
vendored in this repo as engines/rlcd_engine/ (engine code MIT; the LFM weights are
under the LFM Open License v1.0). The engine prefills the context once,
branches the attention/convolution state across every field's candidate
values, scores all branches in one batched pass, and assembles the JSON
from the argmax likelihoods. Needs transformers 5.17 + jsonschema, so it
lives in .venv-von, not the main venv.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/lfm_rlcd_demo.py

Web-UI runner (the app's LFM2.5-RLCD tab spawns this like von_demo.py and
talks JSON over stdin/stdout):
    .venv-von/Scripts/python demos/lfm_rlcd_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...]}
    stdout: {"results": [{"choice": str | None,
                          "log_likelihoods": {value: float}}, ...]}
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


def load_engine():
    from engines.rlcd_engine.engine import Engine

    t0 = time.perf_counter()
    engine = Engine(device="cpu", dtype="float32")
    # stderr: in --serve mode stdout must stay pure JSON for the web UI
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def schema_for(task: str, labels: list[str], description: str) -> dict:
    """Closed, flat, one-field schema - the engine's supported subset."""
    return {"type": "object",
            "properties": {task: {"type": "string",
                                  "description": description,
                                  "enum": list(labels)}},
            "required": [task],
            "additionalProperties": False}


def decide_one(engine, text: str, task: str, labels: list[str]) -> dict:
    res = engine.constrained(
        text, schema_for(task, labels, f"The overall {task} of this text"))
    try:
        choice = json.loads(res["text"])[task]
    except (KeyError, ValueError):
        choice = None
    return {"choice": choice,
            "log_likelihoods": {str(row["value"]): row["log_likelihood"]
                                for row in res["scores"].get(task, [])}}


# ------------------------------------------------------------------ runner
def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        engine = load_engine()
        results = [decide_one(engine, text, payload["task"],
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

    print("Loading LiquidAI/LFM2.5-350M through the vendored rlcd_engine "
          "(350M; first run downloads the weights)...")
    engine = load_engine()

    # ---------------------------------------------------------------- 1
    banner("1. One constrained call, full result")
    text = "This laptop has amazing performance but terrible battery life!"
    print(f'  text: "{text}"')
    res = timed(
        engine.constrained,
        text,
        schema_for("sentiment", SENTIMENT_LABELS,
                   "The overall sentiment of this text"),
    )
    print(f'  assembled: {res["text"]}')
    print(f"  ({res['branches']} candidate branches scored in "
          f"{res['forward_calls']} forward calls)")
    show(res["scores"])

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one call each")
    for text in SHARED_TEXTS:
        row = timed(decide_one, engine, text, "sentiment", SENTIMENT_LABELS)
        print(f"  {str(row['choice']):<8} {text}")

    # ---------------------------------------------------------------- 3
    banner("3. Two fields in one call - string enum + boolean")
    schema = {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string",
                          "description": "The overall sentiment of this text",
                          "enum": SENTIMENT_LABELS},
            "is_opinion": {"type": "boolean",
                           "description": "True if the text states a "
                                          "personal opinion"},
        },
        "required": ["sentiment", "is_opinion"],
        "additionalProperties": False,
    }
    for text in ("The food was cold and the waiter was rude.",
                 "Water boils at 100 degrees Celsius at sea level."):
        res = timed(engine.constrained, text, schema)
        print(f'  {res["text"]:<58} "{text[:44]}"')

    print("\nDone. Same texts through Jev/Laya/von:  "
          "python demos/demo_jev.py / demos/demo_laya.py / "
          ".venv-von/Scripts/python demos/von_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
    else:
        tour()


if __name__ == "__main__":
    main()
