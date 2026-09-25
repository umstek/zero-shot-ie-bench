"""JevK5-Lite demo + one-shot runner for the web UI - runs inside .venv-von.

JevK5-Lite (alibiserikbay/JevK5-Lite, 437M) is the lite, CPU-sized build of
JevK5 - which ranks #3 on JevBench - fine-tuned by the jevk5 project: a
DeBERTa-v3-large label-head encoder that reads a text and any number of
label sets ("heads") in one forward pass and returns a calibrated
probability per label (jevk5 package, transformers 5.17). Like von it needs
transformers 5, so it lives in .venv-von, not the main venv.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python jevk5_demo.py

Web-UI runner (the app's JevK5-Lite tab spawns this like von_demo.py and
talks JSON over stdin/stdout):
    .venv-von/Scripts/python jevk5_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...]}
    stdout: {"results": [{"choice": str | None,
                          "probabilities": {label: float},
                          "confidence": float}, ...]}
            or {"error": "Type: message"}
"""

from __future__ import annotations

import json
import sys
import time

MODEL_ID = "alibiserikbay/JevK5-Lite"

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


def load_lite():
    from jevk5 import JevK5Lite

    t0 = time.perf_counter()
    lite = JevK5Lite.from_pretrained(MODEL_ID)
    # stderr: in --serve mode stdout must stay pure JSON for the web UI
    print(f"Loaded in {time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return lite


def classify_one(lite, text: str, task: str, labels: list[str]) -> dict:
    """One text, one head: top label plus the head's calibrated
    probabilities (softmax within a single-label head)."""
    head = lite.classify(text, {task: labels})[task]
    probabilities = head.get("probabilities") or {}
    return {
        "choice": head["labels"][0] if head.get("labels") else None,
        "probabilities": probabilities,
        "confidence": round(max(probabilities.values()), 4)
        if probabilities else None,
    }


# ------------------------------------------------------------------ runner
def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        lite = load_lite()
        results = [classify_one(lite, text, payload["task"],
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

    print(f"Loading {MODEL_ID} (437M DeBERTa-v3-large label-head encoder; "
          "first run downloads the checkpoint)...")
    lite = load_lite()

    # ---------------------------------------------------------------- 1
    banner("1. Several heads in one encoder pass - support ticket")
    text = ("I was charged twice for the same order, please refund one "
            "of them.")
    print(f'  text: "{text}"')
    out = timed(
        lite.classify,
        text,
        {"intent": ["refund_request", "order_status", "cancel_subscription"],
         "areas": {"labels": ["billing", "shipping", "account"],
                   "multi_label": True}},
    )
    show(out)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one classify call each")
    for text in SHARED_TEXTS:
        row = timed(classify_one, lite, text, "sentiment", SENTIMENT_LABELS)
        print(f"  {str(row['choice']):<8} conf {row['confidence']}  {text}")

    # ---------------------------------------------------------------- 3
    banner("3. Raw answer shape (softmax within a single-label head)")
    show(lite.classify(SHARED_TEXTS[0], {"sentiment": SENTIMENT_LABELS}))

    print("\nDone. Same texts through Jev/Laya/von:  python demo_jev.py / "
          "demo_laya.py / .venv-von/Scripts/python von_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
    else:
        tour()


if __name__ == "__main__":
    main()
