"""von demo + one-shot runner for the web UI - runs inside .venv-von.

von-1.0 (wfzyx/von-1.0, Apache 2.0) is a 396M ModernBERT with its
trained option-marker head: every question is packed into one sequence
whose [MASK] slots are read in a single forward pass - no generation,
no generate(). It needs transformers 5.x while the app's main venv pins
4.57.6, so everything here executes in .venv-von (see
requirements-von.txt).

The three System One question types map onto the pinned SDK's backend:
choice -> evaluate_choice (argmax over described options), judge ->
evaluate_noul (P(condition true), 0.0-1.0), rate -> evaluate_score
(probability-weighted expectation over an ordered rubric).

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive; --serve stays the default so the web-UI spawn needs
no flag):
    .venv-von/Scripts/python demos/von_demo.py --tour

Web-UI runner (the app's von tab spawns this and talks JSON over
stdin/stdout):
    .venv-von/Scripts/python demos/von_demo.py --serve
    choice mode (the default; the tab's Decide section):
      stdin:  {"texts": [str, ...], "instructions": str,
               "choices": {label: description-or-null}}
      stdout: {"results": [{"choice": str, "probabilities": {label: float},
                            "confidence": float}, ...]}
    judge_rate mode (the tab's Judge + rate section; one shared sample):
      stdin:  {"mode": "judge_rate", "text": str,
               "judge_instructions": str, "rate_instructions": str,
               "rubric": [str, ...]}      # rubric levels, lowest to highest
      stdout: {"judgment": {"noul": float, "verdict": "yes" | "no"},
               "rating": {"score": float, "confidence": float,
                          "legend": {level: str},
                          "probabilities": {level: float}}}
    or {"error": "Type: message"}
"""

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
SENTIMENT_QUESTION = "What is the overall sentiment of this text?"
SENTIMENT_CHOICES = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
}

# the judge and rate stops share one sample (the same ticket the
# intern-decision and lumma tours aim their noul/score questions at)
TICKET = ("I was charged twice for my subscription this month and "
          "want a refund.")
JUDGE_INSTRUCTIONS = "Is the customer asking for a refund?"
RATE_INSTRUCTIONS = "How urgent is this issue?"
RATE_RUBRIC = ["could wait a few days",
               "should be fixed soon",
               "needs immediate action"]


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


def load():
    """engines.von_client.load_von_backend over the pinned checkpoint;
    heavy imports stay inside so the module imports without torch and
    transformers (offline tests). The load time goes to stderr: in
    --serve mode stdout must stay pure JSON for the web UI."""
    from engines.von_client import load_von_backend

    t0 = time.perf_counter()
    backend = load_von_backend()
    print(f"Loaded the von option-marker backend in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return backend


def row(answer) -> dict:
    """ChoiceAnswer -> web-UI row."""
    return {"choice": answer.choice,
            "probabilities": answer.probabilities,
            "confidence": answer.confidence}


def judgment_row(answer) -> dict:
    """NoulAnswer -> web-UI row: the yes-probability plus the derived
    yes/no verdict at the conventional 0.5 threshold."""
    return {"noul": answer.noul,
            "verdict": "yes" if answer.noul >= 0.5 else "no"}


def rating_row(answer) -> dict:
    """ScoreAnswer -> web-UI row."""
    return {"score": answer.score,
            "confidence": answer.confidence,
            "legend": answer.legend,
            "probabilities": answer.probabilities}


def parse_choice_payload(payload: dict) -> tuple[list[str], str, dict]:
    """choice-mode stdin payload -> (texts, instructions, choices).
    ValueError names the problem; serve() turns it into the error JSON."""
    texts = [str(t).strip() for t in payload.get("texts", [])
             if str(t).strip()]
    if not texts:
        raise ValueError("payload needs a non-empty 'texts' list")
    instructions = str(payload.get("instructions") or "").strip()
    if not instructions:
        raise ValueError("payload needs an 'instructions' word")
    choices = payload.get("choices")
    if not isinstance(choices, dict) or not choices:
        raise ValueError("payload needs a non-empty 'choices' object")
    return texts, instructions, choices


def parse_judge_rate_payload(payload: dict) -> tuple[str, str, str, list]:
    """judge_rate-mode stdin payload -> (text, judge instructions, rate
    instructions, rubric). ValueError names the problem."""
    text = str(payload.get("text") or "").strip()
    if not text:
        raise ValueError("payload needs a non-empty 'text'")
    judge_instructions = str(payload.get("judge_instructions") or "").strip()
    if not judge_instructions:
        raise ValueError("payload needs 'judge_instructions'")
    rate_instructions = str(payload.get("rate_instructions") or "").strip()
    if not rate_instructions:
        raise ValueError("payload needs 'rate_instructions'")
    rubric = [str(level).strip() for level in payload.get("rubric", [])
              if str(level).strip()]
    if len(rubric) < 2:
        raise ValueError("payload needs a 'rubric' of at least two "
                         "levels, lowest to highest")
    return text, judge_instructions, rate_instructions, rubric


# ------------------------------------------------------------------ runner
def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        mode = str(payload.get("mode") or "choice").strip().lower()
        if mode == "judge_rate":
            text, judge_instructions, rate_instructions, rubric = \
                parse_judge_rate_payload(payload)
            from engines.von_client import load_von_judge, load_von_rate

            backend = load()
            judgment = judgment_row(load_von_judge(backend)(
                state=text, instructions=judge_instructions))
            rating = rating_row(load_von_rate(backend)(
                state=text, rubric=rubric,
                instructions=rate_instructions))
            # ASCII-escaped JSON survives Windows pipes using legacy code pages.
            print(json.dumps({"judgment": judgment, "rating": rating}))
            return
        texts, instructions, choices = parse_choice_payload(payload)
        from engines.von_client import load_von_decider

        decide = load_von_decider()
        results = []
        for text in texts:
            res = decide(state=text, choices=choices,
                         instructions=instructions)
            results.append(row(res))
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading wfzyx/von-1.0 (the complete trained option-marker "
          "checkpoint; first run downloads it)...")
    from engines.von_client import load_von_decider, load_von_judge, \
        load_von_rate

    backend = load()
    decide = load_von_decider(backend)
    judge = load_von_judge(backend)
    rate = load_von_rate(backend)

    # ---------------------------------------------------------------- 1
    banner("1. Choice - the shared sample texts, one decide() per text")
    for text in SHARED_TEXTS:
        answer = row(timed(decide, state=text, choices=SENTIMENT_CHOICES,
                           instructions=SENTIMENT_QUESTION))
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in answer["probabilities"].items())
        print(f'  {str(answer["choice"]):<8} {probs}  "{text[:48]}"')

    # ---------------------------------------------------------------- 2
    banner("2. Judge - yes/no probability for one shared sample")
    print(f'  state: "{TICKET}"')
    print(f'  question: "{JUDGE_INSTRUCTIONS}"  (evaluate_noul packs the '
          "affirmative and the negation into one forward pass)")
    answer = timed(judge, state=TICKET, instructions=JUDGE_INSTRUCTIONS)
    show(judgment_row(answer))

    # ---------------------------------------------------------------- 3
    banner("3. Rate - ordinal rubric over the same sample")
    print(f'  state: "{TICKET}"')
    print(f'  question: "{RATE_INSTRUCTIONS}"')
    print(f'  rubric: {RATE_RUBRIC}  (lowest to highest; evaluate_score '
          "reads one probability per level)")
    answer = timed(rate, state=TICKET, rubric=RATE_RUBRIC,
                   instructions=RATE_INSTRUCTIONS)
    show(rating_row(answer))
    print("  ('score' is the expectation over the levels; the "
          "probabilities are keyed by level index)")

    print("\nDone. Same texts through the other typed-decision engines:  "
          ".venv-von/Scripts/python demos/lumma_demo.py / "
          "demos/intern_decision_demo.py\n")


def main() -> None:
    if "--tour" in sys.argv:
        tour()
        return
    serve()


if __name__ == "__main__":
    main()
