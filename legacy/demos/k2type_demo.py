"""K2-Type-0.9B demo + one-shot runner for the web UI - runs in .venv-von.

IFM's K2-Type-0.9B (Apache 2.0, 0.9B active params on a K2-Horizon-0.9B
backbone) is a Jev-style typed-decision model: a state (text or JSON) and
typed questions (choice / score / noul) go in through ONE forward pass and
come back as full softmax probabilities per question - the state is encoded
once and the questions are isolated by a block-causal attention mask, so
adding a question never changes another's answer. No generation, ever.

The weights and the minimal runtime ship inside the HF repo; the snapshot
lives at C:\\src\\K2-Type-0.9B (K2TYPE_HOME overrides) and
engines/k2type_client.py loads it in-process on the CPU (fp32) - the
upstream `jev.serve` server hardcodes a CUDA GPU this machine does not
have. 0.762 on the JevBench public set per the model card.

    .venv-von/Scripts/python demos/k2type_demo.py

Tour (interactive):
    .venv-von/Scripts/python demos/k2type_demo.py

Web-UI runner (the app's K2-Type tab spawns this and talks JSON over
stdin/stdout):
    .venv-von/Scripts/python demos/k2type_demo.py --serve
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

from engines.k2type_client import MODEL_ID, model_home

SHARED_TEXTS = [
    "The food was cold and the waiter was rude.",
    "This is the best laptop I have ever owned.",
    "The meeting is scheduled for 3 PM in the main conference room.",
    "The flight was delayed for six hours with no explanation.",
    "She was thrilled with her exam results.",
    "Water boils at 100 degrees Celsius at sea level.",
]
QUESTION = {"sentiment": "What is the overall sentiment of this text",
            "topic": "Which topic category does this text belong to"}

# described criteria for the choice questions (same wording as bench.py's
# label dicts; descriptions are what the pointer head scores, so described
# criteria are the faithful shape)
CRITERIA = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual or mixed text without a clear attitude",
    "technology": "Software, hardware, AI, gadgets, engineering",
    "business": "Companies, markets, revenue, deals, management",
    "sports": "Athletes, matches, teams, tournaments",
    "politics": "Government, elections, policy, legislation",
}


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


def describe_labels(labels, task: str) -> dict:
    """criteria dict for arbitrary labels: the bench descriptions where
    they exist, otherwise a task-phrased one."""
    return {label: CRITERIA.get(label, f"Text whose {task} is {label}")
            for label in labels}


def choice_question(instructions: str, criteria: dict) -> dict:
    """One choice question as predict() wants it: criteria maps option
    name -> description (None allowed; the option name is used then)."""
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load():
    """k2type_client.load_engine over the local snapshot; heavy imports
    stay inside so the module imports without torch (offline tests). The
    load time goes to stderr: in --serve mode stdout must stay pure JSON
    for the web UI."""
    from engines.k2type_client import load_engine

    t0 = time.perf_counter()
    engine = load_engine()
    print(f"Loaded {MODEL_ID} from {model_home()} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def question_for(task: str, text: str) -> str:
    """Instruction for one text: the question with the text restated in
    it (the house shape across the typed-decision benchmarks; the bench
    branch comments carry the request-shape probe)."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to")
    return f'{question}: "{text}"'


def answer_row(answer: dict) -> dict:
    """predict() answer -> web-UI row; confidence is the model's own
    renormalized margin ((pmax - 1/K) / (1 - 1/K)), not pmax."""
    return {"choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence")}


def decide_one(engine, text: str, task: str, criteria: dict) -> dict:
    """One predict() per text (one choice question); the answer maps
    straight onto the web-UI row."""
    answers = engine.predict(text, {"q": choice_question(
        question_for(task, text), criteria)})
    return answer_row(answers["answers"]["q"])


def parse_serve_payload(payload: dict) -> tuple[list[str], str, list[str]]:
    """stdin payload -> (texts, task, labels). ValueError names the
    problem; serve() turns it into the error JSON."""
    texts = [str(t).strip() for t in payload.get("texts", [])
             if str(t).strip()]
    if not texts:
        raise ValueError("payload needs a non-empty 'texts' list")
    task = str(payload.get("task") or "").strip()
    if not task:
        raise ValueError("payload needs a 'task' word")
    labels = [str(l).strip() for l in payload.get("labels", [])
              if str(l).strip()]
    if not labels:
        raise ValueError("payload needs a non-empty 'labels' list")
    return texts, task, labels


# ------------------------------------------------------------------ runner
def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        texts, task, labels = parse_serve_payload(payload)
        engine = load()
        criteria = describe_labels(labels, task)
        results = [decide_one(engine, text, task, criteria)
                   for text in texts]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading {MODEL_ID} (IFM's K2-Type typed-decision model; loads "
          f"the local snapshot at {model_home()})...")
    engine = load()

    # ---------------------------------------------------------------- 1
    banner("1. One state, three typed questions, a single predict() call")
    ticket = ("I was charged twice for my subscription this month and "
              "want a refund.")
    questions = {
        "team": choice_question("Which team should handle this?", {
            "billing": "Payments, invoices and refunds",
            "technical": "Bugs, errors and outages",
            "account": "Login, profile and settings"}),
        "refund": {"type": "noul",
                   "instructions": "Is the customer asking for a refund?"},
        "urgency": {"type": "score",
                    "instructions": "How urgent is this issue?",
                    "criteria": ["could wait a few days",
                                 "should be fixed soon",
                                 "needs immediate action"]},
    }
    print(f'  state: "{ticket}"  (the three questions are isolated by the '
          "attention mask in one forward pass; no generation)")
    answers = timed(engine.predict, ticket, questions)
    show(answers)
    print("  (a score answer's 'score' is the expected rubric index and "
          "its probabilities are keyed by index)")

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one predict() per text")
    criteria = describe_labels(["positive", "negative", "neutral"],
                               "sentiment")
    for text in SHARED_TEXTS:
        row = timed(decide_one, engine, text, "sentiment", criteria)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    print("\nDone. Same texts through the other typed-decision engines:  "
          ".venv-von/Scripts/python demos/julia_demo.py / "
          ".venv-von/Scripts/python demos/intern_decision_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    tour()


if __name__ == "__main__":
    main()
