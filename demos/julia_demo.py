"""Julia 1 demo + one-shot runner for the web UI - runs inside .venv-von.

SupersonicLabs' Julia 1 (Apache 2.0, 144.3M params) is a typed-decision
model: mmBERT-small (JHU CLSP's multilingual ModernBERT encoder) plus a
decision head. A state and typed questions (choice/score/noul, 2-20
options each; choice options REQUIRE nonempty descriptions) go in through
ONE predict() call and come back as full softmax probabilities - the
questions are independently scored as one batch, there is no generation.
The `julia` runtime package ships inside the HF repo; the complete
snapshot lives at ../Julia-1 (JULIA_HOME overrides) and is installed
--no-deps into .venv-von because its pyproject pins transformers <5.1
(engines/julia_client.py handles the 5.17 drift).

The engine runs 16 torch CPU threads by default (JULIA_CPU_THREADS
overrides; engines/julia_client.py applies it at load):
    .venv-von/Scripts/python demos/julia_demo.py

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/julia_demo.py

Web-UI runner (the app's Julia tab spawns this and talks JSON over
stdin/stdout):
    .venv-von/Scripts/python demos/julia_demo.py --serve
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

from engines.julia_client import MODEL_ID, model_home

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

# described criteria for the choice questions (same wording as
# bench.py's label dicts - Julia rejects empty descriptions)
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
    they exist, otherwise a task-phrased one (Julia requires a nonempty
    description for every choice option)."""
    return {label: CRITERIA.get(label, f"Text whose {task} is {label}")
            for label in labels}


def choice_question(instructions: str, criteria: dict) -> dict:
    """One choice question as predict() wants it: criteria maps option
    name -> nonempty description."""
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load():
    """julia_client.load_engine over the ../Julia-1 snapshot; heavy
    imports stay inside so the module imports without julia and torch
    (offline tests). The load time goes to stderr: in --serve mode
    stdout must stay pure JSON for the web UI."""
    from engines.julia_client import load_engine

    t0 = time.perf_counter()
    engine = load_engine()
    print(f"Loaded {MODEL_ID} from {model_home()} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def question_for(task: str, text: str) -> str:
    """Instruction for one text: the question with the text restated in
    it (the house shape across the typed-decision benchmarks). Julia's
    probe (eight easy-tier sentiment texts, deterministic across repeats)
    tied both shapes at 3/8 with identical picks - the restatement does
    not move this model, it just stays comparable (bench branch comments
    carry the finding)."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to")
    return f'{question}: "{text}"'


def answer_row(answer: dict) -> dict:
    """predict() answer -> web-UI row; confidence is the answer's
    max_probability (Julia reports full softmax, no separate confidence)."""
    return {"choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("max_probability")}


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

    print(f"Loading {MODEL_ID} (Julia 1 typed-decision model; loads the "
          f"local snapshot at {model_home()})...")
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
    print(f'  state: "{ticket}"  (the three questions are independently '
          "scored in one batch; no generation)")
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
          ".venv-von/Scripts/python demos/lumma_demo.py / "
          ".venv/Scripts/python demos/ollaya_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    tour()


if __name__ == "__main__":
    main()
