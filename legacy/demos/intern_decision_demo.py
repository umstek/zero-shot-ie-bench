"""Intern-Decision demo + one-shot runner for the web UI - runs inside
.venv-von.

internlm's Intern-Decision family (Apache 2.0, fine-tuned from Qwen3.5)
are multimodal structured-decision models: one CAUSAL forward pass over
the state, the decision schema and an assistant JSON skeleton whose
fields hold masked `<decision>` placeholders - logits read immediately
before each placeholder, softmaxed over the field's candidate symbols
(A, B, ... up to 62 options, 1-16 questions per request), then a
per-checkpoint calibration temperature rescales the probabilities
(argmax preserved). No generate() call. Checkpoints: 0.8B (853M), 2B
(2.2B), 4B (4.5B). The runtime ships inside each HF snapshot
(engines/intern_decision_client.py loads it from the local
C:\\src/Intern-Decision-* snapshots, INTERN_DECISION_HOME overrides);
the loader defaults to fp32 on CPU (bf16 runs ~7x slower on this
machine with identical predictions - see the client's docstring).

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/intern_decision_demo.py               # 0.8B
    .venv-von/Scripts/python demos/intern_decision_demo.py --model 2b

Web-UI runner (the app's Intern-Decision tab spawns this and talks JSON
over stdin/stdout):
    .venv-von/Scripts/python demos/intern_decision_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...],
             "model": "0.8b" | "2b" | "4b"}     # "model" optional
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

from engines.intern_decision_client import SIZES, normalize_size, size_home

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
# bench.py's label dicts)
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
    they exist, otherwise a task-phrased one (choice criteria must be an
    object mapping option -> description)."""
    return {label: CRITERIA.get(label, f"Text whose {task} is {label}")
            for label in labels}


def choice_question(instructions: str, criteria: dict) -> dict:
    """One choice question as predict() wants it: criteria maps option
    name -> description."""
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load(model_key=None):
    """intern_decision_client.load_engine over the size's local snapshot;
    heavy imports stay inside so the module imports without torch and
    transformers (offline tests). The load time goes to stderr: in
    --serve mode stdout must stay pure JSON for the web UI."""
    from engines.intern_decision_client import load_engine

    size = normalize_size(model_key)
    t0 = time.perf_counter()
    engine = load_engine(size)
    print(f"Loaded {SIZES[size]} from {size_home(size)} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def question_for(task: str, text: str) -> str:
    """Instruction for one text: the question with the text restated in
    it (the house shape across the typed-decision benchmarks). The 0.8B
    probe (eight easy-tier sentiment texts, deterministic across repeats)
    tied both shapes at 8/8 with identical picks - the restatement does
    not move this model, it just stays comparable (bench branch comments
    carry the finding)."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to")
    return f'{question}: "{text}"'


def answer_row(answer: dict) -> dict:
    """predict() answer -> web-UI row (the answer already carries choice,
    full softmax probabilities and a confidence)."""
    return {"choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence")}


def decide_one(engine, text: str, task: str, criteria: dict) -> dict:
    """One predict() per text (one choice question); the request is a
    single dict with the state and the questions."""
    request = {"state": text,
               "questions": {"q": choice_question(
                   question_for(task, text), criteria)}}
    answers = engine.predict(request)
    return answer_row(answers["answers"]["q"])


def parse_serve_payload(payload: dict) -> tuple[list[str], str, list[str], str]:
    """stdin payload -> (texts, task, labels, size key). ValueError names
    the problem; serve() turns it into the error JSON."""
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
    return texts, task, labels, normalize_size(payload.get("model"))


# ------------------------------------------------------------------ runner
def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        texts, task, labels, _ = parse_serve_payload(payload)
        engine = load(payload.get("model"))
        criteria = describe_labels(labels, task)
        results = [decide_one(engine, text, task, criteria)
                   for text in texts]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour(model_key=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    size = normalize_size(model_key)
    print(f"Loading {SIZES[size]} (Intern-Decision typed-decision model; "
          f"loads the local snapshot at {size_home(size)})...")
    engine = load(size)

    # ---------------------------------------------------------------- 1
    banner("1. One state, three typed questions, a single predict() call")
    ticket = ("I was charged twice for my subscription this month and "
              "want a refund.")
    request = {
        "state": ticket,
        "questions": {
            "team": choice_question("Which team should handle this?", {
                "billing": "Payments, invoices and refunds",
                "technical": "Bugs, errors and outages",
                "account": "Login, profile and settings"}),
            "refund": {"type": "noul",
                       "instructions": "Is the customer asking for a "
                                       "refund?"},
            "urgency": {"type": "score",
                        "instructions": "How urgent is this issue?",
                        "criteria": ["could wait a few days",
                                     "should be fixed soon",
                                     "needs immediate action"]},
        },
    }
    print(f'  state: "{ticket}"  (every field is scored in the same '
          "forward pass at its masked <decision> slot; no generation)")
    answers = timed(engine.predict, request)
    show(answers)
    print("  (a score answer's 'score' is the expectation over the rubric "
          "and its probabilities are keyed by rubric index; a noul answer's "
          "'noul' is the yes-probability)")

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
          "demos/julia_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    model_key = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("--model needs a value; known: "
                             f"{', '.join(SIZES)}")
        model_key = sys.argv[idx + 1]
    tour(model_key)


if __name__ == "__main__":
    main()
