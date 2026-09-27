"""Lumma-Fev demo + one-shot runner for the web UI - runs inside .venv-von.

FrontiersMind's Lumma-Fev family (Apache 2.0) are typed-decision models:
a causal transformer, prefill-only - each question is scored on its own
row over the state (`<state> text <q> instructions <opt> options <decide>`)
and a pointer head reads one probability per option, so ONE decide() call
answers every question about a state in a single forward pass. There is no
generation and no generate(). Checkpoints: Lumma-fev-0.1b (154M, card name
"Lumma-Fev-0.15B"), -0.6b (649M), -4b (4.2B); the 9b sibling is not
benchmarked here (~16 GB bf16, see README). The lumma-fev package needs
transformers >=5.4,<6, so it lives in .venv-von next to von/JevK5-Lite/
MoJev; its loader defaults to fp32 on CPU (the checkpoint stores a bf16
backbone plus an fp32 pointer head), so no dtype workaround is needed.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/lumma_demo.py               # 0.15B
    .venv-von/Scripts/python demos/lumma_demo.py --model 0.6b

Web-UI runner (the app's Lumma tab spawns this like jevk5_demo.py and
talks JSON over stdin/stdout):
    .venv-von/Scripts/python demos/lumma_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...],
             "model": "0.15b" | "0.6b" | "4b"}     # "model" optional
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
QUESTION = {"sentiment": "What is the overall sentiment of this text",
            "topic": "Which topic category does this text belong to"}

# benchmark/demo shorthand -> HF checkpoint (README: Family maps)
MODEL_IDS = {
    "0.15b": "FrontiersMind/Lumma-fev-0.1b",
    "0.6b": "FrontiersMind/Lumma-fev-0.6b",
    "4b": "FrontiersMind/Lumma-fev-4b",
}
DEFAULT_MODEL_KEY = "0.15b"

# described criteria for the tour's choice questions (same wording as the
# Ollaya demo, so outputs compare directly)
SENTIMENT = {
    "positive": "Text expresses a clearly positive attitude",
    "negative": "Text expresses a clearly negative attitude",
    "neutral": "Factual text without a clear attitude",
}

# described criteria for the serve path - same wording as bench.py's
# SENTIMENT_LABELS/TOPIC_LABELS (and the julia demo), because bare-label
# criteria measurably hurt these scorers (the Ollaya finding)
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


def resolve_model_id(value=None) -> str:
    """Registry key (0.15b/0.6b/4b), full HF id or None -> checkpoint id
    (None = the default 0.15B). ValueError names the problem so the
    --serve runner can turn it into the error JSON."""
    if value is None or not str(value).strip():
        return MODEL_IDS[DEFAULT_MODEL_KEY]
    key = str(value).strip().lower()
    if key in MODEL_IDS:
        return MODEL_IDS[key]
    for canonical in MODEL_IDS.values():
        if key == canonical.lower():
            return canonical
    raise ValueError(f"unknown Lumma checkpoint {value!r}; known: "
                     f"{', '.join(MODEL_IDS)} or the HF id")


def describe_labels(labels) -> dict:
    """criteria dict for the serve path: the bench description for each
    known sentiment/topic label (bare-label criteria measurably hurt
    these scorers, per the Ollaya finding), bare (None) for unknown
    labels."""
    return {label: CRITERIA.get(label) for label in labels}


def choice_question(instructions: str, criteria: dict) -> dict:
    """One choice question as decide() wants it: criteria maps option
    name -> description (None renders the bare name as the option)."""
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load(model_key=None):
    """lumma_fev.load for a registry key / HF id; heavy imports stay inside
    so the module imports without lumma_fev and torch (offline tests). The
    load time goes to stderr: in --serve mode stdout must stay pure JSON
    for the web UI."""
    import lumma_fev

    model_id = resolve_model_id(model_key)
    t0 = time.perf_counter()
    # fp32 on CPU is the package default (bf16 on GPU); the pointer head
    # is fp32-trained either way
    model = lumma_fev.load(model_id)
    print(f"Loaded {model_id} in {time.perf_counter() - t0:.1f}s",
          file=sys.stderr)
    return model


def question_for(task: str, text: str) -> str:
    """Instruction for one text: the question with the text restated in
    it. The 0.1b probe (eight easy-tier sentiment texts, deterministic
    across repeats) scored this shape - text in the state AND the
    instructions - 5/8, bare instructions 3/8, and the kev shape (empty
    state, text only in the instructions) 3/8 with its argmax collapsing
    onto one label: the text is effectively dropped. Same shape as the
    benchmarks."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to")
    return f'{question}: "{text}"'


def decide_one(model, text: str, task: str, criteria: dict) -> dict:
    """One decide() per text (one choice question); the answer maps
    straight onto the web-UI row."""
    answers = model.decide(
        text, {"q": choice_question(question_for(task, text), criteria)})
    answer = answers["q"]
    return {"choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence")}


def parse_serve_payload(payload: dict) -> tuple[list[str], str, list[str], str]:
    """stdin payload -> (texts, task, labels, checkpoint id). ValueError
    names the problem; serve() turns it into the error JSON."""
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
    return texts, task, labels, resolve_model_id(payload.get("model"))


# ------------------------------------------------------------------ runner
def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        texts, task, labels, _ = parse_serve_payload(payload)
        model = load(payload.get("model"))
        criteria = describe_labels(labels)
        results = [decide_one(model, text, task, criteria)
                   for text in texts]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour(model_key=None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    model_id = resolve_model_id(model_key)
    print(f"Loading {model_id} (Lumma-Fev typed-decision model; first run "
          "downloads the checkpoint)...")
    model = load(model_key)

    # ---------------------------------------------------------------- 1
    banner("1. One state, three typed questions, a single decide() call")
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
    print(f'  state: "{ticket}"  (all three scored in one forward pass; '
          "questions never see each other)")
    answers = timed(model.decide, ticket, questions)
    show(answers)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one decide() per text")
    for text in SHARED_TEXTS:
        row = timed(decide_one, model, text, "sentiment", SENTIMENT)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    print("\nDone. Same texts through the other typed-decision engines:  "
          ".venv/Scripts/python demos/ollaya_demo.py / "
          ".venv-von/Scripts/python demos/mojev_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    model_key = None
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 >= len(sys.argv):
            raise SystemExit("--model needs a value; known: "
                             f"{', '.join(MODEL_IDS)}")
        model_key = sys.argv[idx + 1]
    tour(model_key)


if __name__ == "__main__":
    main()
