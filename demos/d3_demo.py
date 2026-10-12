"""Decision 3.0 demo + one-shot runner for the web UI - runs in .venv-von.

vLLM Semantic Router's Decision 3.0 family (Apache 2.0, "Towards Open
Multimodal Foundation Decision Models") — the third generation of the
line behind the Ollaya-served `decision` model this repo already benches
(and the successor to the benched Decision 2.0 trio). Multimodal
Qwen3.5-backbone checkpoints with a code-readout head: a state (text or
JSON — images and videos ride the same system_one call on a GPU host)
and typed questions (choice / noul / score) go in, the prompt lists
every option under a single-token answer code and a 255-way readout
scores those codes at the last prompt position — full softmax
probabilities per question, one forward pass per question, no
generation. The packages are self-verifying (MODEL_MANIFEST.json: every
file up to 64 MiB SHA-256-checked, weight shards size-checked, before
anything loads).

Sizes: edge (default, 594M), lite (853M), nano (2.2B). The snapshots
live under C:\\src (D3_HOME overrides the prefix) and
engines/d3_client.py loads them in-process on the CPU (bf16 backbone,
fp32 readout; the Gated DeltaNet layers run transformers' PyTorch
reference) - no server, nothing pip-installed (the runtime ships inside
each snapshot). The 0.6B card self-reports 18.0 Jev Decision Index 0.3.1
against Decision 2.0 0.6B's 13.6.

    .venv-von/Scripts/python demos/d3_demo.py
    .venv-von/Scripts/python demos/d3_demo.py --size nano

Web-UI runner (the app's Decision 3.0 tab spawns this and talks JSON
over stdin/stdout):
    .venv-von/Scripts/python demos/d3_demo.py --serve [--size S]
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

from engines.d3_client import DEFAULT_SIZE, SIZES, normalize_size

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
# label dicts; descriptions are what the answer codes are keyed to)
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
    name -> description."""
    return {"type": "choice", "instructions": instructions,
            "criteria": criteria}


def load(size=DEFAULT_SIZE):
    """d3_client.load_engine over the local snapshot; heavy imports stay
    inside so the module imports without torch (offline tests). The load
    time goes to stderr: in --serve mode stdout must stay pure JSON for
    the web UI."""
    from engines.d3_client import load_engine, model_dir

    t0 = time.perf_counter()
    engine = load_engine(size)
    print(f"Loaded {engine.name} from {model_dir(size)} in "
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
    normalized-entropy measure from the vendored product_answer, not
    pmax."""
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


def parse_size(argv: list[str]) -> str:
    """--size edge | lite | nano from the command line (normalize_size
    validates; ValueError names the problem)."""
    for flag in ("--size", "-s"):
        if flag in argv:
            pos = argv.index(flag)
            if pos + 1 >= len(argv):
                raise ValueError(f"{flag} needs a size "
                                 f"(one of {', '.join(SIZES)})")
            return normalize_size(argv[pos + 1])
    return DEFAULT_SIZE


# ------------------------------------------------------------------ runner
def serve(size=DEFAULT_SIZE) -> None:
    """Answer one stdin payload (the web UI's runner). The size comes in
    as a parameter - main() owns argv parsing, so serving never re-reads
    the command line (a stray -s from another harness must not be taken
    for --size)."""
    try:
        payload = json.loads(sys.stdin.read())
        texts, task, labels = parse_serve_payload(payload)
        engine = load(size)
        criteria = describe_labels(labels, task)
        results = [decide_one(engine, text, task, criteria)
                   for text in texts]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"results": results}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour(size: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading {SIZES[size]} (vLLM Semantic Router's Decision 3.0; "
          "loads the local snapshot and verifies the package against "
          "the manifest)...")
    engine = load(size)

    # ---------------------------------------------------------------- 1
    banner("1. One state, three typed questions, a single predict() call")
    ticket = ("The order arrived damaged yesterday. The customer has a "
              "receipt and asks for a replacement today.")
    questions = {
        "route": choice_question("Which team should handle this request?", {
            "returns": "Refunds, replacements and damaged deliveries",
            "billing": "Payments, invoices and charges",
            "technical": "Product setup and faults"}),
        "receipt": {"type": "noul",
                    "instructions": "Does the customer have a receipt?"},
        "urgency": {"type": "score",
                    "instructions": "How urgent is this request?",
                    "criteria": ["Routine", "Soon", "Today"]},
    }
    print(f'  state: "{ticket}"  (the model card\'s own example: the three '
          "questions share one request — one forward per question over "
          "the state, no generation)")
    answers = timed(engine.predict, ticket, questions)
    show(answers)
    print("  (a score answer's 'score' is the expected rubric index and "
          "its probabilities are keyed by index; a noul answer is the "
          "true probability alone)")

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one predict() per text")
    criteria = describe_labels(["positive", "negative", "neutral"],
                               "sentiment")
    for text in SHARED_TEXTS:
        row = timed(decide_one, engine, text, "sentiment", criteria)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    print("\nDone. The other sizes: --size lite / --size nano. "
          "Same texts through the other typed-decision engines:  "
          ".venv-von/Scripts/python demos/decision2_demo.py / "
          ".venv-von/Scripts/python demos/k2type_demo.py\n")


def main() -> None:
    argv = sys.argv[1:]
    size = parse_size(argv)
    if "--serve" in argv:
        serve(size)
        return
    tour(size)


if __name__ == "__main__":
    main()
