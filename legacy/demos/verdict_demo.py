"""Verdict (RLCD ModernBERT 151M) demo - runs under the agent-jev venv
python (transformers 5): C:/venvs/agent-jev/Scripts/python.

Verdict is the heman10x/rlcd-modernbert-151m decision encoder: a
ModernBERT-base backbone with a trained abstention head. The repo's
benches only ever send it Choice queries and read res.is_abstention
passively; the rlcd runtime also serves Score rubrics (ordinal Level
objects with numerical values, expectation + selected level back) and
Noul propositions (true / false / insufficient-evidence with the
evidence-v2 semantics) - and evaluate() takes a MIXED batch of all
three in one call. This demo exercises that full surface:

  1. one evaluate() over a support ticket answering a Choice (team),
     a Score (urgency rubric) and a Noul (refund proposition) at once;
  2. an abstention probe: an unrelated context where the trained
     insufficient-evidence head should fire on the same questions.

The engine runs in-process from the VERDICT_HOME checkout (default
C:\\src\\verdict); its artifacts load from artifacts/v2. Heavy imports
(rlcd, torch, transformers) stay inside load() so offline tests import
this module freely.

Tour (interactive):
    C:/venvs/agent-jev/Scripts/python demos/verdict_demo.py

Web-UI runner (a tab can spawn this and talk JSON over stdin/stdout):
    C:/venvs/agent-jev/Scripts/python demos/verdict_demo.py --serve
    stdin:  {"context": str, "mode": "choice" | "score" | "noul",
             "labels": [str, ...],          # choice options / score
                                              rubric levels, low->high
             "question" | "proposition": str}
    stdout: {"result": {...}} or {"error": "Type: message"}
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import os
import sys
import time

VERDICT_HOME = _os.environ.get("VERDICT_HOME", r"C:\src\verdict")

TICKET = ("I was charged twice for my subscription this month and "
          "want a refund.")
UNRELATED = "Water boils at 100 degrees Celsius at sea level."
TEAM_OPTIONS = [("billing", "Payments, invoices and refunds"),
                ("technical", "Bugs, errors and outages"),
                ("account", "Login, profile and settings")]
URGENCY_LEVELS = [("low", "Could wait a few days", 0.0),
                  ("medium", "Should be fixed soon", 1.0),
                  ("high", "Needs immediate action", 2.0)]
REFUND_PROPOSITION = "The customer is asking for a refund."
# the evidence-v2 literal the Noul query's semantics field requires
NOUL_SEMANTICS = "conditional_on_sufficient_evidence_v2"


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
    """DecisionEngine over the Verdict checkout's artifacts/v2. rlcd
    imports stay inside so the module imports without torch; the load
    message goes to stderr (--serve keeps stdout pure JSON)."""
    sys.path.insert(0, VERDICT_HOME)
    from rlcd import DecisionEngine

    t0 = time.perf_counter()
    engine = DecisionEngine(model_name_or_path=os.path.join(
        VERDICT_HOME, "artifacts", "v2"), device="cpu")
    print(f"Loaded rlcd DecisionEngine from {VERDICT_HOME} in "
          f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
    return engine


def mixed_queries():
    """The tour's Choice + Score + Noul batch (one evaluate() call)."""
    sys.path.insert(0, VERDICT_HOME)
    from rlcd import Choice, Level, Noul, Option, Score

    return [
        Choice(id="team",
               question="Which team should handle this ticket?",
               options=[Option(id=name, description=description)
                        for name, description in TEAM_OPTIONS]),
        Score(id="urgency",
              question="How urgent is this issue?",
              levels=[Level(id=name, description=description, value=value)
                      for name, description, value in URGENCY_LEVELS]),
        Noul(id="refund", proposition=REFUND_PROPOSITION,
             semantics=NOUL_SEMANTICS),
    ]


def result_row(result) -> dict:
    """One evaluate() result -> plain dict for the tour/show and serve."""
    row = {"kind": result.kind, "is_abstention": result.is_abstention}
    if result.kind == "choice":
        row["selected"] = result.selected_id
        row["probabilities"] = result.probabilities
    elif result.kind == "score":
        row["selected_level"] = result.selected_level_id
        row["selected_value"] = result.selected_value
        row["expected_score"] = result.expected_score
        row["probabilities"] = result.probabilities
    else:
        row["outcome"] = result.selected_outcome
        row["p_true_given_evidence"] = (
            result.p_true_given_sufficient_evidence)
        row["p_insufficient_evidence"] = result.p_insufficient_evidence
    return row


# ------------------------------------------------------------------ serve
def parse_serve_payload(payload: dict):
    """stdin payload -> (context, one rlcd query). ValueError names the
    problem; serve() turns it into the error JSON."""
    sys.path.insert(0, VERDICT_HOME)
    from rlcd import Choice, Level, Noul, Option, Score

    context = str(payload.get("context") or "").strip()
    if not context:
        raise ValueError("payload needs a non-empty 'context'")
    mode = str(payload.get("mode") or "").strip().lower()
    labels = [str(label).strip() for label in payload.get("labels", [])
              if str(label).strip()]
    if mode == "choice":
        if len(labels) < 2:
            raise ValueError("choice needs at least two 'labels'")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("choice needs a 'question'")
        query = Choice(id="q", question=question,
                       options=[Option(id=label, description=label)
                                for label in labels])
    elif mode == "score":
        if len(labels) < 2:
            raise ValueError("score needs at least two 'labels' rubric "
                             "levels, low to high")
        question = str(payload.get("question") or "").strip()
        if not question:
            raise ValueError("score needs a 'question'")
        query = Score(id="q", question=question,
                      levels=[Level(id=label, description=label,
                                    value=float(index))
                              for index, label in enumerate(labels)])
    elif mode == "noul":
        proposition = str(payload.get("proposition")
                          or payload.get("question") or "").strip()
        if not proposition:
            raise ValueError("noul needs a 'proposition'")
        query = Noul(id="q", proposition=proposition,
                     semantics=NOUL_SEMANTICS)
    else:
        raise ValueError("mode must be choice, score or noul")
    return context, query


def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        context, query = parse_serve_payload(payload)
        engine = load()
        result = engine.evaluate(context=context, queries=[query]).results[0]
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"result": result_row(result)}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Loading Verdict (rlcd DecisionEngine over {VERDICT_HOME})...")
    engine = load()

    # ---------------------------------------------------------------- 1
    banner("1. One ticket, THREE query kinds in one evaluate() call")
    print(f'  context: "{TICKET}"')
    batch = timed(engine.evaluate, context=TICKET, queries=mixed_queries())
    for result in batch.results:
        show(result_row(result))

    # ---------------------------------------------------------------- 2
    banner("2. Abstention probe - unrelated context, same questions")
    print(f'  context: "{UNRELATED}"')
    print("  (the trained insufficient-evidence head should fire - the "
          "benches only ever read this flag after a Choice; here it is "
          "the point)")
    batch = timed(engine.evaluate, context=UNRELATED,
                  queries=mixed_queries())
    for result in batch.results:
        row = result_row(result)
        if row["kind"] == "noul":
            note = (f'outcome={row["outcome"]} '
                    f'(p_insufficient={row["p_insufficient_evidence"]:.2f})')
        else:
            note = (f'abstain={row["is_abstention"]} '
                    f'selected={row.get("selected", row.get("selected_level"))}')
        print(f"  {row['kind']:<7} {note}")

    print("\nDone. Verdict has no web-UI tab (bench-only until now); the "
          "other one-forward decision encoders: demos/certo_demo.py, "
          "demos/jevk5_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    tour()


if __name__ == "__main__":
    main()
