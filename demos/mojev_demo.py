"""MoJev 0.85B demo + one-shot runner for the web UI - runs inside .venv-von.

MoJev (MoLeMo-Lab/mojev, 0.85B) is a packed one-pass typed-decision scorer:
state, question and every candidate value are packed under one tree
attention mask (Qwen3.5 encoder with fla linear-attention kernels) and one
forward pass returns a calibrated probability per candidate. The scorer
class ships inside the HF repo and loads via trust_remote_code; the request
path is adapted in the vendored engines/mojev_engine/ (MIT). It needs transformers
5.17 for the Qwen3.5 encoder, so it lives in .venv-von, not the main venv.

Sample texts are shared with the other demos so outputs compare directly.

Tour (interactive):
    .venv-von/Scripts/python demos/mojev_demo.py
    Stops 1-3: single-choice sentiment/topic via score(), one packed pass
    per question. Stop 4: noul, a yes/no question answered as one P(yes)
    scalar. Stop 5: score, an ordered rubric whose expected level falls
    between integers. Stop 6: one state asked a choice + a noul + a score
    question in ONE packed forward (answer_typed), contrasted with three
    separate single-choice forwards.

Web-UI runner (the app's MoJev tab spawns this like jevk5_demo.py and
talks JSON over stdin/stdout):
    .venv-von/Scripts/python demos/mojev_demo.py --serve
    stdin:  {"texts": [str, ...], "task": str, "labels": [str, ...]}
    stdout: {"results": [{"choice": str | None,
                          "probabilities": {label: float},
                          "confidence": float}, ...]}
            or {"error": "Type: message"}

Typed-questions runner (mode "typed": k System One-style questions about
one state, ONE packed forward — stop 6 of the tour as a service):
    stdin:  {"mode": "typed", "state": str,
             "questions": {name: {"type": "choice" | "noul" | "score",
                                  "instructions": str,
                                  "criteria": ...}}}
    stdout: {"answers": {name: {...}}, "usage": {"input_tokens": int,
                                                 "output_tokens": 0}}
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
QUESTION = {"sentiment": "What is the overall sentiment of this text?",
            "topic": "Which topic category does this text belong to?"}
TICKET = ("I was charged twice for my subscription this month and "
          "want a refund.")
URGENCY_RUBRIC = [
    "low: routine request, handle in the normal queue",
    "medium: annoying but nothing is broken, this week",
    "high: customer blocked or money at risk, today",
    "urgent: churn or legal risk, drop everything",
]
# The stop-6 trio, one question per typed kind over the same ticket.
TYPED_QUESTIONS = {
    "team": {"type": "choice",
             "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, refunds, subscriptions",
                          "technical": "bugs, crashes, outages",
                          "account": "logins and account settings"}},
    "refund": {"type": "noul",
               "instructions": "Is the customer asking for money back?",
               "criteria": {"false": "no money involved",
                            "true": "requests money back"}},
    "urgency": {"type": "score",
                "instructions": "How urgent is this issue?",
                "criteria": URGENCY_RUBRIC},
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


def decide_one(score, text: str, task: str, labels: list[str]) -> dict:
    """One text, one choice field: packed candidates, one pass."""
    question = QUESTION.get(
        task, f"Which {task} category does this text belong to?")
    choice, probs = score(text, task, question, labels)
    return {
        "choice": choice,
        "probabilities": {label: round(prob, 4)
                          for label, prob in zip(labels, probs)},
        "confidence": round(max(probs), 4) if probs else None,
    }


def one_forward_each(score, state: str, questions: dict) -> None:
    """The contrast arm for the answer_typed stop: the same questions
    answered one score() call at a time, one packed forward per question."""
    from engines.mojev_engine import option_texts

    for name, question in questions.items():
        options, _ = option_texts(name, question)
        pick, probs = score(state, name, question.get("instructions", ""),
                            options)
        print(f"  {name:<8} {str(pick)[:56]:<56} p={max(probs):.2f}")


# ------------------------------------------------------------------ runner
def parse_typed_payload(payload: dict) -> tuple[str, dict]:
    """--serve typed mode: {"state", "questions": {name: {"type",
    "instructions", "criteria"}}} -> the (state, questions) pair
    answer_typed takes, with the problem named on bad shapes (criteria
    shapes are validated downstream by option_texts)."""
    state = payload.get("state")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("'state' must be a non-empty string")
    questions = payload.get("questions")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("'questions' must be a non-empty dict of typed "
                         "questions")
    for name, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in (
                "choice", "noul", "score"):
            raise ValueError(f"question {name!r}: 'type' must be 'choice', "
                             "'noul' or 'score'")
        if not isinstance(question.get("instructions", ""), str):
            raise ValueError(f"question {name!r}: 'instructions' must be "
                             "a string")
    return state, questions


def serve() -> None:
    payload = json.loads(sys.stdin.read())
    try:
        if payload.get("mode") == "typed":
            state, questions = parse_typed_payload(payload)
            from engines.mojev_engine import load_typed_engine

            _, answer_typed, _ = load_typed_engine("cpu")
            answers, usage = answer_typed(state, questions)
            # ASCII-escaped JSON survives Windows pipes using legacy code pages.
            print(json.dumps({"answers": answers, "usage": usage}))
            return

        from engines.mojev_engine import load_engine

        score, _ = load_engine("cpu")
        results = [decide_one(score, text, payload["task"],
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

    print("Loading MoLeMo-Lab/mojev (0.85B Qwen3.5 encoder via "
          "trust_remote_code; first run downloads the checkpoint)...")
    from engines.mojev_engine import load_typed_engine

    t0 = time.perf_counter()
    score, answer_typed, _ = load_typed_engine("cpu")
    print(f"  ready in {time.perf_counter() - t0:.1f}s")

    # ---------------------------------------------------------------- 1
    banner("1. One packed pass - every candidate scored at once")
    text = "This laptop has amazing performance but terrible battery life!"
    print(f'  text: "{text}"')
    row = timed(decide_one, score, text, "sentiment", SENTIMENT_LABELS)
    show(row)

    # ---------------------------------------------------------------- 2
    banner("2. Sentiment - the shared sample texts, one call each")
    for text in SHARED_TEXTS:
        row = timed(decide_one, score, text, "sentiment", SENTIMENT_LABELS)
        probs = " ".join(f"{k}={v:.2f}"
                         for k, v in row["probabilities"].items())
        print(f'  {str(row["choice"]):<8} {probs}  "{text[:48]}"')

    # ---------------------------------------------------------------- 3
    banner("3. Topic - four packed candidates")
    text = "The striker scored twice in the final minutes of the match."
    print(f'  text: "{text}"')
    row = timed(decide_one, score, text, "topic", TOPIC_LABELS)
    show(row["probabilities"])

    # ---------------------------------------------------------------- 4
    banner("4. Noul - a yes/no question answered as one P(yes) scalar")
    print(f'  state: "{TICKET}"')
    answers, usage = timed(answer_typed, TICKET,
                           {"refund": TYPED_QUESTIONS["refund"]})
    print(f"  P(yes) = {answers['refund']['noul']:.4f}  "
          f"({usage['input_tokens']} input tokens, one forward)")

    # ---------------------------------------------------------------- 5
    banner("5. Score - an ordered rubric; the expected level falls "
           "between integers")
    text = SHARED_TEXTS[3]
    print(f'  text: "{text}"')
    answers, _ = timed(answer_typed, text,
                       {"urgency": TYPED_QUESTIONS["urgency"]})
    show(answers["urgency"])

    # ---------------------------------------------------------------- 6
    banner("6. Showpiece - one state, three typed questions, ONE packed "
           "forward (answer_typed)")
    print(f'  state: "{TICKET}"')
    answers, usage = timed(answer_typed, TICKET, TYPED_QUESTIONS)
    print(f"  choice + noul + score packed into one sequence "
          f"({usage['input_tokens']} input tokens):")
    for name, row in answers.items():
        if row["type"] == "noul":
            print(f"  {name:<8} P(yes)={row['noul']:.4f}")
        elif row["type"] == "choice":
            probs = " ".join(f"{k}={v:.2f}"
                             for k, v in row["probabilities"].items())
            print(f"  {name:<8} {row['choice']:<9} {probs}")
        else:
            best = max(row["probabilities"],
                       key=row["probabilities"].__getitem__)
            print(f"  {name:<8} expected={row['score']:.3f}  "
                  f"peak level {best} ({row['legend'][best][:40]}...)")
    print("  the same three questions as separate single-choice calls "
          "(three forwards):")
    timed(one_forward_each, score, TICKET, TYPED_QUESTIONS)

    print("\nDone. Same texts through the other decision engines:  "
          ".venv/Scripts/python demos/certo_demo.py / "
          ".venv-von/Scripts/python demos/jevk5_demo.py\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
    else:
        tour()


if __name__ == "__main__":
    main()
