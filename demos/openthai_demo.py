"""OpenThai-SystemOne demo - the Thai/English decision engine's score
and noul questions, which the benches never ask (choice-only rows).

OpenThai-SystemOne (iapp technology; Apache-2.0 package, weights gated
on HF) serves the same System One wire format as Jev
(POST /v1/systemone {model, state, questions}) from a local uvicorn
server, so engines.jev_client.JevClient talks to it directly. The card
documents all three question kinds - choice (up to 255 options), score
(2-10 ordered level descriptions, probability-weighted fractional
answer) and noul (probability of yes) - over Thai and English states;
the repo's benches only ever sent choice questions (the feature table
flags score "(untested here)"), and nothing demos a Thai state at all.
This demo does both:

  1. one ask() answering a choice + a score + a noul question over a
     Thai complaint (the model's home turf), and
  2. the same three kinds over the shared English support ticket.

The server must be up first (README setup): the first request pays the
lazy weight load, so ask() runs with a 600 s timeout.

Run (main venv; plain HTTP, no model deps needed):
    OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne <agent-jev venv
    python> -m uvicorn openthai_systemone.server:app --port 8029
    .venv/Scripts/python demos/openthai_demo.py

Web-UI runner:
    .venv/Scripts/python demos/openthai_demo.py --serve
    stdin:  {"state": str, "mode": "choice" | "score" | "noul",
             "labels": [str, ...], "question" | "proposition": str}
    stdout: {"answer": {...}} or {"error": "Type: message"}
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import sys
import time

from engines.jev_client import JevClient, choice, noul, score

OPENTHAI_URL = "http://127.0.0.1:8029/v1/systemone"

THAI_TICKET = ("พนักงานไม่สุภาพและอาหารมาช้า ฉันอยากได้เงินคืน "
               "(the staff were rude and the food was late - I want a "
               "refund)")
ENGLISH_TICKET = ("I was charged twice for my subscription this month "
                  "and want a refund.")

QUESTIONS = {
    "team": lambda: choice("Which team should handle this ticket?", {
        "billing": "Payments, invoices and refunds",
        "technical": "Bugs, errors and outages",
        "account": "Login, profile and settings"}),
    "urgency": lambda: score("How urgent is this ticket?",
                             ["could wait a few days",
                              "should be fixed soon",
                              "needs immediate action"]),
    "refund": lambda: noul("Is the customer asking for a refund?"),
}


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def connect():
    """JevClient pointed at the local OpenThai server; the warm-up ask
    pays the lazy weight load (minutes when cold) inside a 600 s
    timeout, exactly like the bench branch."""
    client = JevClient(base_url=OPENTHAI_URL, model="openthai-latest")
    t0 = time.perf_counter()
    client.ask({"task": "warmup"},
               {"w": choice('Sentiment of "good"?',
                            {"positive": None, "negative": None})},
               timeout=600)
    print(f"  warm-up answered in {time.perf_counter() - t0:.1f}s "
          "(the first request pays the lazy weight load)", file=sys.stderr)
    return client


def answer_lines(answers: dict) -> list[str]:
    """The three answers as printable lines (one per question kind)."""
    return [f"  team    -> {answers['team'].get('choice')} "
            f"(confidence {answers['team'].get('confidence', 0):.2f})",
            f"  urgency -> score {answers['urgency'].get('score')} "
            f"over {len(answers['urgency'].get('legend', {})) or 3} levels",
            f"  refund? -> yes-probability {answers['refund'].get('noul')}"]


# ------------------------------------------------------------------ serve
def parse_serve_payload(payload: dict):
    """stdin payload -> (state, one System One question dict). ValueError
    names the problem; serve() turns it into the error JSON."""
    state = str(payload.get("state") or "").strip()
    if not state:
        raise ValueError("payload needs a non-empty 'state'")
    mode = str(payload.get("mode") or "").strip().lower()
    labels = [str(label).strip() for label in payload.get("labels", [])
              if str(label).strip()]
    question = str(payload.get("proposition")
                   or payload.get("question") or "").strip()
    if mode == "choice":
        if len(labels) < 2:
            raise ValueError("choice needs at least two 'labels'")
        if not question:
            raise ValueError("choice needs a 'question'")
        return state, choice(question, {label: None for label in labels})
    if mode == "score":
        if len(labels) < 2:
            raise ValueError("score needs at least two 'labels' rubric "
                             "levels, low to high")
        if not question:
            raise ValueError("score needs a 'question'")
        return state, score(question, labels)
    if mode == "noul":
        if not question:
            raise ValueError("noul needs a 'proposition'")
        return state, noul(question)
    raise ValueError("mode must be choice, score or noul")


def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        state, question = parse_serve_payload(payload)
        client = connect()
        result = client.ask(state, {"q": question}, timeout=600)
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"answer": result["answers"]["q"]}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print(f"Connecting to OpenThai-SystemOne at {OPENTHAI_URL} ...")
    client = connect()

    # ---------------------------------------------------------------- 1
    banner("1. Thai state, all three question kinds, ONE request")
    print(f'  state: "{THAI_TICKET}"')
    result = client.ask(THAI_TICKET, {name: build()
                                      for name, build in QUESTIONS.items()},
                        timeout=600)
    for line in answer_lines(result["answers"]):
        print(line)

    # ---------------------------------------------------------------- 2
    banner("2. Same questions over the shared English ticket")
    print(f'  state: "{ENGLISH_TICKET}"')
    result = client.ask(ENGLISH_TICKET, {name: build()
                                         for name, build in QUESTIONS.items()},
                        timeout=600)
    for line in answer_lines(result["answers"]):
        print(line)

    print("\nDone. The benches' choice-only OpenThai rows and the other "
          "System One locals: demos/demo_jev.py (cloud Jev, same wire "
          "format)\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    tour()


if __name__ == "__main__":
    main()
