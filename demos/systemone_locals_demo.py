"""System One locals demo - kev 0.8B, decider 0.8B and AgentJev 0.6B's
undemoed question kinds (the benches only ever sent choice rows to
them).

Two of the three speak the TypeSafe System One wire format from local
uvicorn servers, so engines.jev_client.JevClient talks to them with no
API key:

  - kev 0.8B (jaredpalmer/kev-0.8b) on port 8009
  - decider 0.8B (Mapika/decider-0.8b) on port 8018

AgentJev 0.6B speaks its own loopback contract (POST /api/evaluate,
engines.agentjev_client) on port 8149; its ready line advertises types
["boolean", "choice", "score"] and jev_service.contract validates all
three. The repo's benches only ever exercised choice on all three
servers. This demo tours the undemoed kinds over one shared English
support ticket:

  1. kev 0.8B - a 4-level urgency score + a refund noul, one request
  2. decider 0.8B - the same two questions on identical input, so the
     two open-weight Jev lookalikes compare directly
  3. AgentJev 0.6B - the same state through a score question (levels)
     and a boolean question, one /api/evaluate call
  4. the six answers side by side

The servers must be up first (commands below); kev and decider
lazy-load their weights on the first request, so every ask() runs with
a 600 s timeout.

Run (main venv; plain HTTP, no model deps needed):
    cd ../kev && uv run --extra serve python -m kev.serve \\
        --run jaredpalmer/kev-0.8b --port 8009
    DECIDER_MODEL=Mapika/decider-0.8b DECIDER_DEVICE=cpu <py> -m uvicorn \\
        decider.serve:app --host 127.0.0.1 --port 8018
    cd ../agent-jev && <python> -m jev_service.server \\
        --checkpoint agentjev_v1.pt --model-path <Qwen3-0.6B snapshot> \\
        --temperatures temperatures.json --port 8149 --device cpu
    .venv/Scripts/python demos/systemone_locals_demo.py

Web-UI runner:
    .venv/Scripts/python demos/systemone_locals_demo.py --serve
    stdin:  {"system": "kev" | "decider" | "agentjev", "state": str,
             "score_instructions": str, "levels": [str, ...],
             "noul_instructions": str}
    stdout: {"answers": ..., "usage": ..., "_latency_s": ...} or
            {"error": "Type: message"}   (usage is null for agentjev -
            the loopback contract reports none)
"""

from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import json
import sys
import time

from engines.agentjev_client import ask, boolean_question, score_question
from engines.jev_client import JevClient, choice, noul, score

KEV_URL = "http://127.0.0.1:8009/v1/systemone"
DECIDER_URL = "http://127.0.0.1:8018/v1/systemone"
AGENTJEV_URL = "http://127.0.0.1:8149/api/evaluate"
SYSTEMONE_URLS = {"kev": KEV_URL, "decider": DECIDER_URL}
# the model field the benches send: decider is versioned
# (decider-0.8b), kev has no versioned id (kev-latest)
MODEL_NAMES = {"kev": "kev-latest", "decider": "decider-0.8b"}

ENGLISH_TICKET = ("I was charged twice for my subscription this month "
                  "and want a refund.")
URGENCY_QUESTION = "How urgent is this ticket?"
REFUND_QUESTION = "Is the customer asking for a refund?"
URGENCY_LEVELS = ["can wait a few days",
                  "should be handled soon",
                  "needs action today",
                  "business-blocking - escalate now"]


def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n  {title}\n{line}")


def show(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def systemone_rows(score_text: str, levels: list[str],
                   noul_text: str) -> dict:
    """The two System One rows; kev and decider take identical input."""
    return {"urgency": score(score_text, list(levels)),
            "refund": noul(noul_text)}


def agentjev_rows(score_text: str, levels: list[str],
                  noul_text: str) -> dict:
    """The two agentjev rows; the answer keys are the row ids, which
    the builders set to the question texts."""
    return {score_text: score_question(score_text, levels),
            noul_text: boolean_question(noul_text)}


def connect_systemone(base_url: str, model: str) -> JevClient:
    """JevClient pointed at a local System One server; the warm-up ask
    pays the lazy weight load (minutes when cold) inside a 600 s
    timeout, exactly like the bench branch."""
    client = JevClient(base_url=base_url, model=model)
    t0 = time.perf_counter()
    client.ask({"task": "warmup"},
               {"w": choice('Sentiment of "good"?',
                            {"positive": None, "negative": None})},
               timeout=600)
    print(f"  warm-up answered in {time.perf_counter() - t0:.1f}s "
          "(the first request pays the lazy weight load)", file=sys.stderr)
    return client


def connect_agentjev() -> None:
    """Warm-up ask; AgentJev loads its checkpoint at server start (no
    lazy load like kev/decider), so this only absorbs first-pass
    overhead. Still 600 s, to be safe."""
    t0 = time.perf_counter()
    ask({"task": "warmup"},
        {"w": {"id": "w", "type": "choice",
               "question": 'Sentiment of "good"?',
               "options": {"positive": "positive sentiment",
                           "negative": "negative sentiment"}}},
        timeout=600)
    print(f"  warm-up answered in {time.perf_counter() - t0:.1f}s",
          file=sys.stderr)


def systemone_stop(name: str, base_url: str, model: str,
                   rows: list) -> None:
    """One System One server asked the urgency score + refund noul;
    raw answers go to stdout, the two picks to the comparison."""
    print(f'  state: "{ENGLISH_TICKET}"')
    client = connect_systemone(base_url, model)
    result = client.ask(ENGLISH_TICKET,
                        systemone_rows(URGENCY_QUESTION, URGENCY_LEVELS,
                                       REFUND_QUESTION),
                        timeout=600)
    show(result["answers"])
    print(f"  usage: {result.get('usage')}  "
          f"latency {result['_latency_s']}s")
    rows.append((name, result["answers"]["urgency"]["score"],
                 result["answers"]["refund"]["noul"]))


# ------------------------------------------------------------------ serve
def parse_serve_payload(payload: dict):
    """stdin payload -> (system, state, questions dict). ValueError
    names the problem; serve() turns it into the error JSON."""
    state = str(payload.get("state") or "").strip()
    if not state:
        raise ValueError("payload needs a non-empty 'state'")
    system = str(payload.get("system") or "").strip().lower()
    if system not in ("kev", "decider", "agentjev"):
        raise ValueError("system must be kev, decider or agentjev")
    score_text = str(payload.get("score_instructions") or "").strip()
    levels = payload.get("levels", [])
    if not isinstance(levels, list):
        raise ValueError("'levels' must be a list of rubric descriptions")
    levels = [str(level).strip() for level in levels if str(level).strip()]
    noul_text = str(payload.get("noul_instructions") or "").strip()
    if not score_text:
        raise ValueError("score needs a 'score_instructions'")
    if not 2 <= len(levels) <= 10:
        raise ValueError("score needs 2..10 'levels' rubric levels, "
                         "low to high")
    if not noul_text:
        raise ValueError("noul needs a 'noul_instructions'")
    if system == "agentjev" and score_text == noul_text:
        raise ValueError("agentjev needs distinct 'score_instructions' "
                         "and 'noul_instructions' (they are the row ids)")
    if system == "agentjev":
        return system, state, agentjev_rows(score_text, levels, noul_text)
    return system, state, systemone_rows(score_text, levels, noul_text)


def serve() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        system, state, questions = parse_serve_payload(payload)
        if system == "agentjev":
            t0 = time.perf_counter()
            answers = ask(state, questions, timeout=600)
            result = {"answers": answers, "usage": None,
                      "_latency_s": round(time.perf_counter() - t0, 3)}
        else:
            client = connect_systemone(SYSTEMONE_URLS[system],
                                       MODEL_NAMES[system])
            result = client.ask(state, questions, timeout=600)
        # ASCII-escaped JSON survives Windows pipes using legacy code pages.
        print(json.dumps({"answers": result["answers"],
                          "usage": result.get("usage"),
                          "_latency_s": result["_latency_s"]}))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))


# -------------------------------------------------------------------- tour
def tour() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    rows = []

    # ---------------------------------------------------------------- 1
    banner("1. Kev 0.8B (:8009) - urgency score + refund noul, "
           "ONE request")
    systemone_stop("kev 0.8B", KEV_URL, "kev-latest", rows)

    # ---------------------------------------------------------------- 2
    banner("2. decider 0.8B (:8018) - identical input, direct lookalike "
           "comparison")
    systemone_stop("decider 0.8B", DECIDER_URL, "decider-0.8b", rows)

    # ---------------------------------------------------------------- 3
    banner("3. AgentJev 0.6B (:8149) - urgency score + refund boolean, "
           "ONE /api/evaluate call")
    print(f'  state: "{ENGLISH_TICKET}"')
    connect_agentjev()
    answers = ask(ENGLISH_TICKET,
                  agentjev_rows(URGENCY_QUESTION, URGENCY_LEVELS,
                                REFUND_QUESTION),
                  timeout=600)
    show(answers)
    rows.append(("AgentJev 0.6B", answers[URGENCY_QUESTION]["score"],
                 answers[REFUND_QUESTION]["probability"]))

    # ---------------------------------------------------------------- 4
    banner("4. Same two questions, three engines side by side")
    print("  urgency = expected score over the 4 rubric levels (0..3),")
    print("  refund P(yes) = kev/decider noul, AgentJev P(true)\n")
    print(f"  {'engine':<14}{'urgency':>10}{'refund P(yes)':>16}")
    for name, urgency, p_yes in rows:
        print(f"  {name:<14}{urgency:>10.2f}{p_yes:>16.3f}")

    print("\nDone. The other System One locals: demos/openthai_demo.py "
          "(OpenThai 0.8B, choice + score + noul over Thai and English) "
          "and demos/demo_jev.py (cloud Jev, same wire format)\n")


def main() -> None:
    if "--serve" in sys.argv:
        serve()
        return
    tour()


if __name__ == "__main__":
    main()
