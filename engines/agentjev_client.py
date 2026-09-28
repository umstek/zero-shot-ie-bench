"""Minimal Python client for a local AgentJev-0.6B server (loopback API).

Upstream: https://github.com/malevrigns/agent-jev (weights
hf:aimeigaoshou/agent-jev). The server speaks its own JSON contract on
POST /api/evaluate — NOT the TypeSafe System One contract, so this is a
separate client rather than another JevClient base_url. One request
carries many typed questions over one shared state; the server's ready
line advertises types ["boolean", "choice", "score"] and
jev_service.contract.prepare() validates all three:

    {"id", "type": "choice",  "question": str,
     "options": {label: description}}              # 2..255 options
    {"id", "type": "boolean", "question": str,
     "criteria": {"true": desc, "false": desc}}    # sides optional - the
                                                   # server defaults a
                                                   # missing side to the
                                                   # bare word TRUE/FALSE
    {"id", "type": "score",   "question": str,
     "levels": [desc, ...]}                        # 2..10, low to high

    -> {"results": [{"answers": [{"id", "type", "distribution", ...}]}]}
    boolean rows add {"probability", "value"} (value = P(true) >= .5),
    score rows add {"score", "level", "legend"} (probability-weighted
    level index), choice rows add {"value", "description",
    "top_probability", "margin"}.

The id doubles as the answer key, so ask() returns {question_id: row};
the question builders below use the question text as the id (ids must
be unique within one call, and distinct questions are distinct texts).

Start the server first (own venv, CPU works — fp32, autocast is CUDA-only):
    cd ../agent-jev
    <python> -m jev_service.server --checkpoint agentjev_v1.pt \\
        --model-path <Qwen3-0.6B snapshot> --temperatures temperatures.json \\
        --port 8149 --device cpu

No third-party deps on purpose: urllib only. Mirrors the upstream
agentjev_client.py (which lives inside the agent-jev clone) so the
benchmark repo stays self-contained.
"""

from __future__ import annotations

import json
import urllib.request

AGENTJEV_URL = "http://127.0.0.1:8149/api/evaluate"


def score_question(question: str, levels: list[str]) -> dict:
    """score row: 2..10 ordered level descriptions, low to high; the
    answer's score is the probability-weighted level index."""
    return {"id": question, "type": "score",
            "question": question, "levels": list(levels)}


def boolean_question(question: str, true_desc: str | None = None,
                     false_desc: str | None = None) -> dict:
    """boolean row: the answer's value is P(true) >= .5. Criterion
    descriptions are optional (the server defaults them to the bare
    words); a side is only sent when described - None values would be
    rejected by the contract's semantic() check."""
    criteria = {side: desc for side, desc in
                (("true", true_desc), ("false", false_desc))
                if desc is not None}
    return {"id": question, "type": "boolean",
            "question": question, "criteria": criteria}


def ask(state, questions: dict, timeout: int = 120) -> dict:
    """One call, many typed questions against one shared state.

    Returns {question_id: {"value": chosen_label,
                           "distribution": {label: prob}}}.
    """
    body = json.dumps({"state": _state_text(state),
                       "questions": list(questions.values())}).encode("utf-8")
    request = urllib.request.Request(
        AGENTJEV_URL, data=body, method="POST",
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    answers = payload["results"][0]["answers"]
    return {ans["id"]: ans for ans in answers}


def _state_text(state) -> str:
    if isinstance(state, (dict, list)):
        return json.dumps(state, ensure_ascii=False, indent=2)
    return str(state)
