"""Minimal Python client for a local AgentJev-0.6B server (loopback API).

Upstream: https://github.com/malevrigns/agent-jev (weights
hf:aimeigaoshou/agent-jev). The server speaks its own JSON contract on
POST /api/evaluate — NOT the TypeSafe System One contract, so this is a
separate client rather than another JevClient base_url:

    {"state": str, "questions": [{"id", "type": "choice",
                                  "question": str,
                                  "options": {label: description}}]}
    -> {"results": [{"answers": [{"value", "distribution", ...}]}], ...}

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
