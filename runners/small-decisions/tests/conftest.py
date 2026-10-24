"""Shared stubs + payload helpers for the small-decisions tests.

Offline by construction: the engines' heavy stacks are stubbed through
sys.modules or by replacing the module-level engine caches, the Ollaya
proxies run against a stubbed HTTP server on an ephemeral port, and the
weight-gated tests skip unless the snapshots are locally present (CI never
downloads weights).
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import Any

import pytest

import small_decisions.models.decision2 as decision2
import small_decisions.models.d3 as d3
import small_decisions.models.intern_decision as intern_decision

# ---------------------------------------------------------------- payloads
# The suites/decision item shape the runner protocol hands predict().
CHOICE = {
    "input": {"text": "The food was cold and the waiter was rude."},
    "question": {"type": "choice", "options": ["positive", "negative"]},
}
NOUL = {
    "input": {"text": "Charged twice, want a refund."},
    "question": {"type": "noul", "instructions": "Is this a refund request?"},
}
SCORE = {
    "input": {"text": "Now."},
    "question": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "mid", "high"]},
}


@pytest.fixture(autouse=True)
def _clear_engine_caches():
    """No loaded (or stubbed) engine may leak between tests."""
    for cache in (decision2._ENGINE, d3._ENGINE, intern_decision._ENGINE_KEYED):
        cache.clear()
    yield
    for cache in (decision2._ENGINE, d3._ENGINE, intern_decision._ENGINE_KEYED):
        cache.clear()


# ------------------------------------------------------- stub HTTP server
class _StubSystemOne(http.server.BaseHTTPRequestHandler):
    """A stand-in Ollaya daemon: records requests, answers one canned
    choice row regardless of the question (the proxies' contract is what
    the tests police — URL building, body shape, answer mapping)."""

    answer = {"type": "choice", "choice": "positive", "probabilities": {"positive": 0.9, "negative": 0.1}}
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 — http.server naming
        length = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        _StubSystemOne.requests.append(body)
        payload = {"model": body.get("model"), "answers": {"q": dict(self.answer)}, "usage": {"input_tokens": 42}}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 — the parent's param name
        pass


@pytest.fixture()
def ollaya_stub(monkeypatch: pytest.MonkeyPatch):
    """A stubbed Ollaya daemon on an ephemeral port; OLLAYA_BASE_URL
    points at it. Yields the request recorder."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubSystemOne)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _StubSystemOne.requests = []
    monkeypatch.setenv("OLLAYA_BASE_URL", f"http://127.0.0.1:{server.server_address[1]}")
    yield _StubSystemOne.requests
    server.shutdown()
    server.server_close()
