"""The weight-gated end-to-end tests: real snapshots, real loads.

Skipped unless the snapshot is locally present (available() never touches
the network, so CI never downloads weights). Marked slow (docs/plan/02:
"tests that load real weights are marked slow and run locally on a machine
that can hold the model") — run with `uv run pytest -m slow`.

Snapshot homes (seed once, then these run offline):
    DECISION2_HOME / D3_HOME / INTERN_DECISION_HOME prefixes at C:\src,
    K2TYPE_HOME at C:\src\K2-Type-0.9B (the defaults).r"""

from __future__ import annotations

import pytest

import small_decisions.models.decision2 as decision2
import small_decisions.models.intern_decision as intern_decision
import small_decisions.models.k2type as k2type
from conftest import CHOICE

pytestmark = pytest.mark.slow

SENTIMENT = {
    "input": {"text": "This is the best laptop I have ever owned."},
    "question": {"type": "choice",
                 "instructions": "What is the overall sentiment of this text?",
                 "criteria": {"positive": "praise", "negative": "complaint"}},
}


def _require(condition: bool, where: str):
    if not condition:
        pytest.skip(f"the {where} snapshot is not locally present")


def test_decision2_kai_loads_and_answers():
    _require(decision2.available("kai-0.6b"), "Decision-2.0-Kai-0.6B")
    result = decision2.predict("vllm-sr/decision-2.0-kai", SENTIMENT)
    assert result["prediction"] in ("positive", "negative")
    assert 0.0 <= (result["prob"] or 0.0) <= 1.0


def test_k2type_loads_and_answers():
    _require(k2type.available(), "K2-Type-0.9B")
    result = k2type.predict(SENTIMENT)
    assert result["prediction"] in ("positive", "negative")


def test_intern_decision_08b_loads_and_answers():
    _require(intern_decision.available("0.8B"), "Intern-Decision-0.8B")
    result = intern_decision.predict("internlm/intern-decision-0.8b", SENTIMENT)
    assert result["prediction"] in ("positive", "negative")


def test_ollaya_daemon_answers_when_it_is_up(ollaya_stub):
    """ollaya_stub must be requested for the URL env; a real-daemon check
    would need the daemon — this test runs against the stub to keep the
    slow suite daemon-free (the stub exercises the same HTTP path)."""
    from small_decisions.models import ollaya

    result = ollaya.predict("vllm-sr/decision", CHOICE)
    assert result["prediction"] == "positive"
