"""Live weight tests — one real question per engine, skipped on absence.

Marked slow and guarded by each engine's available() (the same offline
check the island itself uses: env-var homes + HF-cache snapshot scan, no
network), so `uv run pytest` is green anywhere and CI never downloads
weights (docs/plan/02: stub models in CI, real weights locally).

On this repo's machine the historical serving put kev-0.8b, decider-0.8b
and Qwen3-0.6B into the HF cache and agentjev_v1.pt into the agent-jev
clone; caches get cleaned, so skips are the normal outcome — set
KEV_WEIGHTS / DECIDER_WEIGHTS / OPENTHAI_WEIGHTS / AGENTJEV_* to run them.
"""

from __future__ import annotations

import pytest
from systemone_local.models import _systemone as s1
from systemone_local.models import agentjev, decider, kev, openthai

# the systemone_locals_demo comparison ticket + sentiment labels
TICKET = "I was charged twice for my subscription this month and want a refund."
SENTIMENT = {"type": "choice", "options": ["positive", "negative", "neutral"]}


def decision_payload(question):
    return {"input": {"text": TICKET, "images": []}, "question": question}


def assert_choice_shape(result, options):
    assert set(result) == {"prediction", "prob"}
    assert result["prediction"] in options
    assert result["prob"] is None or 0.0 <= result["prob"] <= 1.0


@pytest.mark.slow
@pytest.mark.skipif(not kev.available(), reason="kev-0.8b weights not locally resolvable (KEV_WEIGHTS/KEV_RUN)")
def test_kev_live_choice():
    result = kev.predict(decision_payload(SENTIMENT))
    assert_choice_shape(result, SENTIMENT["options"])


@pytest.mark.slow
@pytest.mark.skipif(
    not decider.available(), reason="decider-0.8b weights not in the HF cache (DECIDER_WEIGHTS)"
)
def test_decider_live_choice():
    result = decider.predict(decision_payload(SENTIMENT))
    assert_choice_shape(result, SENTIMENT["options"])


@pytest.mark.slow
@pytest.mark.skipif(
    not openthai.available(), reason="OpenThai weights not in the HF cache (gated; OPENTHAI_WEIGHTS)"
)
def test_openthai_live_choice():
    result = openthai.predict(decision_payload(SENTIMENT))
    assert_choice_shape(result, SENTIMENT["options"])


@pytest.mark.slow
@pytest.mark.skipif(
    not agentjev.available(), reason="agentjev_v1.pt / Qwen3-0.6B snapshot not present (AGENTJEV_*)"
)
def test_agentjev_live_choice_and_noul():
    result = agentjev.predict(decision_payload(SENTIMENT))
    assert_choice_shape(result, SENTIMENT["options"])
    question = s1.normalize_question({"type": "noul", "instructions": "Is the customer asking for a refund?"})
    refund = agentjev.predict(decision_payload(question))
    assert refund["prediction"] in ("true", "false")
    assert refund["prob"] is None or 0.0 <= refund["prob"] <= 1.0
