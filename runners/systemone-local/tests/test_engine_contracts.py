"""The island's payload translation against the REAL engine code — offline.

The engine packages/clones are installed or cloned on the serving machine
even where the weights are not, and their request-side code is pure: kev's
SystemOneRequest/to_record, decider's systemone.render_question/
format_answer, openthai's types.parse_question and answer models, and the
agent-jev clone's jev_service.contract.prepare. This file runs the island's
normalize_question / agentjev_row / answer_to_result through those real
validators without loading any weights. Each test skips when its engine's
code is not present (fresh clone, no venv) — the same guard style as
test_weights_live.py, on code instead of weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from systemone_local.models import _systemone as s1
from systemone_local.models._base import DEFAULT_AGENTJEV_HOME, DEFAULT_KEV_HOME

SENTIMENT = {"type": "choice", "options": ["positive", "negative", "neutral"]}
LEVELS = ["can wait a few days", "should be handled soon", "needs action today"]


def _with_clone(home: Path) -> None:
    if not home.is_dir():
        pytest.skip(f"engine clone not present at {home}")
    if str(home) not in sys.path:
        sys.path.insert(0, str(home))


class TestKevRequestBuilding:
    def test_suite_question_becomes_a_kev_system_one_request(self):
        _with_clone(DEFAULT_KEV_HOME)
        from kev.api import SystemOneRequest, to_record  # noqa: PLC0415 # pyrefly: ignore[missing-import] # pyrefly: ignore[missing-import] — clone on sys.path

        question = s1.normalize_question(SENTIMENT)
        record, meta = to_record(
            SystemOneRequest(state="I love this.", model="kev-0.8b", questions={"q": question})
        )
        assert meta[0] == {"id": "q", "type": "choice", "keys": ["positive", "negative", "neutral"]}
        assert record["questions"][0]["options"] == ["positive", "negative", "neutral"]

    def test_score_and_noul_option_orders(self):
        _with_clone(DEFAULT_KEV_HOME)
        from kev.api import SystemOneRequest, to_record  # noqa: PLC0415 # pyrefly: ignore[missing-import]

        score = s1.normalize_question({"type": "score", "instructions": "How urgent?", "levels": LEVELS})
        noul = s1.normalize_question({"type": "noul", "instructions": "Refund?"})
        record, meta = to_record(
            SystemOneRequest(state="s", model="kev-0.8b", questions={"a": score, "b": noul})
        )
        assert [m["type"] for m in meta] == ["score", "noul"]
        # kev ranks noul options [no, yes] — kev.predict reads probs[0][1] for p(true)
        assert record["questions"][1]["options"] == ["no", "yes"]

    def test_probs_back_to_a_result(self):
        _with_clone(DEFAULT_KEV_HOME)
        from kev.api import SystemOneRequest, to_record  # noqa: PLC0415 # pyrefly: ignore[missing-import]

        question = s1.normalize_question(SENTIMENT)
        record, meta = to_record(
            SystemOneRequest(state="s", model="kev-0.8b", questions={"q": question})
        )
        row = s1.choice_row(meta[0]["keys"], [0.7, 0.2, 0.1])
        assert s1.answer_to_result("choice", row) == {"prediction": "positive", "prob": 0.7}


class TestDeciderRendering:
    def test_normalized_question_through_render_question(self):
        pytest.importorskip("decider")
        from decider.systemone import format_answer, render_question  # noqa: PLC0415

        question = s1.normalize_question(SENTIMENT)
        rendered = render_question(question)
        assert rendered["question"] == s1.DEFAULT_INSTRUCTIONS
        assert rendered["names"] == ["positive", "negative", "neutral"]
        row = format_answer(rendered, [0.6, 0.3, 0.1])
        assert s1.answer_to_result("choice", row) == {"prediction": "positive", "prob": 0.6}

    def test_score_and_noul_through_render_question(self):
        pytest.importorskip("decider")
        from decider.systemone import format_answer, render_question  # noqa: PLC0415

        score = s1.normalize_question({"type": "score", "instructions": "How urgent?", "levels": LEVELS})
        rendered = render_question(score)
        row = format_answer(rendered, [0.2, 0.3, 0.5])
        result = s1.answer_to_result("score", row)
        assert result["prediction"] == pytest.approx(1.3)  # 0*0.2 + 1*0.3 + 2*0.5

        noul = s1.normalize_question({"type": "noul", "instructions": "Refund?"})
        row = format_answer(render_question(noul), [0.25, 0.75])
        assert s1.answer_to_result("noul", row) == {"prediction": "true", "prob": 0.75}


class TestOpenthaiParsing:
    def test_normalized_question_through_parse_question(self):
        pytest.importorskip("openthai_systemone")
        from openthai_systemone.types import ChoiceAnswer, NoulAnswer, parse_question  # noqa: PLC0415

        question = parse_question(s1.normalize_question(SENTIMENT))
        assert question.type == "choice"
        assert set(question.criteria) == {"positive", "negative", "neutral"}

        answer = ChoiceAnswer(choice="positive", probabilities={"positive": 0.6, "negative": 0.4}, confidence=0.2)
        assert s1.answer_to_result("choice", answer.model_dump()) == {"prediction": "positive", "prob": 0.6}
        noul = NoulAnswer(noul=0.8)
        assert s1.answer_to_result("noul", noul.model_dump())["prediction"] == "true"

    def test_score_question_and_answer(self):
        pytest.importorskip("openthai_systemone")
        from openthai_systemone.types import ScoreAnswer, parse_question  # noqa: PLC0415

        question = parse_question(
            s1.normalize_question({"type": "score", "instructions": "How urgent?", "levels": LEVELS})
        )
        assert question.criteria == LEVELS
        answer = ScoreAnswer(
            score=1.3, legend={0: LEVELS[0], 1: LEVELS[1], 2: LEVELS[2]},
            probabilities={"0": 0.2, "1": 0.3, "2": 0.5}, confidence=0.4,
        )
        result = s1.answer_to_result("score", answer.model_dump())
        assert result["prediction"] == pytest.approx(1.3)
        assert result["prob"] == pytest.approx(0.5)


class TestAgentjevContract:
    """agentjev_row against the REAL jev_service.contract.prepare()."""

    @pytest.mark.parametrize(
        "question",
        [
            SENTIMENT,
            {"type": "choice", "criteria": {"billing": "money", "technical": None}},
            {"type": "score", "instructions": "How urgent?", "levels": ["low", "high"]},
            {"type": "noul", "instructions": "Refund?"},
            {"type": "noul", "instructions": "Refund?", "criteria": {"true": "money back", "false": "no refund"}},
        ],
        ids=["choice-suite", "choice-described", "score", "boolean-defaults", "boolean-described"],
    )
    def test_row_survives_the_real_prepare(self, question):
        _with_clone(DEFAULT_AGENTJEV_HOME)
        from jev_service.contract import prepare  # noqa: PLC0415 # pyrefly: ignore[missing-import] — clone on sys.path

        row = s1.agentjev_row(s1.normalize_question(question))
        prepared = prepare({"state": "a ticket", "questions": [row]})[0]["questions"][0]
        assert prepared["type"] == row["type"]
        assert len(prepared["candidates"]) == len(prepared["keys"])

    def test_answer_row_from_the_real_answer_shape(self):
        _with_clone(DEFAULT_AGENTJEV_HOME)
        from jev_service.contract import answer  # noqa: PLC0415 # pyrefly: ignore[missing-import]

        # the choice distribution the engine emits for ["positive", "negative"]
        row = answer(
            {"id": "q", "type": "choice", "text": "t", "keys": ["positive", "negative"],
             "candidates": ["positive", "negative"]},
            [0.7, 0.3],
        )
        result = s1.answer_to_result("choice", s1.agentjev_answer_row(row))
        assert result == {"prediction": "positive", "prob": 0.7}
        boolean = answer(
            {"id": "q", "type": "boolean", "text": "t", "keys": ["true", "false"],
             "candidates": ["TRUE", "FALSE"]},
            [0.8, 0.2],
        )
        assert s1.answer_to_result("noul", s1.agentjev_answer_row(boolean)) == {
            "prediction": "true",
            "prob": 0.8,
        }
