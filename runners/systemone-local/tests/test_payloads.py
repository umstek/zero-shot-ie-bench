"""Offline tests for the island's payload translation — no weights, no
server, no torch. Ports the relevant pieces of legacy/tests
test_systemone_locals_demo.py (the AgentJev row shapes checked against
the rules jev_service.contract.prepare() enforces) and
test_verdict_and_openthai_demos.py (the System One payload parsing the
OpenThai demo exercised), adapted from demo payload dicts to the runner
payload the island actually receives ({input, question}).
"""

from __future__ import annotations

import pytest
from systemone_local.models import _systemone as s1

ENGLISH_TICKET = "I was charged twice for this month's subscription."
URGENCY_LEVELS = [
    "can wait a few days",
    "should be handled soon",
    "needs action today",
    "business-blocking - escalate now",
]


# --------------------------------------------------------------- payload side
class TestStateText:
    def test_suite_input_shape(self):
        assert s1.state_text({"text": ENGLISH_TICKET, "images": []}) == ENGLISH_TICKET

    def test_bare_string_and_rejections(self):
        assert s1.state_text(ENGLISH_TICKET) == ENGLISH_TICKET
        with pytest.raises(ValueError, match="input"):
            s1.state_text(None)
        with pytest.raises(ValueError, match="input"):
            s1.state_text({"images": []})


class TestNormalizeQuestion:
    def test_suite_choice_shape_is_the_system_one_shape(self):
        question = s1.normalize_question({"type": "choice", "options": ["positive", "negative", "neutral"]})
        assert question == {
            "type": "choice",
            "instructions": s1.DEFAULT_INSTRUCTIONS,
            "criteria": {"positive": None, "negative": None, "neutral": None},
        }

    def test_described_options_and_own_instructions_pass_through(self):
        question = s1.normalize_question(
            {
                "type": "choice",
                "instructions": "Which team owns this?",
                "criteria": {"billing": "money questions", "technical": "bugs"},
            }
        )
        assert question == {
            "type": "choice",
            "instructions": "Which team owns this?",
            "criteria": {"billing": "money questions", "technical": "bugs"},
        }

    def test_agentjev_question_key_is_understood_as_instructions(self):
        question = s1.normalize_question({"type": "choice", "question": "Refund?", "options": ["yes", "no"]})
        assert question["instructions"] == "Refund?"

    def test_score_keeps_rubric_order(self):
        question = s1.normalize_question({"type": "score", "instructions": "How urgent?", "levels": URGENCY_LEVELS})
        assert question == {"type": "score", "instructions": "How urgent?", "criteria": URGENCY_LEVELS}

    def test_score_legend_dict_is_read_low_to_high(self):
        question = s1.normalize_question({"type": "score", "criteria": {"1": "low", "0": "none", "2": "high"}})
        assert question["criteria"] == ["none", "low", "high"]

    def test_noul_and_its_bool_alias(self):
        for qtype in ("noul", "bool"):
            question = s1.normalize_question(
                {"type": qtype, "instructions": "Refund?", "criteria": {"true": "money back"}}
            )
            assert question == {
                "type": "noul",
                "instructions": "Refund?",
                "criteria": {"true": "money back"},
            }

    def test_bad_questions_are_rejected_with_named_problems(self):
        with pytest.raises(ValueError, match="options"):
            s1.normalize_question({"type": "choice", "options": ["only"]})
        with pytest.raises(ValueError, match="2..255"):
            s1.normalize_question({"type": "choice", "options": [str(i) for i in range(256)]})
        with pytest.raises(ValueError, match="2..10"):
            s1.normalize_question({"type": "score", "levels": ["low"]})
        with pytest.raises(ValueError, match="2..10"):
            s1.normalize_question({"type": "score", "levels": [str(i) for i in range(11)]})
        with pytest.raises(ValueError, match="criteria"):
            s1.normalize_question({"type": "noul", "instructions": "p", "criteria": ["no", "yes"]})
        with pytest.raises(ValueError, match="question"):
            s1.normalize_question({"type": "choice"})
        with pytest.raises(ValueError, match="unknown question type"):
            s1.normalize_question({"type": "rank", "options": ["a", "b"]})


# ---------------------------------------------------------------- answer side
class TestAnswerToResult:
    def test_choice_row(self):
        row = s1.choice_row(["positive", "negative"], [0.7, 0.3])
        assert s1.answer_to_result("choice", row) == {"prediction": "positive", "prob": 0.7}

    def test_noul_row_reports_the_predicted_side(self):
        # the decision track's prob is the probability OF THE PREDICTION
        yes = s1.answer_to_result("noul", {"type": "noul", "noul": 0.8})
        assert yes == {"prediction": "true", "prob": 0.8}
        flipped = s1.answer_to_result("noul", {"type": "noul", "noul": 0.2})
        assert flipped == {"prediction": "false", "prob": 0.8}

    def test_score_row_expected_level_and_modal_confidence(self):
        row = s1.score_row([0.1, 0.2, 0.3, 0.4])
        result = s1.answer_to_result("score", row)
        assert result["prediction"] == pytest.approx(0.2 + 0.6 + 1.2)
        assert result["prob"] == pytest.approx(0.4)

    def test_wire_rows_from_the_packages_extract_the_same_way(self):
        # decider rounds to 4 and kev's to_answers to 2 — both survive the
        # extractor; the island's kev path builds unrounded rows instead.
        decider_style = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.1},
            "confidence": 0.8,
        }
        assert s1.answer_to_result("choice", decider_style) == {"prediction": "billing", "prob": 0.9}
        openthai_style = {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.6, "technical": 0.4},
            "confidence": 0.2,
            "abstain": 0.01,
        }
        assert s1.answer_to_result("choice", openthai_style)["prediction"] == "billing"

    def test_malformed_rows_are_named_errors(self):
        with pytest.raises(ValueError, match="probabilities"):
            s1.choice_row(["a", "b"], [1.0])
        with pytest.raises(ValueError, match="unrecognized"):
            s1.answer_to_result("choice", {"type": "mystery"})


# --------------------------------------------------- AgentJev contract rows
class TestAgentJevRows:
    """The rules jev_service.contract.prepare() enforces, restated offline
    exactly as legacy/tests/test_systemone_locals_demo.py did."""

    @staticmethod
    def check_contract(row):
        assert row["id"] and row["question"]
        assert row["id"] == "q"
        if row["type"] == "score":
            levels = row["levels"]
            assert isinstance(levels, list) and 2 <= len(levels) <= 10
            assert all(isinstance(level, str) and level.strip() for level in levels)
            assert len(set(levels)) == len(levels)
        elif row["type"] == "boolean":
            criteria = row["criteria"]
            assert isinstance(criteria, dict) and set(criteria) <= {"true", "false"}
            assert all(isinstance(v, str) and v.strip() for v in criteria.values())
            candidates = [criteria.get("true", "TRUE"), criteria.get("false", "FALSE")]
            assert candidates[0] != candidates[1]
        else:
            assert row["type"] == "choice"
            options = row["options"]
            assert isinstance(options, dict) and 2 <= len(options) <= 255
            assert all(
                isinstance(k, str) and k and isinstance(v, str) and v.strip()
                for k, v in options.items()
            )
            assert len(set(options.values())) == len(options)

    def test_choice_row_from_the_suite_shape_survives_the_contract(self):
        question = s1.normalize_question({"type": "choice", "options": ["positive", "negative", "neutral"]})
        row = s1.agentjev_row(question)
        self.check_contract(row)
        # a None description becomes the bare label (semantic() rejects None)
        assert row["options"] == {"positive": "positive", "negative": "negative", "neutral": "neutral"}

    def test_choice_row_descriptions(self):
        question = s1.normalize_question({"type": "choice", "criteria": {"billing": "money", "technical": None}})
        assert s1.agentjev_row(question)["options"] == {"billing": "money", "technical": "technical"}

    def test_score_row_shape_and_ten_level_ceiling(self):
        row = s1.agentjev_row(
            s1.normalize_question({"type": "score", "instructions": "How urgent?", "levels": ["low", "high"]})
        )
        assert row == {"id": "q", "type": "score", "question": "How urgent?", "levels": ["low", "high"]}
        self.check_contract(row)
        ceiling = s1.agentjev_row(
            s1.normalize_question({"type": "score", "criteria": [f"level {i}" for i in range(10)]})
        )
        self.check_contract(ceiling)

    def test_boolean_row_defaults_criteria_to_the_bare_words(self):
        row = s1.agentjev_row(s1.normalize_question({"type": "noul", "instructions": "Refund?"}))
        assert row == {"id": "q", "type": "boolean", "question": "Refund?", "criteria": {}}
        self.check_contract(row)
        described = s1.agentjev_row(
            s1.normalize_question({"type": "noul", "instructions": "Refund?", "criteria": {"true": "money back"}})
        )
        assert described["criteria"] == {"true": "money back"}

    def test_answer_row_back_to_system_one(self):
        assert s1.agentjev_answer_row({"type": "boolean", "probability": 0.8, "value": True}) == {
            "type": "noul",
            "noul": 0.8,
        }
        choice = s1.agentjev_answer_row(
            {"type": "choice", "value": "billing", "distribution": {"billing": 0.6, "technical": 0.4}}
        )
        assert s1.answer_to_result("choice", choice) == {"prediction": "billing", "prob": 0.6}
        score = s1.agentjev_answer_row(
            {"type": "score", "score": 1.5, "level": 2, "distribution": {"0": 0.0, "1": 0.5, "2": 0.5}}
        )
        result = s1.answer_to_result("score", score)
        assert result["prediction"] == pytest.approx(1.5)
        assert result["prob"] == pytest.approx(0.5)
        with pytest.raises(ValueError, match="unrecognized"):
            s1.agentjev_answer_row({"type": "box"})


# ------------------------------------------------------------- availability
class TestRegistry:
    def test_every_engine_reports_availability_without_touching_the_network(self):
        from systemone_local.models import agentjev, decider, kev, openthai

        # On a machine without clones/weights these are simply False; either
        # way they must answer fast and offline (find_spec + cache scan).
        for available in (kev.available, decider.available, openthai.available, agentjev.available):
            assert isinstance(available(), bool)

    def test_registry_covers_every_served_model(self):
        from systemone_local.models import MODELS, predict

        assert [m["model_id"] for m in MODELS] == [
            "kev-0.8b",
            "decider-0.8b",
            "openthai-systemone",
            "agentjev-v1",
        ]
        with pytest.raises(KeyError, match="unknown model_id"):
            predict("nope", {})
