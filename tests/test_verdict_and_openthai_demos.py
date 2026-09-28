"""Offline tests for the Verdict (rlcd) and OpenThai-SystemOne demos.

Nothing here loads a model or needs a server: only the pure helpers
(payload parsing, result rendering) and module import hygiene are
exercised. The Verdict demo imports rlcd only inside its functions, and
the OpenThai demo speaks plain HTTP through engines.jev_client, so both
modules import in the main venv with the heavy stack blocked.

    .venv/Scripts/python -m unittest discover -s tests -p "test_verdict_and_openthai_demos.py"
"""

import os
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demos import openthai_demo as openthai
from demos import verdict_demo as verdict


class ImportTests(unittest.TestCase):
    def test_imports_pull_no_heavy_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            import importlib

            importlib.reload(verdict)
            importlib.reload(openthai)

    def test_noul_semantics_literal_matches_rlcd_contract(self):
        # the rlcd Noul query rejects any other semantics value
        self.assertEqual(verdict.NOUL_SEMANTICS,
                         "conditional_on_sufficient_evidence_v2")


class VerdictResultRowTests(unittest.TestCase):
    def test_choice_score_and_noul_rows_render(self):
        choice = NS(kind="choice", is_abstention=False, selected_id="billing",
                    probabilities={"billing": 0.9})
        score = NS(kind="score", is_abstention=False,
                   selected_level_id="high", selected_value=2.0,
                   expected_score=1.5,
                   probabilities={"high": 0.5})
        noul = NS(kind="noul", is_abstention=False,
                  selected_outcome="true",
                  p_true_given_sufficient_evidence=0.8,
                  p_insufficient_evidence=0.1)
        self.assertEqual(verdict.result_row(choice)["selected"], "billing")
        row = verdict.result_row(score)
        self.assertEqual(row["selected_level"], "high")
        self.assertEqual(row["expected_score"], 1.5)
        row = verdict.result_row(noul)
        self.assertEqual(row["outcome"], "true")
        self.assertEqual(row["p_insufficient_evidence"], 0.1)


class OpenThaiParseTests(unittest.TestCase):
    def test_choice_payload_builds_a_choice_question(self):
        state, question = openthai.parse_serve_payload(
            {"state": "a ticket", "mode": "choice",
             "question": "Which team?",
             "labels": ["billing", "technical"]})
        self.assertEqual(state, "a ticket")
        self.assertEqual(question["type"], "choice")
        self.assertEqual(set(question["criteria"]),
                         {"billing", "technical"})

    def test_score_payload_keeps_rubric_order(self):
        _, question = openthai.parse_serve_payload(
            {"state": "s", "mode": "score", "question": "How urgent?",
             "labels": ["low", "mid", "high"]})
        self.assertEqual(question["type"], "score")
        self.assertEqual(question["criteria"], ["low", "mid", "high"])

    def test_noul_payload_accepts_proposition_or_question(self):
        _, question = openthai.parse_serve_payload(
            {"state": "s", "mode": "noul", "proposition": "Refund?"})
        self.assertEqual(question["type"], "noul")
        _, question = openthai.parse_serve_payload(
            {"state": "s", "mode": "noul", "question": "Refund?"})
        self.assertEqual(question["instructions"], "Refund?")

    def test_bad_payloads_are_rejected_with_named_problems(self):
        for payload in (
                {},  # no state
                {"state": "s"},  # no mode
                {"state": "s", "mode": "rank"},  # unknown mode
                {"state": "s", "mode": "choice", "question": "q",
                 "labels": ["only"]},  # one label
                {"state": "s", "mode": "score", "labels": ["low"],
                 "question": "q"},  # short rubric
                {"state": "s", "mode": "noul"},  # no proposition
        ):
            with self.assertRaises(ValueError):
                openthai.parse_serve_payload(payload)

    def test_tour_questions_cover_the_three_kinds(self):
        questions = {name: build() for name, build in
                     openthai.QUESTIONS.items()}
        self.assertEqual(set(questions), {"team", "urgency", "refund"})
        self.assertEqual(questions["team"]["type"], "choice")
        self.assertEqual(questions["urgency"]["type"], "score")
        self.assertEqual(questions["refund"]["type"], "noul")


if __name__ == "__main__":
    unittest.main()
