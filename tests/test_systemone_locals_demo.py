"""Offline tests for the System One locals demo (kev 0.8B, decider
0.8B, AgentJev 0.6B).

Nothing here loads a model or needs a server: the serve payload
parsing, the agentjev question builders (checked against the shapes
jev_service.contract.prepare() accepts) and the demo's request
assembly are exercised. The demo speaks plain HTTP through
engines.jev_client and engines.agentjev_client, so it imports in the
main venv with the heavy stack blocked.

    .venv/Scripts/python -m unittest discover -s tests -p "test_systemone_locals_demo.py"
"""

import io
import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demos import systemone_locals_demo as demo
from engines import agentjev_client, jev_client


class ImportTests(unittest.TestCase):
    def test_imports_pull_no_heavy_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            import importlib

            importlib.reload(demo)


class AgentJevQuestionRowTests(unittest.TestCase):
    def test_score_row_shape(self):
        row = agentjev_client.score_question("How urgent?",
                                             ["low", "mid", "high"])
        self.assertEqual(row, {"id": "How urgent?", "type": "score",
                               "question": "How urgent?",
                               "levels": ["low", "mid", "high"]})

    def test_boolean_row_without_descriptions_defaults_criteria(self):
        row = agentjev_client.boolean_question("Refund?")
        self.assertEqual(row, {"id": "Refund?", "type": "boolean",
                               "question": "Refund?", "criteria": {}})

    def test_boolean_row_sends_only_described_sides(self):
        row = agentjev_client.boolean_question("Refund?", "money back")
        self.assertEqual(row["criteria"], {"true": "money back"})
        row = agentjev_client.boolean_question("Refund?", "yes side",
                                               "no side")
        self.assertEqual(row["criteria"],
                         {"true": "yes side", "false": "no side"})

    def test_rows_survive_the_contract_validation_rules(self):
        # The parts of jev_service.contract.prepare() a row must
        # survive, restated here so the test stays offline: question
        # text nonempty; score.levels 2..10 ordered distinct
        # descriptions; boolean.criteria an object with optional
        # "true"/"false" text sides whose candidate texts stay distinct
        # (the server defaults a missing side to the bare TRUE/FALSE).
        def check(row):
            self.assertTrue(row["id"] and row["question"])
            self.assertEqual(row["id"], row["question"])
            if row["type"] == "score":
                levels = row["levels"]
                self.assertIsInstance(levels, list)
                self.assertTrue(2 <= len(levels) <= 10)
                self.assertTrue(all(isinstance(l, str) and l.strip()
                                    for l in levels))
                self.assertEqual(len(set(levels)), len(levels))
            elif row["type"] == "boolean":
                criteria = row["criteria"]
                self.assertIsInstance(criteria, dict)
                self.assertTrue(set(criteria) <= {"true", "false"})
                self.assertTrue(all(isinstance(v, str) and v.strip()
                                    for v in criteria.values()))
                candidates = [criteria.get("true", "TRUE"),
                              criteria.get("false", "FALSE")]
                self.assertNotEqual(candidates[0], candidates[1])
            else:
                self.fail(f"unexpected row type {row['type']!r}")

        check(agentjev_client.score_question("q", ["a", "b"]))
        check(agentjev_client.boolean_question("q"))
        check(agentjev_client.boolean_question("q", "yes", "no"))
        # contract ceiling: 10 levels is the most prepare() accepts
        check(agentjev_client.score_question(
            "q", [f"level {i}" for i in range(10)]))


class DemoAssemblyTests(unittest.TestCase):
    def test_systemone_rows_are_score_and_noul(self):
        rows = demo.systemone_rows(demo.URGENCY_QUESTION,
                                   demo.URGENCY_LEVELS,
                                   demo.REFUND_QUESTION)
        self.assertEqual(set(rows), {"urgency", "refund"})
        self.assertEqual(rows["urgency"]["type"], "score")
        self.assertEqual(rows["urgency"]["criteria"], demo.URGENCY_LEVELS)
        self.assertEqual(rows["refund"],
                         jev_client.noul(demo.REFUND_QUESTION))

    def test_agentjev_rows_are_score_and_boolean_keyed_by_text(self):
        rows = demo.agentjev_rows(demo.URGENCY_QUESTION,
                                  demo.URGENCY_LEVELS,
                                  demo.REFUND_QUESTION)
        self.assertEqual(set(rows),
                         {demo.URGENCY_QUESTION, demo.REFUND_QUESTION})
        self.assertEqual(rows[demo.URGENCY_QUESTION]["type"], "score")
        self.assertEqual(rows[demo.URGENCY_QUESTION]["levels"],
                         demo.URGENCY_LEVELS)
        self.assertEqual(rows[demo.REFUND_QUESTION]["type"], "boolean")

    def test_both_families_ask_the_same_two_questions(self):
        # the comparison table only means something if every server saw
        # identical input
        self.assertEqual(len(demo.URGENCY_LEVELS), 4)
        self.assertTrue(demo.URGENCY_QUESTION and demo.REFUND_QUESTION)
        self.assertEqual(demo.ENGLISH_TICKET,
                         "I was charged twice for my subscription this "
                         "month and want a refund.")


class ServeParseTests(unittest.TestCase):
    def test_kev_payload_builds_systemone_rows(self):
        system, state, questions = demo.parse_serve_payload(
            {"system": "kev", "state": "a ticket",
             "score_instructions": "How urgent?",
             "levels": ["low", "mid", "high"],
             "noul_instructions": "Refund?"})
        self.assertEqual((system, state), ("kev", "a ticket"))
        self.assertEqual(questions["urgency"]["type"], "score")
        self.assertEqual(questions["urgency"]["criteria"],
                         ["low", "mid", "high"])
        self.assertEqual(questions["refund"],
                         jev_client.noul("Refund?"))

    def test_decider_takes_the_same_shapes_as_kev(self):
        _, _, questions = demo.parse_serve_payload(
            {"system": "DECIDER", "state": "s",
             "score_instructions": "q", "levels": ["a", "b"],
             "noul_instructions": "p"})
        self.assertEqual(questions["urgency"]["type"], "score")
        self.assertEqual(questions["refund"]["type"], "noul")

    def test_agentjev_payload_builds_agentjev_rows(self):
        system, _, questions = demo.parse_serve_payload(
            {"system": "agentjev", "state": "a ticket",
             "score_instructions": "How urgent?",
             "levels": ["low", "high"],
             "noul_instructions": "Refund?"})
        self.assertEqual(system, "agentjev")
        self.assertEqual(questions["How urgent?"],
                         agentjev_client.score_question(
                             "How urgent?", ["low", "high"]))
        self.assertEqual(questions["Refund?"],
                         agentjev_client.boolean_question("Refund?"))

    def test_bad_payloads_are_rejected_with_named_problems(self):
        cases = (
            ({}, "non-empty 'state'"),
            ({"state": "s"}, "system must be kev, decider or agentjev"),
            ({"state": "s", "system": "gpt"},
             "system must be kev, decider or agentjev"),
            ({"state": "s", "system": "kev"}, "score_instructions"),
            ({"state": "s", "system": "kev",
              "score_instructions": "q"}, "2..10"),
            ({"state": "s", "system": "kev", "score_instructions": "q",
              "levels": ["only"]}, "2..10"),
            ({"state": "s", "system": "kev", "score_instructions": "q",
              "levels": [str(i) for i in range(11)]}, "2..10"),
            ({"state": "s", "system": "kev", "score_instructions": "q",
              "levels": "low,high"}, "'levels' must be a list"),
            ({"state": "s", "system": "agentjev",
              "score_instructions": "q", "levels": ["a", "b"]},
             "noul_instructions"),
        )
        for payload, fragment in cases:
            with self.assertRaises(ValueError) as ctx:
                demo.parse_serve_payload(payload)
            self.assertIn(fragment, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def test_serve_turns_a_bad_payload_into_error_json(self):
        with patch("sys.stdin", io.StringIO("{}")), \
                patch("sys.stdout", io.StringIO()) as out:
            demo.serve()
        payload = json.loads(out.getvalue())
        self.assertEqual(list(payload), ["error"])
        self.assertIn("non-empty 'state'", payload["error"])


if __name__ == "__main__":
    unittest.main()
