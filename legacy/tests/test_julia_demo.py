"""Offline regression tests for demos/julia_demo.py.

No model, no downloads: the module must import without julia/torch (heavy
imports are lazy, the module loads in the main venv too), the pure
request/response mapping functions are exercised against stubs, and the
JULIA registries of both benchmark drivers are checked against the client's
model id (a name drift would silently drop or misroute a system).
"""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demos.julia_demo as julia_demo
from engines import julia_client


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.julia_demo")
        added = set(sys.modules) - before
        self.assertNotIn("julia", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)

    def test_load_refuses_to_run_without_julia(self):
        # the heavy import sits inside load(), not at module level
        with patch.dict(sys.modules, {"julia": None}):
            with self.assertRaises(ImportError):
                julia_demo.load()


class ModelHomeTests(unittest.TestCase):
    def test_default_is_the_snapshot_next_to_the_repo(self):
        self.assertTrue(julia_client.model_home().endswith("Julia-1"))

    def test_julia_home_env_overrides(self):
        with patch.dict(os.environ, {"JULIA_HOME": "D:/models/Julia-1"}):
            self.assertEqual(julia_client.model_home(), "D:/models/Julia-1")

    def test_registry_carries_the_hf_model_id(self):
        self.assertEqual(julia_demo.MODEL_ID, julia_client.MODEL_ID)
        self.assertEqual(julia_client.MODEL_ID, "SupersonicLabs/Julia-1")


class QuestionTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = julia_demo.choice_question("Which team?", {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_describe_labels_uses_the_bench_descriptions(self):
        self.assertEqual(
            julia_demo.describe_labels(["positive", "negative"], "sentiment"),
            {"positive": "Text expresses a clearly positive attitude",
             "negative": "Text expresses a clearly negative attitude"})

    def test_describe_labels_never_leaves_an_empty_description(self):
        # Julia rejects choice criteria without nonempty descriptions
        described = julia_demo.describe_labels(["urgent", "calm"], "mood")
        self.assertEqual(described,
                         {"urgent": "Text whose mood is urgent",
                          "calm": "Text whose mood is calm"})

    def test_question_for_restates_the_text(self):
        # the house shape (see the bench branch comment): text in the
        # state AND in the instructions
        self.assertEqual(
            julia_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            julia_demo.question_for("topic", "cpu benchmarks"),
            'Which topic category does this text belong to: '
            '"cpu benchmarks"')

    def test_question_for_falls_back_to_the_task_word(self):
        self.assertEqual(
            julia_demo.question_for("urgency", "now"),
            'Which urgency category does this text belong to: "now"')


class _StubEngine:
    """Records predict() requests, answers every question alike."""

    def __init__(self):
        self.answer = {"type": "choice", "choice": "positive",
                       "max_probability": 0.9,
                       "probabilities": {"positive": 0.9, "negative": 0.1}}
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answer)
                            for name in questions}}


class DecideOneTests(unittest.TestCase):
    def test_text_is_the_state_and_the_row_maps_choice_probabilities(self):
        engine = _StubEngine()
        row = julia_demo.decide_one(
            engine, "The food was cold.", "sentiment",
            {"positive": "praise", "negative": "complaint"})
        state, questions = engine.calls[0]
        self.assertEqual(state, "The food was cold.")
        question = questions["q"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(
            question["instructions"],
            'What is the overall sentiment of this text: '
            '"The food was cold."')
        self.assertEqual(question["criteria"],
                         {"positive": "praise", "negative": "complaint"})
        # confidence is Julia's max_probability (full softmax, no separate
        # confidence field)
        self.assertEqual(row, {"choice": "positive",
                               "probabilities": {"positive": 0.9,
                                                 "negative": 0.1},
                               "confidence": 0.9})

    def test_missing_answer_fields_map_to_none(self):
        engine = _StubEngine()
        engine.answer = {"type": "choice"}
        row = julia_demo.decide_one(engine, "x", "sentiment",
                                    {"positive": "p", "negative": "n"})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels(self):
        parsed = julia_demo.parse_serve_payload(
            {"texts": ["a", "", "b"], "task": "sentiment",
             "labels": ["positive", "negative"]})
        self.assertEqual(parsed, (["a", "b"], "sentiment",
                                  ["positive", "negative"]))

    def test_empty_or_missing_fields_raise_with_the_field_named(self):
        for payload, field in (({}, "texts"),
                               ({"texts": ["", "  "]}, "texts"),
                               ({"texts": ["a"]}, "task"),
                               ({"texts": ["a"], "task": "sentiment"},
                                "labels")):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError) as ctx:
                    julia_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict):
        stub = _StubEngine()
        stdout = io.StringIO()
        with patch.object(julia_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            julia_demo.serve()
        return json.loads(stdout.getvalue()), stub

    def test_serve_answers_every_text_in_the_wire_contract(self):
        payload, stub = self.run_serve(
            {"texts": ["a", "b"], "task": "sentiment",
             "labels": ["positive", "negative"]})
        self.assertNotIn("error", payload)
        self.assertEqual([row["choice"] for row in payload["results"]],
                         ["positive", "positive"])
        # one model load per click, one predict() per text
        self.assertEqual(len(stub.calls), 2)

    def test_serve_maps_errors_to_the_error_json(self):
        payload, _ = self.run_serve({"texts": [], "task": "sentiment",
                                     "labels": ["positive"]})
        self.assertIn("error", payload)
        self.assertTrue(payload["error"].startswith("ValueError:"),
                        payload["error"])

    def test_serve_reports_malformed_stdin_as_the_error_json(self):
        # stdin is decoded inside the try: malformed input follows the
        # documented error-JSON path instead of crashing with a traceback
        stdout = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("not json")), \
                contextlib.redirect_stdout(stdout):
            julia_demo.serve()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["error"].startswith("JSONDecodeError:"),
                        payload["error"])


class RegistryTests(unittest.TestCase):
    def test_julia_is_wired_but_culled_from_the_roster(self):
        import bench_multilingual
        import bench_spectrum

        self.assertIn("Julia 1 144M", bench_multilingual.JULIA)
        self.assertNotIn("Julia 1 144M", bench_spectrum.ALL_SYSTEMS)
        self.assertNotIn("Julia 1 144M", bench_multilingual.ALL_SYSTEMS)

    def test_registry_entries_share_the_model_id(self):
        import bench_multilingual
        import bench_spectrum

        # both drivers point at the same local snapshot
        self.assertEqual(bench_spectrum.JULIA["Julia 1 144M"],
                         julia_client.MODEL_ID)
        self.assertEqual(bench_multilingual.JULIA["Julia 1 144M"],
                         julia_client.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
