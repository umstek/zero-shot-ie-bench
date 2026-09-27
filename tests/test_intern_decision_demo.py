"""Offline regression tests for demos/intern_decision_demo.py.

No model, no downloads: the module must import without torch/
transformers (heavy imports are lazy, the module loads in the main venv
too), the pure request/response mapping functions are exercised against
stubs, and the INTERN_DECISION registries of both benchmark drivers are
checked against the client's size registry (a name drift would silently
drop or misroute a system).
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

import demos.intern_decision_demo as intern_demo
from engines import intern_decision_client


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.intern_decision_demo")
        added = set(sys.modules) - before
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)
        self.assertNotIn("inference", added)

    def test_load_refuses_to_run_without_the_snapshot(self):
        # the heavy import sits inside load(), not at module level; a
        # missing snapshot fails fast with the env var named
        with patch.dict(os.environ, {"INTERN_DECISION_HOME": "D:/nowhere"}):
            with self.assertRaises(FileNotFoundError) as ctx:
                intern_demo.load()
        self.assertIn("INTERN_DECISION_HOME", str(ctx.exception))


class SizeHomeTests(unittest.TestCase):
    def test_default_homes_are_the_c_src_snapshots(self):
        self.assertTrue(
            intern_decision_client.size_home("0.8B").endswith(
                "Intern-Decision-0.8B"))
        self.assertTrue(
            intern_decision_client.size_home("4B").endswith(
                "Intern-Decision-4B"))

    def test_intern_decision_home_env_is_a_prefix(self):
        with patch.dict(os.environ,
                        {"INTERN_DECISION_HOME": "D:/models"}):
            self.assertEqual(
                intern_decision_client.size_home("2B"),
                os.path.abspath(os.path.join("D:/models",
                                             "Intern-Decision-2B")))

    def test_registry_carries_the_three_hf_checkpoint_ids(self):
        self.assertEqual(set(intern_decision_client.SIZES),
                         {"0.8B", "2B", "4B"})
        self.assertEqual(intern_decision_client.SIZES["0.8B"],
                         "internlm/Intern-Decision-0.8B")
        self.assertEqual(intern_decision_client.DEFAULT_SIZE, "0.8B")

    def test_normalize_size_accepts_shorthand_hf_ids_and_none(self):
        self.assertEqual(intern_decision_client.normalize_size(None), "0.8B")
        self.assertEqual(intern_decision_client.normalize_size("2b"), "2B")
        self.assertEqual(
            intern_decision_client.normalize_size(
                "internlm/Intern-Decision-4B"), "4B")
        with self.assertRaises(ValueError):
            intern_decision_client.normalize_size("9b")


class QuestionTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = intern_demo.choice_question("Which team?", {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_describe_labels_uses_the_bench_descriptions(self):
        self.assertEqual(
            intern_demo.describe_labels(["positive", "negative"], "sentiment"),
            {"positive": "Text expresses a clearly positive attitude",
             "negative": "Text expresses a clearly negative attitude"})

    def test_describe_labels_falls_back_to_a_task_phrase(self):
        described = intern_demo.describe_labels(["urgent", "calm"], "mood")
        self.assertEqual(described,
                         {"urgent": "Text whose mood is urgent",
                          "calm": "Text whose mood is calm"})

    def test_question_for_restates_the_text(self):
        # the house shape (see the bench branch comment): text in the
        # state AND in the instructions
        self.assertEqual(
            intern_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            intern_demo.question_for("topic", "cpu benchmarks"),
            'Which topic category does this text belong to: '
            '"cpu benchmarks"')

    def test_question_for_falls_back_to_the_task_word(self):
        self.assertEqual(
            intern_demo.question_for("urgency", "now"),
            'Which urgency category does this text belong to: "now"')


class _StubEngine:
    """Records predict() requests, answers every question alike."""

    def __init__(self):
        self.answer = {"type": "choice", "choice": "positive",
                       "confidence": 0.9,
                       "probabilities": {"positive": 0.9, "negative": 0.1}}
        self.calls = []

    def predict(self, request):
        self.calls.append(request)
        return {"answers": {name: dict(self.answer)
                            for name in request["questions"]}}


class DecideOneTests(unittest.TestCase):
    def test_state_and_question_travel_in_one_request_dict(self):
        # unlike Lumma's decide(state, questions), Intern-Decision takes
        # ONE request dict: {"state": ..., "questions": {...}}
        engine = _StubEngine()
        row = intern_demo.decide_one(
            engine, "The food was cold.", "sentiment",
            {"positive": "praise", "negative": "complaint"})
        request = engine.calls[0]
        self.assertEqual(request["state"], "The food was cold.")
        question = request["questions"]["q"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(
            question["instructions"],
            'What is the overall sentiment of this text: '
            '"The food was cold."')
        self.assertEqual(question["criteria"],
                         {"positive": "praise", "negative": "complaint"})
        # the answer already carries choice, probabilities, confidence
        self.assertEqual(row, {"choice": "positive",
                               "probabilities": {"positive": 0.9,
                                                 "negative": 0.1},
                               "confidence": 0.9})

    def test_missing_answer_fields_map_to_none(self):
        engine = _StubEngine()
        engine.answer = {"type": "choice"}
        row = intern_demo.decide_one(engine, "x", "sentiment",
                                     {"positive": "p", "negative": "n"})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels_and_size(self):
        parsed = intern_demo.parse_serve_payload(
            {"texts": ["a", "", "b"], "task": "sentiment",
             "labels": ["positive", "negative"], "model": "2B"})
        self.assertEqual(parsed, (["a", "b"], "sentiment",
                                  ["positive", "negative"], "2B"))

    def test_model_is_optional_and_defaults_to_0_8b(self):
        parsed = intern_demo.parse_serve_payload(
            {"texts": ["a"], "task": "sentiment",
             "labels": ["positive", "negative"]})
        self.assertEqual(parsed[3], "0.8B")

    def test_empty_or_missing_fields_raise_with_the_field_named(self):
        for payload, field in (({}, "texts"),
                               ({"texts": ["", "  "]}, "texts"),
                               ({"texts": ["a"]}, "task"),
                               ({"texts": ["a"], "task": "sentiment"},
                                "labels")):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError) as ctx:
                    intern_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))

    def test_unknown_model_key_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            intern_demo.parse_serve_payload(
                {"texts": ["a"], "task": "sentiment",
                 "labels": ["positive", "negative"], "model": "9b"})
        self.assertIn("9b", str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict):
        stub = _StubEngine()
        stdout = io.StringIO()
        with patch.object(intern_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            intern_demo.serve()
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
            intern_demo.serve()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["error"].startswith("JSONDecodeError:"),
                        payload["error"])


class RegistryTests(unittest.TestCase):
    def test_intern_decision_is_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for system in ("Intern-Decision 0.8B", "Intern-Decision 2B",
                       "Intern-Decision 4B"):
            self.assertIn(system, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(system, bench_multilingual.ALL_SYSTEMS)
            self.assertIn(system, bench_multilingual.INTERN_DECISION)

    def test_registry_entries_share_the_client_size_keys(self):
        import bench_multilingual
        import bench_spectrum

        # all three drivers point at the same local snapshots
        for registry in (bench_spectrum.INTERN_DECISION,
                         bench_multilingual.INTERN_DECISION):
            self.assertEqual(set(registry.values()),
                             set(intern_decision_client.SIZES))
            self.assertEqual(
                list(registry),
                ["Intern-Decision 0.8B", "Intern-Decision 2B",
                 "Intern-Decision 4B"])


if __name__ == "__main__":
    unittest.main()
