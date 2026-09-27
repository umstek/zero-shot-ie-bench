"""Offline regression tests for demos/lumma_demo.py.

No model, no downloads: the module must import without lumma_fev/torch
(heavy imports are lazy, the module loads in the main venv too), the pure
request/response mapping functions are exercised against stubs, and the
LUMMA registries of both benchmark drivers are checked against the demo's
checkpoint map (a name drift would silently drop or misroute a system).
"""

import contextlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demos.lumma_demo as lumma_demo


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.lumma_demo")
        added = set(sys.modules) - before
        self.assertNotIn("lumma_fev", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)

    def test_load_refuses_to_run_without_lumma_fev(self):
        # the heavy import sits inside load(), not at module level
        with patch.dict(sys.modules, {"lumma_fev": None}):
            with self.assertRaises(ImportError):
                lumma_demo.load("0.15b")


class ModelIdTests(unittest.TestCase):
    def test_none_and_empty_default_to_0_15b(self):
        self.assertEqual(lumma_demo.resolve_model_id(None),
                         "FrontiersMind/Lumma-fev-0.1b")
        self.assertEqual(lumma_demo.resolve_model_id(""),
                         "FrontiersMind/Lumma-fev-0.1b")

    def test_registry_keys_and_case_insensitive_keys_resolve(self):
        self.assertEqual(lumma_demo.resolve_model_id("0.6b"),
                         "FrontiersMind/Lumma-fev-0.6b")
        self.assertEqual(lumma_demo.resolve_model_id("0.15B"),
                         "FrontiersMind/Lumma-fev-0.1b")
        self.assertEqual(lumma_demo.resolve_model_id("4B"),
                         "FrontiersMind/Lumma-fev-4b")

    def test_full_hf_id_passes_through(self):
        self.assertEqual(lumma_demo.resolve_model_id("FrontiersMind/Lumma-fev-4b"),
                         "FrontiersMind/Lumma-fev-4b")

    def test_unknown_key_raises_with_the_known_ones(self):
        with self.assertRaises(ValueError) as ctx:
            lumma_demo.resolve_model_id("9b")
        self.assertIn("0.15b", str(ctx.exception))


class QuestionTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = lumma_demo.choice_question("Which team?", {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice", "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_question_for_restates_the_text(self):
        # the shape the 0.1b probe scored best (see the bench branch
        # comment): text in the state AND in the instructions
        self.assertEqual(
            lumma_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            lumma_demo.question_for("topic", "cpu benchmarks"),
            'Which topic category does this text belong to: '
            '"cpu benchmarks"')

    def test_question_for_falls_back_to_the_task_word(self):
        self.assertEqual(
            lumma_demo.question_for("urgency", "now"),
            'Which urgency category does this text belong to: "now"')


class _StubModel:
    """Records decide() requests, answers every question alike."""

    def __init__(self):
        self.answer = {"type": "choice", "choice": "positive",
                       "confidence": 0.9,
                       "probabilities": {"positive": 0.9, "negative": 0.1}}
        self.calls = []

    def decide(self, state, questions):
        self.calls.append((state, questions))
        return {name: dict(self.answer) for name in questions}


class DecideOneTests(unittest.TestCase):
    def test_text_is_the_state_and_the_row_maps_choice_probabilities(self):
        model = _StubModel()
        row = lumma_demo.decide_one(
            model, "The food was cold.", "sentiment",
            {"positive": None, "negative": None})
        state, questions = model.calls[0]
        self.assertEqual(state, "The food was cold.")
        question = questions["q"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(
            question["instructions"],
            'What is the overall sentiment of this text: '
            '"The food was cold."')
        self.assertEqual(question["criteria"],
                         {"positive": None, "negative": None})
        self.assertEqual(row, {"choice": "positive",
                               "probabilities": {"positive": 0.9,
                                                 "negative": 0.1},
                               "confidence": 0.9})

    def test_missing_answer_fields_map_to_none(self):
        model = _StubModel()
        model.answer = {"type": "choice"}
        row = lumma_demo.decide_one(model, "x", "sentiment",
                                    {"positive": None, "negative": None})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels_model(self):
        parsed = lumma_demo.parse_serve_payload(
            {"texts": ["a", "", "b"], "task": "sentiment",
             "labels": ["positive", "negative"]})
        self.assertEqual(parsed, (["a", "b"], "sentiment",
                                  ["positive", "negative"],
                                  "FrontiersMind/Lumma-fev-0.1b"))

    def test_optional_model_key_resolves(self):
        parsed = lumma_demo.parse_serve_payload(
            {"texts": ["a"], "task": "topic",
             "labels": ["technology", "sports"], "model": "4b"})
        self.assertEqual(parsed[3], "FrontiersMind/Lumma-fev-4b")

    def test_empty_or_missing_fields_raise_with_the_field_named(self):
        for payload, field in (({}, "texts"),
                               ({"texts": ["", "  "]}, "texts"),
                               ({"texts": ["a"]}, "task"),
                               ({"texts": ["a"], "task": "sentiment"},
                                "labels")):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError) as ctx:
                    lumma_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict):
        stub = _StubModel()
        stdout = io.StringIO()
        with patch.object(lumma_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            lumma_demo.serve()
        return json.loads(stdout.getvalue()), stub

    def test_serve_answers_every_text_in_the_wire_contract(self):
        payload, stub = self.run_serve(
            {"texts": ["a", "b"], "task": "sentiment",
             "labels": ["positive", "negative"]})
        self.assertNotIn("error", payload)
        self.assertEqual([row["choice"] for row in payload["results"]],
                         ["positive", "positive"])
        # one model load per click, one decide() per text
        self.assertEqual(len(stub.calls), 2)

    def test_serve_maps_errors_to_the_error_json(self):
        payload, _ = self.run_serve({"texts": [], "task": "sentiment",
                                     "labels": ["positive"]})
        self.assertIn("error", payload)
        self.assertTrue(payload["error"].startswith("ValueError:"),
                        payload["error"])

    def test_serve_rejects_unknown_checkpoints_by_name(self):
        payload, _ = self.run_serve({"texts": ["a"], "task": "sentiment",
                                     "labels": ["positive", "negative"],
                                     "model": "9b"})
        self.assertIn("unknown Lumma checkpoint '9b'", payload["error"])


class RegistryTests(unittest.TestCase):
    def test_every_lumma_system_is_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for name in bench_spectrum.LUMMA:
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.LUMMA)

    def test_demo_checkpoint_map_covers_the_bench_registry(self):
        import bench_spectrum

        # keys differ by design (shorthand vs display names); the
        # checkpoint ids must line up
        self.assertEqual(sorted(lumma_demo.MODEL_IDS.values()),
                         sorted(bench_spectrum.LUMMA.values()))


if __name__ == "__main__":
    unittest.main()
