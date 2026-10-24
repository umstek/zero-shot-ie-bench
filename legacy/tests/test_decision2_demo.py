"""Offline regression tests for engines/decision2_client.py and
demos/decision2_demo.py.

No model, no downloads: the modules must import without decision2/torch
(heavy imports are lazy), the pure request/response mapping functions
are exercised, and the DECISION2 registries of both benchmark drivers
are checked against the client's size keys (a name drift would
silently drop or misroute a system).
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

import demos.decision2_demo as decision2_demo
from engines import decision2_client


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.decision2_demo")
        added = set(sys.modules) - before
        self.assertNotIn("decision2", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)

    def test_load_refuses_to_run_without_the_snapshot(self):
        # the heavy import sits inside load(); a missing snapshot fails
        # fast with the pull command, before torch is ever touched (the
        # sys.modules None-patch trick would not work here: load_engine
        # pops stale decision2 modules so a re-load picks the right
        # snapshot, which also pops the patch)
        with patch.dict(os.environ,
                        {"DECISION2_HOME": "Z:/no/such/prefix"}):
            with self.assertRaises(FileNotFoundError) as ctx:
                decision2_demo.load()
        self.assertIn("hf download", str(ctx.exception))


class SizeTests(unittest.TestCase):
    def test_sizes_cover_the_three_benched_checkpoints(self):
        self.assertEqual(decision2_client.SIZES, {
            "kai-0.6b": "Decision-2.0-Kai-0.6B",
            "eos-0.8b": "Decision-2.0-Eos-0.8B",
            "sol-2b": "Decision-2.0-Sol-2B"})
        self.assertEqual(decision2_client.DEFAULT_SIZE, "kai-0.6b")

    def test_normalize_takes_shorthand_full_repo_or_none(self):
        self.assertEqual(decision2_client.normalize_size(None), "kai-0.6b")
        self.assertEqual(decision2_client.normalize_size("Kai-0.6B"),
                         "kai-0.6b")
        self.assertEqual(
            decision2_client.normalize_size("vllm-sr/Decision-2.0-Sol-2B"),
            "sol-2b")
        with self.assertRaises(ValueError) as ctx:
            decision2_client.normalize_size("nox-4b")
        self.assertIn("nox-4b", str(ctx.exception))


class ModelDirTests(unittest.TestCase):
    def test_default_homes_sit_under_the_c_drive_base(self):
        # D: is full; the snapshots follow the K2-Type / Intern-Decision
        # precedent at C:\src (see the client docstring)
        self.assertEqual(
            decision2_client.model_dir("kai-0.6b"),
            os.path.join(r"C:\src", "Decision-2.0-Kai-0.6B"))
        self.assertTrue(
            decision2_client.model_dir("sol-2b").endswith(
                "Decision-2.0-Sol-2B"))

    def test_decision2_home_env_overrides_the_prefix(self):
        with patch.dict(os.environ, {"DECISION2_HOME": "D:/models"}):
            self.assertEqual(
                decision2_client.model_dir("eos-0.8b"),
                os.path.join("D:/models", "Decision-2.0-Eos-0.8B"))

    def test_missing_runtime_names_the_download_command(self):
        with patch.dict(os.environ,
                        {"DECISION2_HOME": os.path.dirname(__file__)}):
            with self.assertRaises(FileNotFoundError) as ctx:
                decision2_client._ensure_runtime_on_path("kai-0.6b")
            self.assertIn("hf download vllm-sr/Decision-2.0-Kai-0.6B",
                          str(ctx.exception))


class DemoMappingTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = decision2_demo.choice_question("Which team?",
                                           {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice",
                             "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_describe_labels_uses_the_bench_descriptions(self):
        self.assertEqual(
            decision2_demo.describe_labels(["positive", "negative"],
                                           "sentiment"),
            {"positive": "Text expresses a clearly positive attitude",
             "negative": "Text expresses a clearly negative attitude"})

    def test_describe_labels_falls_back_to_a_task_phrase(self):
        self.assertEqual(decision2_demo.describe_labels(["urgent"], "mood"),
                         {"urgent": "Text whose mood is urgent"})

    def test_question_for_restates_the_text(self):
        self.assertEqual(
            decision2_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            decision2_demo.question_for("urgency", "now"),
            'Which urgency category does this text belong to: "now"')


class _StubEngine:
    """Records predict() requests, answers every question alike."""

    def __init__(self):
        self.answer = {"type": "choice", "choice": "positive",
                       "confidence": 0.8,
                       "probabilities": {"positive": 0.9,
                                         "negative": 0.1}}
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, questions))
        return {"answers": {name: dict(self.answer)
                            for name in questions},
                "input_tokens": 42}


class DecideOneTests(unittest.TestCase):
    def test_text_is_the_state_and_the_row_maps_choice_probabilities(self):
        engine = _StubEngine()
        row = decision2_demo.decide_one(
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
        # confidence is the model's own entropy measure, not pmax
        self.assertEqual(row, {"choice": "positive",
                               "probabilities": {"positive": 0.9,
                                                 "negative": 0.1},
                               "confidence": 0.8})

    def test_missing_answer_fields_map_to_none(self):
        engine = _StubEngine()
        engine.answer = {"type": "choice", "error": "invalid_question"}
        row = decision2_demo.decide_one(engine, "x", "sentiment",
                                        {"positive": "p", "negative": "n"})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels(self):
        parsed = decision2_demo.parse_serve_payload(
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
                    decision2_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict, argv=None):
        stub = _StubEngine()
        stdout = io.StringIO()
        with patch.object(decision2_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            decision2_demo.serve()
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
        stdout = io.StringIO()
        with patch.object(sys, "stdin", io.StringIO("not json")), \
                contextlib.redirect_stdout(stdout):
            decision2_demo.serve()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["error"].startswith("JSONDecodeError:"),
                        payload["error"])

    def test_parse_size_flags_and_default(self):
        self.assertEqual(
            decision2_demo.parse_size(["prog", "--size", "sol-2b"]),
            "sol-2b")
        self.assertEqual(
            decision2_demo.parse_size(["prog", "-s", "Eos-0.8B"]),
            "eos-0.8b")
        self.assertEqual(decision2_demo.parse_size(["prog"]), "kai-0.6b")
        with self.assertRaises(ValueError):
            decision2_demo.parse_size(["prog", "--size"])


class RegistryTests(unittest.TestCase):
    def test_all_three_sizes_are_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for name in ("Decision 2.0 Kai 0.6B", "Decision 2.0 Eos 0.8B",
                     "Decision 2.0 Sol 2B"):
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)
            self.assertIn(name, bench_spectrum.DECISION2)
            self.assertIn(name, bench_multilingual.DECISION2)

    def test_registry_entries_share_the_client_size_keys(self):
        import bench_multilingual
        import bench_spectrum

        for registry in (bench_spectrum.DECISION2,
                         bench_multilingual.DECISION2):
            for name, size in registry.items():
                self.assertEqual(
                    size, decision2_client.normalize_size(size))
                self.assertIn(size, decision2_client.SIZES)


if __name__ == "__main__":
    unittest.main()
