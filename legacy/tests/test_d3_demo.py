"""Offline regression tests for engines/d3_client.py and
demos/d3_demo.py.

No model, no downloads: the modules must import without
d3_runtime/torch (heavy imports are lazy), the pure request/response
mapping functions are exercised, and the D3 registries of both
benchmark drivers are checked against the client's size keys (a name
drift would silently drop or misroute a system).
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

import demos.d3_demo as d3_demo
from engines import d3_client


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.d3_demo")
        added = set(sys.modules) - before
        self.assertNotIn("d3_runtime", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)

    def test_load_refuses_to_run_without_the_snapshot(self):
        # the heavy import sits inside load(); a missing snapshot fails
        # fast with the pull command, before torch is ever touched
        with patch.dict(os.environ, {"D3_HOME": "Z:/no/such/prefix"}):
            with self.assertRaises(FileNotFoundError) as ctx:
                d3_demo.load()
        self.assertIn("hf download", str(ctx.exception))


class SizeTests(unittest.TestCase):
    def test_sizes_cover_the_three_benched_checkpoints(self):
        self.assertEqual(d3_client.SIZES,
                         {"edge": "d3-edge", "lite": "d3-lite",
                          "nano": "d3-nano"})
        self.assertEqual(d3_client.DEFAULT_SIZE, "edge")

    def test_normalize_takes_shorthand_full_repo_or_none(self):
        self.assertEqual(d3_client.normalize_size(None), "edge")
        self.assertEqual(d3_client.normalize_size("Edge"), "edge")
        self.assertEqual(d3_client.normalize_size("d3-nano"), "nano")
        self.assertEqual(d3_client.normalize_size("vllm-sr/d3-lite"),
                         "lite")
        with self.assertRaises(ValueError) as ctx:
            d3_client.normalize_size("mini")
        self.assertIn("mini", str(ctx.exception))


class ModelDirTests(unittest.TestCase):
    def test_default_homes_sit_under_the_c_drive_base(self):
        # D: is full; the snapshots follow the Decision 2.0 / K2-Type /
        # Intern-Decision precedent at C:\src (see the client docstring)
        self.assertEqual(d3_client.model_dir("edge"),
                         os.path.join(r"C:\src", "d3-edge"))
        self.assertTrue(d3_client.model_dir("nano").endswith("d3-nano"))

    def test_d3_home_env_overrides_the_prefix(self):
        with patch.dict(os.environ, {"D3_HOME": "D:/models"}):
            self.assertEqual(d3_client.model_dir("lite"),
                             os.path.join("D:/models", "d3-lite"))

    def test_missing_runtime_names_the_download_command(self):
        with patch.dict(os.environ,
                        {"D3_HOME": os.path.dirname(__file__)}):
            with self.assertRaises(FileNotFoundError) as ctx:
                d3_client._ensure_runtime_on_path("edge")
            self.assertIn("hf download vllm-sr/d3-edge",
                          str(ctx.exception))


class DemoMappingTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = d3_demo.choice_question("Which team?", {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice",
                             "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_describe_labels_uses_the_bench_descriptions(self):
        self.assertEqual(
            d3_demo.describe_labels(["positive", "negative"], "sentiment"),
            {"positive": "Text expresses a clearly positive attitude",
             "negative": "Text expresses a clearly negative attitude"})

    def test_describe_labels_falls_back_to_a_task_phrase(self):
        self.assertEqual(d3_demo.describe_labels(["urgent"], "mood"),
                         {"urgent": "Text whose mood is urgent"})

    def test_question_for_restates_the_text(self):
        self.assertEqual(
            d3_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            d3_demo.question_for("urgency", "now"),
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
        row = d3_demo.decide_one(
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
        row = d3_demo.decide_one(engine, "x", "sentiment",
                                 {"positive": "p", "negative": "n"})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels(self):
        parsed = d3_demo.parse_serve_payload(
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
                    d3_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict, argv=None):
        stub = _StubEngine()
        stdout = io.StringIO()
        with patch.object(d3_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            d3_demo.serve()
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
            d3_demo.serve()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["error"].startswith("JSONDecodeError:"),
                        payload["error"])

    def test_parse_size_flags_and_default(self):
        self.assertEqual(
            d3_demo.parse_size(["prog", "--size", "nano"]), "nano")
        self.assertEqual(d3_demo.parse_size(["prog", "-s", "lite"]), "lite")
        self.assertEqual(d3_demo.parse_size(["prog"]), "edge")
        with self.assertRaises(ValueError):
            d3_demo.parse_size(["prog", "--size"])


class RegistryTests(unittest.TestCase):
    def test_all_three_sizes_are_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for name in ("Decision 3.0 Edge 0.6B", "Decision 3.0 Lite 0.85B",
                     "Decision 3.0 Nano 2B"):
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)
            self.assertIn(name, bench_spectrum.D3)
            self.assertIn(name, bench_multilingual.D3)

    def test_registry_entries_share_the_client_size_keys(self):
        import bench_multilingual
        import bench_spectrum

        for registry in (bench_spectrum.D3, bench_multilingual.D3):
            for name, size in registry.items():
                self.assertEqual(size, d3_client.normalize_size(size))
                self.assertIn(size, d3_client.SIZES)


if __name__ == "__main__":
    unittest.main()
