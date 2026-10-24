"""Offline regression tests for engines/k2type_client.py and
demos/k2type_demo.py.

No model, no downloads: the modules must import without jev/torch (heavy
imports are lazy, and to_record() only touches the snapshot when an
instructions/criteria payload carries a dict that needs jev.encode.render),
the pure request/response mapping functions are exercised, and the K2TYPE
registries of both benchmark drivers are checked against the client's
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

import demos.k2type_demo as k2type_demo
from engines import k2type_client


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.k2type_demo")
        added = set(sys.modules) - before
        self.assertNotIn("jev", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)

    def test_load_refuses_to_run_without_the_runtime(self):
        # the heavy import sits inside load(), not at module level
        with patch.dict(sys.modules, {"jev": None}):
            with self.assertRaises(ImportError):
                k2type_demo.load()


class ModelHomeTests(unittest.TestCase):
    def test_default_is_the_c_drive_snapshot(self):
        # D: is full; the snapshot follows the Intern-Decision precedent
        # at C:\src (see the client docstring)
        self.assertTrue(k2type_client.model_home().endswith("K2-Type-0.9B"))
        self.assertIn("K2-Type-0.9B", k2type_client.model_home())

    def test_k2type_home_env_overrides(self):
        with patch.dict(os.environ, {"K2TYPE_HOME": "D:/models/K2-Type"}):
            self.assertEqual(k2type_client.model_home(), "D:/models/K2-Type")

    def test_registry_carries_the_hf_model_id(self):
        self.assertEqual(k2type_demo.MODEL_ID, k2type_client.MODEL_ID)
        self.assertEqual(k2type_client.MODEL_ID, "IFM/K2-Type-0.9B")


class ToRecordTests(unittest.TestCase):
    def test_choice_takes_string_instructions_and_criteria(self):
        rec = k2type_client.to_record({
            "state": "a ticket",
            "questions": {"q": {
                "type": "choice", "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": None}}}})
        self.assertEqual(rec["state"], "a ticket")
        self.assertEqual(rec["questions"]["q"], {
            "type": "choice", "instructions": "Which team?",
            "criteria": {"billing": "Payments", "technical": None},
            # label is a placeholder the forward never reads
            "label": "billing"})

    def test_score_and_noul_rows_map_with_placeholders(self):
        rec = k2type_client.to_record({
            "state": None,
            "questions": {
                "s": {"type": "score", "instructions": "How urgent?",
                      "criteria": ["low", "high"]},
                "n": {"type": "noul", "instructions": "Refund?"}}})
        self.assertEqual(rec["questions"]["s"]["criteria"], ["low", "high"])
        self.assertEqual(rec["questions"]["s"]["label"], 0)
        self.assertFalse(rec["questions"]["n"]["label"])
        self.assertNotIn("criteria", rec["questions"]["n"])

    def test_option_counts_are_validated_like_the_server(self):
        with self.assertRaises(ValueError) as ctx:
            k2type_client.to_record({
                "state": "s",
                "questions": {"q": {"type": "choice",
                                    "instructions": "?",
                                    "criteria": {}}}})
        self.assertIn("1..255", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            k2type_client.to_record({
                "state": "s",
                "questions": {"q": {"type": "score",
                                    "instructions": "?",
                                    "criteria": ["only"]}}})
        self.assertIn("2..255", str(ctx.exception))

    def test_unknown_type_names_the_question(self):
        with self.assertRaises(ValueError) as ctx:
            k2type_client.to_record({
                "state": "s",
                "questions": {"weird": {"type": "bool",
                                        "instructions": "?"}}})
        self.assertIn("weird", str(ctx.exception))


class AnswerTests(unittest.TestCase):
    def test_choice_picks_argmax_with_the_margin_confidence(self):
        row = k2type_client.answer(
            {"type": "choice",
             "criteria": {"a": None, "b": None, "c": None, "d": None}},
            [0.1, 0.7, 0.1, 0.1])
        self.assertEqual(row["choice"], "b")
        # (pmax - 1/K) / (1 - 1/K) with K=4
        self.assertAlmostEqual(row["confidence"], (0.7 - 0.25) / 0.75, 4)
        self.assertEqual(row["probabilities"]["b"], 0.7)

    def test_single_option_choice_is_fully_confident(self):
        row = k2type_client.answer({"type": "choice", "criteria": {"x": 1}},
                                   [1.0])
        self.assertEqual(row["confidence"], 1.0)

    def test_noul_reports_p_true(self):
        row = k2type_client.answer({"type": "noul", "label": False},
                                   [0.25, 0.75])
        self.assertEqual(row["noul"], 0.75)

    def test_score_returns_the_expected_level_and_legend(self):
        row = k2type_client.answer(
            {"type": "score", "criteria": ["low", "mid", "high"],
             "label": 0},
            [0.2, 0.5, 0.3])
        self.assertAlmostEqual(row["score"], 1.1, 4)
        self.assertEqual(row["legend"], {"0": "low", "1": "mid",
                                         "2": "high"})
        self.assertEqual(row["probabilities"], {"0": 0.2, "1": 0.5,
                                                "2": 0.3})


class DemoMappingTests(unittest.TestCase):
    def test_choice_question_carries_type_instructions_criteria(self):
        q = k2type_demo.choice_question("Which team?",
                                        {"billing": "Payments"})
        self.assertEqual(q, {"type": "choice",
                             "instructions": "Which team?",
                             "criteria": {"billing": "Payments"}})

    def test_describe_labels_uses_the_bench_descriptions(self):
        self.assertEqual(
            k2type_demo.describe_labels(["positive", "negative"],
                                        "sentiment"),
            {"positive": "Text expresses a clearly positive attitude",
             "negative": "Text expresses a clearly negative attitude"})

    def test_describe_labels_falls_back_to_a_task_phrase(self):
        self.assertEqual(k2type_demo.describe_labels(["urgent"], "mood"),
                         {"urgent": "Text whose mood is urgent"})

    def test_question_for_restates_the_text(self):
        self.assertEqual(
            k2type_demo.question_for("sentiment", "good day"),
            'What is the overall sentiment of this text: "good day"')
        self.assertEqual(
            k2type_demo.question_for("urgency", "now"),
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
        row = k2type_demo.decide_one(
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
        # confidence is the model's own margin, not pmax
        self.assertEqual(row, {"choice": "positive",
                               "probabilities": {"positive": 0.9,
                                                 "negative": 0.1},
                               "confidence": 0.8})

    def test_missing_answer_fields_map_to_none(self):
        engine = _StubEngine()
        engine.answer = {"type": "choice"}
        row = k2type_demo.decide_one(engine, "x", "sentiment",
                                     {"positive": "p", "negative": "n"})
        self.assertIsNone(row["choice"])
        self.assertIsNone(row["probabilities"])
        self.assertIsNone(row["confidence"])


class ServePayloadTests(unittest.TestCase):
    def test_payload_parses_to_texts_task_labels(self):
        parsed = k2type_demo.parse_serve_payload(
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
                    k2type_demo.parse_serve_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeTests(unittest.TestCase):
    def run_serve(self, payload: dict):
        stub = _StubEngine()
        stdout = io.StringIO()
        with patch.object(k2type_demo, "load", return_value=stub), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            k2type_demo.serve()
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
            k2type_demo.serve()
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["error"].startswith("JSONDecodeError:"),
                        payload["error"])


class RegistryTests(unittest.TestCase):
    def test_k2type_is_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        self.assertIn("K2-Type 0.9B (local)", bench_spectrum.ALL_SYSTEMS)
        self.assertIn("K2-Type 0.9B (local)",
                      bench_multilingual.ALL_SYSTEMS)
        self.assertIn("K2-Type 0.9B (local)",
                      bench_multilingual.K2TYPE)

    def test_registry_entries_share_the_model_id(self):
        import bench_multilingual
        import bench_spectrum

        # both drivers point at the same local snapshot
        self.assertEqual(bench_spectrum.K2TYPE["K2-Type 0.9B (local)"],
                         k2type_client.MODEL_ID)
        self.assertEqual(bench_multilingual.K2TYPE["K2-Type 0.9B (local)"],
                         k2type_client.MODEL_ID)


if __name__ == "__main__":
    unittest.main()
