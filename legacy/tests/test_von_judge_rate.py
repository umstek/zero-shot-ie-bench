"""Offline regression tests for the von judge/rate wrappers.

No model, no downloads: engines/von_client.py must import without torch,
demos/von_demo.py without torch/transformers, and the wrapper/payload/
row logic is exercised against stubs (the real SDK only exists in
.venv-von, so von.types is stubbed here like tests/test_von_loading.py).
"""

import contextlib
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demos.von_demo as von_demo
from engines import von_client

VON_TYPES_STUB = types.SimpleNamespace(
    Choice=lambda **kwargs: kwargs,
    Noul=lambda **kwargs: kwargs,
    Score=lambda **kwargs: kwargs,
)


class ImportTests(unittest.TestCase):
    def test_demo_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.von_demo")
        added = set(sys.modules) - before
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)
        self.assertNotIn("von", added)

    def test_client_module_never_imports_the_sdk_at_module_level(self):
        # the client imports pathlib only; the stub pass below would be
        # unnecessary otherwise
        self.assertNotIn("von", sys.modules)


class JudgeWrapperTests(unittest.TestCase):
    def test_judge_calls_evaluate_noul_with_the_backend_temperature(self):
        backend = Mock()
        backend._default_temp = 0.7
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}):
            judge = von_client.load_von_judge(backend)
            answer = judge(state="ticket text", instructions="Is it a "
                                                          "refund?")
        backend.evaluate_noul.assert_called_once_with(
            "judgment", "ticket text", {"instructions": "Is it a refund?"},
            temperature=0.7)
        self.assertIs(answer, backend.evaluate_noul.return_value)

    def test_judge_without_a_backend_loads_one(self):
        sentinel = Mock()
        sentinel._default_temp = 1.0
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}), \
                patch.object(von_client, "load_von_backend",
                             return_value=sentinel) as load:
            judge = von_client.load_von_judge()
            judge(state="s", instructions="i")
        load.assert_called_once_with()
        sentinel.evaluate_noul.assert_called_once()


class RateWrapperTests(unittest.TestCase):
    def test_rate_calls_evaluate_score_with_the_ordered_rubric(self):
        backend = Mock()
        backend._default_temp = 1.0
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}):
            rate = von_client.load_von_rate(backend)
            answer = rate(state="ticket text", rubric=["low", "high"],
                          instructions="How urgent?")
        backend.evaluate_score.assert_called_once_with(
            "rating", "ticket text", {"instructions": "How urgent?",
                                      "criteria": ["low", "high"]},
            temperature=1.0)
        self.assertIs(answer, backend.evaluate_score.return_value)

    def test_wrappers_share_one_preloaded_backend(self):
        backend = Mock()
        backend._default_temp = 1.0
        with patch.dict(sys.modules, {"von.types": VON_TYPES_STUB}), \
                patch.object(von_client, "load_von_backend") as load:
            von_client.load_von_judge(backend)
            von_client.load_von_rate(backend)
            von_client.load_von_decider(backend)
        load.assert_not_called()


class RowTests(unittest.TestCase):
    def test_judgment_row_maps_noul_and_derives_the_verdict(self):
        self.assertEqual(
            von_demo.judgment_row(types.SimpleNamespace(noul=0.87)),
            {"noul": 0.87, "verdict": "yes"})
        self.assertEqual(
            von_demo.judgment_row(types.SimpleNamespace(noul=0.2)),
            {"noul": 0.2, "verdict": "no"})
        # the 0.5 threshold itself counts as yes
        self.assertEqual(
            von_demo.judgment_row(types.SimpleNamespace(noul=0.5))["verdict"],
            "yes")

    def test_rating_row_maps_score_confidence_legend_probabilities(self):
        answer = types.SimpleNamespace(
            score=2.4, confidence=0.4, legend={"0": "low", "1": "high"},
            probabilities={"0": 0.1, "1": 0.9})
        self.assertEqual(von_demo.rating_row(answer),
                         {"score": 2.4, "confidence": 0.4,
                          "legend": {"0": "low", "1": "high"},
                          "probabilities": {"0": 0.1, "1": 0.9}})


class JudgeRatePayloadTests(unittest.TestCase):
    def test_complete_payload_parses(self):
        parsed = von_demo.parse_judge_rate_payload({
            "text": "  ticket  ", "judge_instructions": "refund?",
            "rate_instructions": "urgency?",
            "rubric": ["a", " b ", ""]})
        self.assertEqual(parsed, ("ticket", "refund?", "urgency?",
                                  ["a", "b"]))

    def test_missing_or_thin_fields_raise_with_the_field_named(self):
        for payload, field in (
                ({}, "text"),
                ({"text": "  "}, "text"),
                ({"text": "t"}, "judge_instructions"),
                ({"text": "t", "judge_instructions": "j"},
                 "rate_instructions"),
                ({"text": "t", "judge_instructions": "j",
                  "rate_instructions": "r"}, "rubric"),
                ({"text": "t", "judge_instructions": "j",
                  "rate_instructions": "r", "rubric": ["only one"]},
                 "rubric")):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError) as ctx:
                    von_demo.parse_judge_rate_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ChoicePayloadTests(unittest.TestCase):
    def test_complete_payload_parses(self):
        parsed = von_demo.parse_choice_payload(
            {"texts": ["a", ""], "instructions": "sentiment",
             "choices": {"positive": None}})
        self.assertEqual(parsed, (["a"], "sentiment", {"positive": None}))

    def test_missing_fields_raise_with_the_field_named(self):
        for payload, field in (({}, "texts"), ({"texts": ["a"]}, "instructions"),
                               ({"texts": ["a"], "instructions": "i"},
                                "choices"),
                               ({"texts": ["a"], "instructions": "i",
                                 "choices": ["not", "a", "dict"]},
                                "choices")):
            with self.subTest(missing=field):
                with self.assertRaises(ValueError) as ctx:
                    von_demo.parse_choice_payload(payload)
                self.assertIn(field, str(ctx.exception))


class ServeJudgeRateTests(unittest.TestCase):
    def run_serve(self, payload: dict):
        stdout = io.StringIO()
        with patch.object(von_demo, "load", return_value=Mock()), \
                patch("engines.von_client.load_von_judge",
                      return_value=lambda **kw: types.SimpleNamespace(
                          noul=0.87)), \
                patch("engines.von_client.load_von_rate",
                      return_value=lambda **kw: types.SimpleNamespace(
                          score=2.4, confidence=0.4,
                          legend={"0": "wait", "1": "soon", "2": "now"},
                          probabilities={"0": 0.1, "1": 0.2, "2": 0.7})), \
                patch.object(sys, "stdin",
                             io.StringIO(json.dumps(payload))), \
                contextlib.redirect_stdout(stdout):
            von_demo.serve()
        return json.loads(stdout.getvalue())

    def test_judge_rate_mode_answers_both_types_on_the_shared_sample(self):
        payload = self.run_serve(
            {"mode": "judge_rate", "text": "ticket",
             "judge_instructions": "refund?", "rate_instructions": "urgency?",
             "rubric": ["wait", "soon", "now"]})
        self.assertNotIn("error", payload)
        self.assertEqual(payload["judgment"], {"noul": 0.87, "verdict": "yes"})
        self.assertEqual(payload["rating"]["score"], 2.4)
        self.assertEqual(payload["rating"]["probabilities"]["2"], 0.7)

    def test_judge_rate_mode_maps_errors_to_the_error_json(self):
        payload = self.run_serve({"mode": "judge_rate", "text": ""})
        self.assertIn("error", payload)
        self.assertTrue(payload["error"].startswith("ValueError:"),
                        payload["error"])


if __name__ == "__main__":
    unittest.main()
