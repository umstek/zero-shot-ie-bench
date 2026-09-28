"""Offline tests for the Intern-Decision IMAGE-input demo.

They mirror tests/test_intern_decision_demo.py: nothing here loads the
model (the snapshot's runtime, torch and transformers stay unimported);
only the payload parsing/validation and the deterministic Pillow sample
images are exercised. Run with the main venv:

    .venv/Scripts/python -m unittest discover -s tests -p "test_intern_decision_image_demo.py"
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demos import intern_decision_image_demo as demo


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        with patch.dict(sys.modules, {"torch": None, "transformers": None}):
            import importlib

            importlib.reload(demo)

    def test_module_documented_and_exports_serve_pieces(self):
        for name in ("serve", "parse_serve_payload", "build_question",
                     "build_samples", "sample_request"):
            self.assertTrue(callable(getattr(demo, name)), name)


class SampleImageTests(unittest.TestCase):
    def test_build_samples_draws_three_png_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = demo.build_samples(tmp)
            self.assertEqual(set(paths), {"ticket", "receipt", "chart"})
            for name, path in paths.items():
                self.assertTrue(Path(path).is_file(), name)
                self.assertGreater(Path(path).stat().st_size, 0)

    def test_build_samples_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, \
                tempfile.TemporaryDirectory() as second:
            one = demo.build_samples(first)
            two = demo.build_samples(second)
            for name in one:
                self.assertEqual(Path(one[name]).read_bytes(),
                                 Path(two[name]).read_bytes(), name)

    def test_sample_request_carries_image_state_and_three_questions(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(demo, "build_samples",
                              return_value=demo.build_samples(tmp)):
                request = demo.sample_request("ticket")
            self.assertTrue(Path(request["images"][0]).is_file())
            self.assertEqual(len(request["images"]), 1)
            self.assertTrue(request["state"])
            self.assertEqual(set(request["questions"]),
                             {"team", "refund", "urgency"})
            self.assertEqual(request["questions"]["team"]["type"], "choice")
            self.assertEqual(request["questions"]["refund"]["type"], "noul")
            self.assertEqual(request["questions"]["urgency"]["type"], "score")


class BuildQuestionTests(unittest.TestCase):
    def test_choice_requires_criteria_object_of_two_plus(self):
        question = demo.build_question(
            "choice", "Which team?", {"a": "first", "b": "second"})
        self.assertEqual(question["type"], "choice")
        self.assertNotIn("criteria_problem", question)
        with self.assertRaises(ValueError):
            demo.build_question("choice", "Which team?", {"a": "only"})
        with self.assertRaises(ValueError):
            demo.build_question("choice", "Which team?", ["a", "b"])

    def test_score_requires_rubric_list(self):
        question = demo.build_question(
            "score", "How urgent?", ["low", "high"])
        self.assertEqual(question["criteria"], ["low", "high"])
        with self.assertRaises(ValueError):
            demo.build_question("score", "How urgent?", {"low": 0})

    def test_noul_needs_no_criteria_and_type_is_normalized(self):
        question = demo.build_question(" NoUl ", "Is it?", None)
        self.assertEqual(question["type"], "noul")
        self.assertNotIn("criteria", question)

    def test_unknown_type_and_blank_instructions_are_rejected(self):
        with self.assertRaises(ValueError):
            demo.build_question("rank", "Rate it?", None)
        with self.assertRaises(ValueError):
            demo.build_question("choice", "   ", {"a": 1, "b": 2})


class ParseServePayloadTests(unittest.TestCase):
    def _png(self, tmp):
        return demo.build_samples(tmp)["ticket"]

    def test_valid_choice_payload_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = self._png(tmp)
            payload = {"image": image, "type": "choice",
                       "instructions": "Which team?",
                       "criteria": {"billing": "refunds",
                                    "technical": "bugs"},
                       "state": "a ticket", "model": "2b"}
            images, state, question = demo.parse_serve_payload(payload)
        self.assertEqual(images, [image])
        self.assertEqual(state, "a ticket")
        self.assertEqual(question["criteria"]["billing"], "refunds")

    def test_images_list_form_is_accepted_up_to_eight(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = self._png(tmp)
            images, _, _ = demo.parse_serve_payload(
                {"images": [image] * 8, "type": "noul",
                 "instructions": "Yes?"})
            self.assertEqual(len(images), 8)
            with self.assertRaises(ValueError):
                demo.parse_serve_payload(
                    {"images": [image] * 9, "type": "noul",
                     "instructions": "Yes?"})

    def test_missing_and_nonexistent_images_are_rejected(self):
        with self.assertRaises(ValueError):
            demo.parse_serve_payload({"type": "noul", "instructions": "Y?"})
        with self.assertRaises(ValueError):
            demo.parse_serve_payload({"image": "Z:/nope.png",
                                      "type": "noul", "instructions": "Y?"})

    def test_blank_state_gets_a_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = self._png(tmp)
            _, state, _ = demo.parse_serve_payload(
                {"image": image, "type": "noul", "instructions": "Y?"})
        self.assertEqual(state, "An uploaded image.")


if __name__ == "__main__":
    unittest.main()
