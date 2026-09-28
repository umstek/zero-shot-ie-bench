"""Offline regression tests for the GLiNER2.5-Decide score rubric demo.

No model, no downloads: demos/demo.py must import without torch (heavy
imports are lazy), the rubric helper must produce the ordered string
labels classify_text expects, and the Decide registry ids must line up
(a name drift would silently drop the sibling from the tour).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import demos.demo as demo


class ImportTests(unittest.TestCase):
    def test_import_pulls_no_heavy_dependencies(self):
        # the delta matters: another test (or venv) may already hold torch
        before = set(sys.modules)
        import importlib

        module = importlib.import_module("demos.demo")
        added = set(sys.modules) - before
        self.assertNotIn("gliner2", added)
        self.assertNotIn("torch", added)
        self.assertNotIn("transformers", added)


class RubricLevelsTests(unittest.TestCase):
    def test_default_is_the_card_example_0_to_10_scale(self):
        levels = demo.rubric_levels()
        self.assertEqual(levels, [str(i) for i in range(0, 11)])
        self.assertEqual(levels[0], "0")
        self.assertEqual(levels[-1], "10")

    def test_custom_ranges_stay_ordered_string_labels(self):
        self.assertEqual(demo.rubric_levels(1, 5), ["1", "2", "3", "4", "5"])

    def test_refuses_a_flat_or_inverted_scale(self):
        with self.assertRaisesRegex(ValueError, "greater than low"):
            demo.rubric_levels(5, 1)
        with self.assertRaisesRegex(ValueError, "greater than low"):
            demo.rubric_levels(3, 3)


class DecideRegistryTests(unittest.TestCase):
    def test_decide_sibling_is_a_tour_section_on_the_pinned_checkpoint(self):
        self.assertIn(demo.decide_sibling, demo.SECTIONS)
        self.assertEqual(demo.DECIDE_MODEL_ID, demo.MODELS["decide"])
        self.assertEqual(demo.DECIDE_MODEL_ID, "fastino/GLiNER2.5-Decide")

    def test_decide_sibling_skips_itself_when_the_tour_already_runs_it(self):
        # section 7 must not reload Decide when --model decide chose it
        import contextlib
        import io

        saved = demo.MODEL_ID
        demo.MODEL_ID = demo.DECIDE_MODEL_ID
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertIsNone(demo.decide_sibling(model=None))
        finally:
            demo.MODEL_ID = saved


if __name__ == "__main__":
    unittest.main()
