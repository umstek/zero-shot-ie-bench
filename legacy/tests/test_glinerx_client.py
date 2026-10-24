"""Offline regression tests for engines/glinerx_client.py (Knowledgator's
GLiNER-X family) and its bench wiring.

No network, no model loads: the checkpoint registry, the vendor
multitask classification mapping (prompt shape + single-label reduction)
and the lazy stanza-splitter warm-up are tested on fakes, and the load
guard is exercised with the stanza import blocked (the main venv has no
stanza by design — GLiNER-X runs in .venv-glinerx).
"""

import sys
import unittest
from unittest.mock import patch

from engines import glinerx_client
from engines.glinerx_client import (CLASSIFICATION_PROMPT, MODELS,
                                    classification_prompt,
                                    reduce_classification, warm_splitter)


class RegistryTests(unittest.TestCase):
    def test_registry_pins_the_card_checkpoints(self):
        # the three HF repos' gliner_config.json files, checked 2026-10-05
        self.assertEqual(MODELS, {
            "GLiNER-X-small": "knowledgator/gliner-x-small",
            "GLiNER-X-base": "knowledgator/gliner-x-base",
            "GLiNER-X-large": "knowledgator/gliner-x-large",
        })

    def test_names_follow_the_gliner2_5_hyphen_convention(self):
        # the incumbent family is GLiNER2.5-small/base/multi; a drift
        # would split one family's rows across two naming styles
        for name in MODELS:
            self.assertTrue(name.startswith("GLiNER-X-"))

    def test_every_model_is_wired_but_culled_from_the_roster(self):
        import bench_multilingual
        import bench_spectrum

        for name in MODELS:
            self.assertIn(name, bench_spectrum.GLINER_X)
            self.assertIn(name, bench_multilingual.GLINER_X)
            self.assertNotIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertNotIn(name, bench_multilingual.ALL_SYSTEMS)


class MappingTests(unittest.TestCase):
    def test_prompt_is_the_vendor_multitask_shape(self):
        self.assertEqual(
            classification_prompt(["a", "b"], "some text"),
            CLASSIFICATION_PROMPT.format("a, b", "some text"))
        self.assertEqual(
            classification_prompt(["a", "b"], "some text"),
            "Classify text into the following classes: a, b \n some text")

    def test_reduction_takes_the_top_scoring_entity_text(self):
        # gliner.multitask.GLiNERClassifier.process_predictions'
        # single-label branch, verbatim semantics
        entities = [{"text": "positive", "score": 0.6},
                    {"text": "negative", "score": 0.9}]
        self.assertEqual(reduce_classification(entities, ["a"]), "negative")

    def test_reduction_strips_whitespace_but_keeps_any_text(self):
        # the vendor returns the entity TEXT as the label even when it is
        # not one of the requested classes — the bench scores it wrong,
        # it must not be silently remapped
        entities = [{"text": "  Brazil ", "score": 0.95}]
        self.assertEqual(reduce_classification(
            entities, ["positive", "negative", "neutral"]), "Brazil")

    def test_reduction_answers_the_vendors_other_when_nothing_survives(self):
        self.assertEqual(reduce_classification([], ["positive"]), "other")


class _FakeSplitter:
    def __init__(self):
        self.texts = []

    def __call__(self, text):
        self.texts.append(text)
        return iter([])


class _FakeModel:
    """Just the attribute path warm_splitter exercises."""

    class _Processor:
        pass

    def __init__(self):
        self.splitter = _FakeSplitter()
        self.data_processor = self._Processor()
        self.data_processor.words_splitter = self.splitter


class WarmupTests(unittest.TestCase):
    def test_warmup_walks_every_text_through_the_splitter(self):
        model = _FakeModel()
        warm_splitter(model, ["uno", "二", "three"])
        self.assertEqual(model.splitter.texts, ["uno", "二", "three"])


class LoadGuardTests(unittest.TestCase):
    def test_missing_stanza_names_the_venv(self):
        # the main venv deliberately has no stanza (GLiNER-X resolves
        # transformers 5.x, incompatible with its pinned 4.57.6): the
        # error must point at .venv-glinerx, not leak an ImportError
        with patch.dict(sys.modules, {"stanza": None}):
            with self.assertRaisesRegex(RuntimeError, "venv-glinerx"):
                glinerx_client.load_glinerx(MODELS["GLiNER-X-small"])


if __name__ == "__main__":
    unittest.main()
