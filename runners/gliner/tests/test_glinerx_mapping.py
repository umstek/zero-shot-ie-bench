"""Ported offline regression tests for the vendor multitask classification
mapping that lived in legacy/engines/glinerx_client.py.

The GLiNER-X checkpoints the mapping was measured on are culled and stay in
their own legacy venv (gliner[stanza] wants transformers 5.x), so the
registry/warm-up/load-guard tests of legacy/tests/test_glinerx_client.py
have no island counterpart — the mapping itself (prompt shape + single-label
reduction) now serves the relex checkpoint's census-row classification and
lives in gliner.models.gliner_relex.

No network, no model loads.
"""

from __future__ import annotations

from gliner_runner.models.gliner_relex import (CLASSIFICATION_PROMPT,
                                        classification_prompt,
                                        reduce_classification)


class TestMapping:
    def test_prompt_is_the_vendor_multitask_shape(self):
        assert (classification_prompt(["a", "b"], "some text")
                == CLASSIFICATION_PROMPT.format("a, b", "some text"))
        assert (classification_prompt(["a", "b"], "some text")
                == "Classify text into the following classes: a, b \n some text")

    def test_reduction_takes_the_top_scoring_entity_text(self):
        # gliner.multitask.GLiNERClassifier.process_predictions'
        # single-label branch, verbatim semantics
        entities = [{"text": "positive", "score": 0.6},
                    {"text": "negative", "score": 0.9}]
        assert reduce_classification(entities, ["a"]) == "negative"

    def test_reduction_strips_whitespace_but_keeps_any_text(self):
        # the vendor returns the entity TEXT as the label even when it is
        # not one of the requested classes — the bench scores it wrong,
        # it must not be silently remapped
        assert reduce_classification([{"text": "  Brazil ", "score": 0.95}],
                                     ["positive", "negative", "neutral"]) == "Brazil"

    def test_reduction_answers_the_vendors_other_when_nothing_survives(self):
        assert reduce_classification([], ["positive"]) == "other"
