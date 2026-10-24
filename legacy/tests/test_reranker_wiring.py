"""Offline regression tests for the local cross-encoder reranker wiring
(mxbai-rerank-large-v2 and nvidia's nemotron-rerank-1b-v2, joining the
incumbent trio) in bench_spectrum.py and bench_multilingual.py.

No network, no model loads: the registry pins the card repo ids (a drift
would silently bench a different checkpoint), and the nemotron pair
template is pinned verbatim against the card's trained format (the
remote-code head scores "question:... \\n \\n passage:..." strings; any
whitespace drift would silently change every score).
"""

import unittest

import bench_multilingual
import bench_spectrum
from bench_spectrum import NEMOTRON_RERANKERS, RERANKERS, nemotron_prompt


class RegistryTests(unittest.TestCase):
    def test_registry_pins_the_card_repos(self):
        # the five HF repo ids, checked 2026-10-05
        self.assertEqual(RERANKERS, {
            "mxbai-rerank-base-v2": "mixedbread-ai/mxbai-rerank-base-v2",
            "mxbai-rerank-large-v2": "mixedbread-ai/mxbai-rerank-large-v2",
            "bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
            "GTE-rerank-ModernBERT-base":
                "Alibaba-NLP/gte-reranker-modernbert-base",
        })
        self.assertEqual(NEMOTRON_RERANKERS, {
            "nemotron-rerank-1b-v2": "nvidia/llama-nemotron-rerank-1b-v2",
        })

    def test_no_local_reranker_carries_a_hosted_suffix(self):
        # house naming for the local rerankers is the bare HF name - the
        # "(OpenRouter)" suffix is reserved for the hosted rows
        for name in list(RERANKERS) + list(NEMOTRON_RERANKERS):
            self.assertNotIn("(", name)

    def test_every_local_reranker_is_registered_in_both_benchmarks(self):
        for name in list(RERANKERS) + list(NEMOTRON_RERANKERS):
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)


class NemotronPromptTests(unittest.TestCase):
    def test_prompt_is_the_card_template_verbatim(self):
        # llama-nemotron-rerank-1b-v2 usage example, checked 2026-10-05:
        # f"question:{q} \n \n passage:{p}" (single line, spaces around
        # the blank line)
        self.assertEqual(nemotron_prompt("how much protein?",
                                         "46 grams per day"),
                         "question:how much protein? \n \n "
                         "passage:46 grams per day")

    def test_both_benchmarks_share_the_template(self):
        # the benches duplicate their shape helpers (like the reranker
        # instruction strings); a one-file edit must not split the shape
        self.assertEqual(bench_spectrum.nemotron_prompt("q", "p"),
                         bench_multilingual.nemotron_prompt("q", "p"))

    def test_nemotron_prompts_embed_the_house_label_descriptions(self):
        # the measured shape (bench_spectrum.NEMOTRON_RERANKERS notes):
        # raw text as query, the house label description as passage -
        # bare labels collapse the model (9/24 sentiment on the mixed
        # pool vs 19/24), the same call as the Ollaya NLI criteria
        descriptions = {**bench_spectrum.SENTIMENT_LABELS,
                        **bench_spectrum.TOPIC_LABELS}
        for label in list(bench_spectrum.SENTIMENT_LABELS) + \
                list(bench_spectrum.TOPIC_LABELS):
            self.assertIn(descriptions[label],
                          nemotron_prompt("some text", descriptions[label]))


if __name__ == "__main__":
    unittest.main()
