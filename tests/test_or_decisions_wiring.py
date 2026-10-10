"""Offline registry tests for the OpenRouter Decisions family.

The Decisions arrivals (Perplexity Decider V1 / V1.1, Liquid d1,
Together Tev1 4B, Inception Mercury Decide; extended 2026-10-09 by
Microsoft Decision-1 and Nace.AI's Drex v1.5) must be registered under
the same names and model ids in both benchmark drivers and the demo app,
and must sit in the choice-question subset - a name drift would silently
drop a system from one benchmark, and a Span-style scorer left in the
choice set would get questions it cannot answer.
"""

import unittest

import app
import bench_multilingual
import bench_spectrum

NEW_DECISIONS = {
    "Decider V1 27B (OpenRouter)": "perplexity/pplx-decider-v1-27b",
    "Decider V1.1 27B (OpenRouter)": "perplexity/pplx-decider-v1.1-27b",
    "D1 (OpenRouter)": "liquid/d1",
    "Tev1 4B (OpenRouter)": "togethercomputer/tev1-4b-experimental",
    "Mercury Decide (OpenRouter)": "inception/mercury-decide:free",
    "Microsoft Decision-1 (OpenRouter)": "microsoft/microsoft-decision-1",
    "Drex v1.5 (OpenRouter)": "nace-ai/drex-v1.5",
}


class DecisionsRegistryTests(unittest.TestCase):
    def test_new_decisions_registered_identically_everywhere(self):
        for registry in (bench_spectrum.OPENROUTER_SYSTEMONE,
                         bench_multilingual.OPENROUTER_SYSTEMONE,
                         app.OPENROUTER_SYSTEMONE):
            for name, model_id in NEW_DECISIONS.items():
                self.assertEqual(registry.get(name), model_id,
                                 f"{name} missing or wrong id in "
                                 f"{registry.__class__.__name__}")

    def test_choice_subset_holds_decisions_but_never_the_spans(self):
        for module in (bench_spectrum, bench_multilingual, app):
            subset = module.OR_SYSTEMONE_CHOICE
            self.assertTrue(set(NEW_DECISIONS) <= subset)
            self.assertNotIn("Span-01", subset)
            self.assertNotIn("Span-01 Lite", subset)
            # every choice-set name must have a model id to route to
            for name in subset:
                self.assertIn(name, module.OPENROUTER_SYSTEMONE)

    def test_new_decisions_in_both_cli_choices(self):
        for names in (bench_spectrum.ALL_SYSTEMS,
                      bench_multilingual.ALL_SYSTEMS):
            self.assertTrue(set(NEW_DECISIONS) <= set(names))

    def test_recorded_runs_carry_measured_usage(self):
        # the Decisions endpoints report cost in their usage block, so
        # both results files must carry measured (not derived) usage
        # rows for the new family
        import json

        with open("results/bench_spectrum_results.json",
                  encoding="utf-8") as fh:
            spectrum = json.load(fh)
        with open("results/bench_multilingual_results.json",
                  encoding="utf-8") as fh:
            multilingual = json.load(fh)
        for name in NEW_DECISIONS:
            entry = spectrum["systems"][name]
            self.assertIn("cls_accuracy", entry)
            row = entry["usage"]
            self.assertTrue(row["cost_reported"])
            self.assertGreaterEqual(row["paid_requests"], 1)

            self.assertIn(name, multilingual["by_language"])
            row = multilingual["usage"][name]
            self.assertTrue(row["cost_reported"])
            self.assertGreaterEqual(row["paid_requests"], 1)


if __name__ == "__main__":
    unittest.main()
