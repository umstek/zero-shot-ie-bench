"""app.derived_cost: measured tokens × published list price for the
providers whose usage block reports tokens but no $ (the Clef family,
Jev, GLiDE, TypeLLM), and cost_summary_frame putting Jev on the cost
axis."""

import unittest

import app

# the recorded spectrum usage row (results/bench_spectrum_results.json)
JEV_USAGE = {"paid_requests": 2, "cost_usd": 0.0, "input_tokens": 6094}


class DerivedCostTest(unittest.TestCase):
    def test_jev_derived_from_measured_tokens(self):
        # 6,094 input tokens × $0.042/1M = $0.000255948 for the run
        self.assertAlmostEqual(
            app.derived_cost("Jev", dict(JEV_USAGE)),
            6094 * 0.042 / 1_000_000)

    def test_clef_pair_still_derived(self):
        usage = {"input_tokens": 4600}
        self.assertAlmostEqual(
            app.derived_cost("Clef (Workers AI)", usage),
            4600 * 0.24 / 1_000_000)
        self.assertAlmostEqual(
            app.derived_cost("Clef-flash (Workers AI)", usage),
            4600 * 0.09 / 1_000_000)

    def test_clef_omni_derived_at_its_own_price(self):
        # the MoE member bills $0.15/M input (not the pair's 0.24/0.09),
        # so a shared-price bug would surface here
        self.assertAlmostEqual(
            app.derived_cost("Clef-omni (Workers AI)",
                             {"input_tokens": 4600}),
            4600 * 0.15 / 1_000_000)

    def test_glide_derived_from_measured_tokens(self):
        # GLiDE's recorded spectrum row: 3,833 input tokens × $0.15/M
        # (thinking/output tokens are $0, so input is the whole cost)
        self.assertAlmostEqual(
            app.derived_cost("GLiDE (Fastino)", {"input_tokens": 3833}),
            3833 * 0.15 / 1_000_000)

    def test_typellm_derived_from_input_plus_thinking_tokens(self):
        # TypeLLM's recorded spectrum rows: input bills at $0.05/M and
        # per-question thinking tokens at $0.50/M (answers free) - the
        # plain run spent no thinking tokens, the thinking run both
        self.assertAlmostEqual(
            app.derived_cost("TypeLLM (hosted)",
                             {"input_tokens": 2507, "thinking_tokens": 0}),
            2507 * 0.05 / 1_000_000)
        self.assertAlmostEqual(
            app.derived_cost("TypeLLM thinking (hosted)",
                             {"input_tokens": 3226,
                              "thinking_tokens": 9376}),
            (3226 * 0.05 + 9376 * 0.50) / 1_000_000)

    def test_fastino_gliner_twins_derived_at_the_catalog_price(self):
        # the hosted GLiNER twins' recorded spectrum row: 864 input
        # tokens × $0.03/M each (GET /v1/base-models price; output
        # tokens are $0)
        for system in ("GLiNER2.5-base (Fastino)",
                       "GLiNER2.5-multi (Fastino)",
                       "GLiNER2.5-Decide (Fastino)"):
            self.assertAlmostEqual(
                app.derived_cost(system, {"input_tokens": 864}),
                864 * 0.03 / 1_000_000)

    def test_no_price_or_no_tokens_is_none(self):
        self.assertIsNone(app.derived_cost("Solar Decide (local)",
                                           {"input_tokens": 100}))
        self.assertIsNone(app.derived_cost("Jev", {"input_tokens": 0}))
        self.assertIsNone(app.derived_cost("Jev", {}))
        self.assertIsNone(app.derived_cost("TypeLLM (hosted)",
                                           {"thinking_tokens": 9376}))

    def test_cost_summary_frame_charts_jev_not_free_tiers(self):
        systems = {
            "Jev": {"cls_accuracy": 0.8, "usage": dict(JEV_USAGE)},
            "Span-01 Lite (OpenRouter)": {
                "cls_accuracy": 0.9,
                "usage": {"paid_requests": 1, "cost_usd": 0.0,
                          "input_tokens": 0}},
        }
        frame = app.cost_summary_frame(systems, n_q=48)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "System"], "Jev")
        self.assertAlmostEqual(frame.loc[0, "$ per question"],
                               6094 * 0.042 / 1_000_000 / 48)
        self.assertEqual(frame.loc[0, "Accuracy %"], 80.0)

    def test_cost_summary_frame_divides_by_the_questions_answered(self):
        # the hosted GLiNER twins' usage block bills 48 classification +
        # 18 NER questions (66 paid requests); GLiDE answers the 48 only.
        # 864 input tokens x $0.03/M over 66 questions = $3.9e-7/question
        systems = {
            "GLiNER2.5-base (Fastino)": {
                "cls_accuracy": 0.8333, "ner_exact": [True] * 18,
                "usage": {"paid_requests": 66, "cost_usd": 0.0,
                          "input_tokens": 864}},
            "GLiDE (Fastino)": {
                "cls_accuracy": 0.9792,
                "usage": {"paid_requests": 2, "cost_usd": 0.0,
                          "input_tokens": 3833}},
        }
        frame = app.cost_summary_frame(systems, n_q=48)
        by_name = frame.set_index("System")["$ per question"]
        self.assertAlmostEqual(by_name["GLiNER2.5-base (Fastino)"],
                               864 * 0.03 / 1_000_000 / 66)
        self.assertAlmostEqual(by_name["GLiDE (Fastino)"],
                               3833 * 0.15 / 1_000_000 / 48)


if __name__ == "__main__":
    unittest.main()
