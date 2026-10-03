"""app.derived_cost: measured tokens × published list price for the
providers whose usage block reports tokens but no $ (the Clef pair,
Jev, GLiDE), and cost_summary_frame putting Jev on the cost axis."""

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

    def test_glide_derived_from_measured_tokens(self):
        # GLiDE's recorded spectrum row: 3,833 input tokens × $0.15/M
        # (thinking/output tokens are $0, so input is the whole cost)
        self.assertAlmostEqual(
            app.derived_cost("GLiDE (Fastino)", {"input_tokens": 3833}),
            3833 * 0.15 / 1_000_000)

    def test_no_price_or_no_tokens_is_none(self):
        self.assertIsNone(app.derived_cost("Solar Decide (local)",
                                           {"input_tokens": 100}))
        self.assertIsNone(app.derived_cost("Jev", {"input_tokens": 0}))
        self.assertIsNone(app.derived_cost("Jev", {}))

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


if __name__ == "__main__":
    unittest.main()
