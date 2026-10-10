"""app.cost_speed_scatter: the isometric cost × latency × accuracy
projection for the metered hosted systems, and cost_summary_frame
carrying the latency column it feeds on."""

import json
import math
import unittest

import pandas as pd

import app

# three corners of the real spectrum data: Jev (cheap + fast, high
# accuracy), Clef (mid cost, fastest-ish, most accurate), cohere-4-pro
# (most expensive, slow, mid accuracy)
FRAME = pd.DataFrame([
    {"System": "Jev", "Accuracy %": 93.8, "$ per question": 5.3e-6,
     "Latency s": 0.018},
    {"System": "Clef (Workers AI)", "Accuracy %": 97.9,
     "$ per question": 2.3e-5, "Latency s": 0.052},
    {"System": "cohere-rerank-4-pro (OpenRouter)", "Accuracy %": 83.3,
     "$ per question": 2.5e-3, "Latency s": 0.418},
])


class IsoBasisTest(unittest.TestCase):
    def test_basis_is_a_rotation(self):
        ex, ey, ez = app._iso_basis()
        for axis in (ex, ey, ez):
            self.assertAlmostEqual(sum(c * c for c in axis), 1.0)
        for a, b in ((ex, ey), (ex, ez), (ey, ez)):
            self.assertAlmostEqual(sum(x * y for x, y in zip(a, b)), 0.0)

    def test_axis_directions(self):
        # accuracy points straight up (foreshortened by the pitch),
        # cost to the lower-right, latency to the lower-left
        ex, ey, ez = app._iso_basis()
        self.assertAlmostEqual(ez[0], 0.0)
        self.assertAlmostEqual(ez[1], math.cos(app._ISO_PITCH))
        self.assertGreater(ex[0], 0)
        self.assertLess(ex[1], 0)
        self.assertLess(ey[0], 0)
        self.assertLess(ey[1], 0)


class CostSpeedGeometryTest(unittest.TestCase):
    def test_markers_inside_canvas_above_their_shadows(self):
        g = app._cost_speed_geometry(FRAME)
        self.assertEqual(len(g["points"]), 3)
        self.assertEqual(len(g["feet"]), 3)
        self.assertEqual(len(g["assign"]), 3)
        for p, f in zip(g["points"], g["feet"]):
            for x in (p[0], f[0]):
                self.assertGreater(x, 0)
                self.assertLess(x, app._COST_SPEED_W)
            for y in (p[1], f[1]):
                self.assertGreater(y, 0)
                self.assertLess(y, app._COST_SPEED_H)
            # every fixture system has accuracy > 0, so its marker
            # sits above its floor shadow
            self.assertLess(p[1], f[1])

    def test_axes_and_ticks_present(self):
        g = app._cost_speed_geometry(FRAME)
        self.assertEqual(len(g["seg_floor"]), 4)
        self.assertEqual(len(g["seg_axes"]), 3)
        self.assertEqual(len(g["titles"]), 3)
        # the fixture spans 5.3e-6 .. 2.5e-3 $/q, so at least the
        # 1e-5..1e-3 decades and the 0.1..0.4 s ticks must appear
        self.assertGreaterEqual(len(g["labels_cost"]), 3)
        self.assertGreaterEqual(len(g["labels_lat"]), 3)
        # accuracy ticks are the 20-multiples inside the (possibly
        # truncated) accuracy domain: the fixture's 83.3..97.9 span
        # floors the pole at 80, so only the 100 tick remains — 20/60
        # disappeared with the empty 0-80 stretch
        dom = app._accuracy_domain(FRAME["Accuracy %"])
        expected_acc = [a for a in range(20, 101, 20)
                        if dom and dom[0] < a <= dom[1]]
        self.assertEqual([int(l[2]) for l in g["labels_acc"]],
                         expected_acc)

    def test_leader_line_to_accuracy_pole(self):
        # one faint leader per marker, from the dot to the accuracy
        # axis at exactly the dot's height: it starts at the marker,
        # lands on the pole between base and top, and a higher-
        # accuracy dot lands higher up the pole
        g = app._cost_speed_geometry(FRAME)
        self.assertEqual(len(g["seg_leaders"]), 3)
        pole_x = g["seg_axes"][0][0][0]  # the accuracy axis is first
        landings = []
        for point, leader in zip(g["points"], g["seg_leaders"]):
            self.assertEqual(len(leader), 2)
            self.assertAlmostEqual(leader[0][0], point[0], delta=0.5)
            self.assertAlmostEqual(leader[0][1], point[1], delta=0.5)
            self.assertAlmostEqual(leader[1][0], pole_x, delta=1.5)
            landings.append(leader[1][1])
        pole_base, pole_top = g["seg_axes"][0]
        for y in landings:
            self.assertGreater(y, pole_top[1])
            self.assertLess(y, pole_base[1])
        # fixture accuracy: Clef 97.9 > Jev 93.8 > cohere 83.3, and
        # up the pole is smaller screen y
        self.assertLess(landings[1], landings[0])
        self.assertLess(landings[0], landings[2])

    def test_expensive_slow_sits_toward_the_front(self):
        # cohere-4-pro (max cost, max latency) must sit lower on
        # screen than Jev (min cost, min latency): the origin corner is
        # the far/top corner, the (1, 1) floor corner the near one
        g = app._cost_speed_geometry(FRAME)
        jev, cohere = g["points"][0], g["points"][2]
        self.assertGreater(cohere[1], jev[1])

    def test_per_system_hues_and_lightened_leaders(self):
        # one distinct hue per system, assigned by sorted system name
        # so the web UI and the PNG agree; leader lines carry the same
        # hue blended toward white, labels the same hue toward black
        g = app._cost_speed_geometry(FRAME)
        self.assertEqual(len(g["colors"]), 3)
        self.assertEqual(len(set(g["colors"])), 3)
        for dot, leader, label in zip(g["colors"], g["leader_colors"],
                                      g["label_colors"]):
            self.assertRegex(dot, r"#[0-9a-f]{6}")
            self.assertRegex(leader, r"#[0-9a-f]{6}")
            self.assertRegex(label, r"#[0-9a-f]{6}")
            for i in (1, 3, 5):
                self.assertGreaterEqual(int(leader[i:i + 2], 16),
                                        int(dot[i:i + 2], 16))
                self.assertLessEqual(int(label[i:i + 2], 16),
                                     int(dot[i:i + 2], 16))
        # row order must not change the hue a system gets
        g2 = app._cost_speed_geometry(FRAME.iloc[::-1])
        self.assertEqual(dict(zip(g["names"], g["colors"])),
                         dict(zip(g2["names"], g2["colors"])))

    def test_lighten_blends_toward_white(self):
        self.assertEqual(app._lighten("#000000", 0.5), "#808080")
        self.assertEqual(app._lighten("#ffffff", 0.5), "#ffffff")
        self.assertEqual(app._lighten("#4c78a8", 0.0), "#4c78a8")

    def test_darken_blends_toward_black(self):
        self.assertEqual(app._darken("#ffffff", 0.5), "#808080")
        self.assertEqual(app._darken("#000000", 0.5), "#000000")
        self.assertEqual(app._darken("#4c78a8", 0.0), "#4c78a8")

    def test_chart_serializes(self):
        chart = app.cost_speed_scatter(FRAME, "t")
        spec = chart.to_dict()
        self.assertIn("layer", spec)
        # frame + axes + ticks + 3 tick-label layers + 3 titles +
        # drop lines + shadows + points + labels
        self.assertGreater(len(spec["layer"]), 10)
        # per-system hue wired through as literal color values, for
        # the dots and the hue-matched labels alike
        dumped = json.dumps(spec)
        self.assertIn('"_c"', dumped)
        self.assertIn('"_c_dark"', dumped)


class LatencyColumnTest(unittest.TestCase):
    def test_cost_summary_frame_carries_latency(self):
        systems = {"Jev": {"cls_accuracy": 0.8,
                           "cls_mean_latency_s": 0.018,
                           "usage": {"paid_requests": 2, "cost_usd": 0.0,
                                     "input_tokens": 6094}}}
        frame = app.cost_summary_frame(systems, n_q=48)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "Latency s"], 0.018)

    def test_missing_latency_becomes_zero(self):
        systems = {"Jev": {"cls_accuracy": 0.8,
                           "usage": {"paid_requests": 2, "cost_usd": 0.0,
                                     "input_tokens": 6094}}}
        frame = app.cost_summary_frame(systems, n_q=48)
        self.assertEqual(frame.loc[0, "Latency s"], 0.0)

    def test_rows_without_latency_stay_off_the_projection(self):
        df = FRAME.copy()
        df.loc[0, "Latency s"] = 0.0
        g = app._cost_speed_geometry(
            df[pd.to_numeric(df["Latency s"], errors="coerce")
               .fillna(0) > 0])
        self.assertEqual(len(g["points"]), 2)


if __name__ == "__main__":
    unittest.main()
