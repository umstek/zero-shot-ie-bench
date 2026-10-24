import sys
import types
import unittest
from unittest.mock import patch

from bench_multilingual import run_laya_router


class RouterTimingTests(unittest.TestCase):
    def test_alternating_routes_do_not_load_inside_timed_predictions(self):
        events = []

        class Router:
            def __init__(self, **kwargs):
                self.loaded = set()

            def route(self, state, questions):
                return {"model": state["text"]}

            def preload(self, names):
                events.append("preload")
                self.loaded.update(names)

            def predict(self, state, questions):
                model = state["text"]
                if model not in self.loaded:
                    raise AssertionError("Model loading leaked into inference")
                events.append("predict")
                return {"answers": {"q": {"choice": "neutral"}},
                        "routing": {"model": model}}

        def clock():
            events.append("clock")
            return float(len(events))

        with patch.dict(sys.modules, {"laya": types.SimpleNamespace(Router=Router)}):
            with patch("bench_multilingual.time.perf_counter", side_effect=clock):
                preds, _, routes = run_laya_router(
                    ["english", "multilingual", "english"], ["a", "b", "a"])
        self.assertEqual(events[0], "preload")
        self.assertEqual(events[1:], ["clock", "predict", "clock"] * 3)
        self.assertEqual(preds, ["neutral"] * 3)
        self.assertEqual(routes["a"], {"english": 2})


if __name__ == "__main__":
    unittest.main()
