"""The local D1 pair (d1-3B + d1-omni-600M, engines/d1_client.py) is
wired into both benches' registries and carries recorded rows in both
results files. The loader itself runs under .venv-von (trust_remote_code
AutoModel needs transformers >=5.14), so this checks the wiring, not the
model."""

import json
import unittest

import bench_multilingual
import bench_spectrum

D1_LOCAL = ("D1 3B (local)", "D1-omni 600M (local)")


class TestD1Wiring(unittest.TestCase):

    def test_both_bench_registries(self):
        for name in D1_LOCAL:
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)

    def test_results_rows(self):
        with open("results/bench_spectrum_results.json",
                  encoding="utf-8") as fh:
            spectrum = json.load(fh)
        with open("results/bench_multilingual_results.json",
                  encoding="utf-8") as fh:
            multilingual = json.load(fh)
        for name in D1_LOCAL:
            self.assertIn(name, spectrum["systems"])
            self.assertIn("cls_accuracy", spectrum["systems"][name])
            self.assertIn(name, multilingual["by_language"])


if __name__ == "__main__":
    unittest.main()
