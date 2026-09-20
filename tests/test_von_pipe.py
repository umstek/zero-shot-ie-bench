import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


class VonPipeTests(unittest.TestCase):
    def test_non_ascii_choices_round_trip_through_windows_code_page(self):
        helper = Path(__file__).resolve().parents[1] / "von_demo.py"
        script = """
import runpy, sys, types
def decide(**kw):
    label = next(iter(kw['choices']))
    return types.SimpleNamespace(choice=label, probabilities={label: 1.0}, confidence=1.0)
sys.modules['von_client'] = types.SimpleNamespace(load_von_decider=lambda: decide)
runpy.run_path(sys.argv[1], run_name='__main__')
"""
        for label in ("你好", "හොඳයි"):
            with self.subTest(label=label):
                proc = subprocess.run(
                    [sys.executable, "-c", script, str(helper)],
                    input=json.dumps({"texts": ["test"], "instructions": "test",
                                      "choices": {label: None}}),
                    capture_output=True, text=True, encoding="cp1252", timeout=30,
                    env={**os.environ, "PYTHONIOENCODING": "cp1252"})
                self.assertEqual(proc.returncode, 0, proc.stderr)
                result = json.loads(proc.stdout)["results"][0]
                self.assertEqual(result["choice"], label)
                self.assertEqual(result["probabilities"], {label: 1.0})


if __name__ == "__main__":
    unittest.main()
