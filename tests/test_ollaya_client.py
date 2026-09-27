"""Offline regression tests for engines/ollaya_client.py.

No server, no downloads: URL building, keyless client construction, and
registry consistency with the two benchmark drivers (a name drift between
engines.ollaya_client.MODELS and a bench registry would silently drop or
misroute a system).
"""

import unittest
from unittest.mock import patch

from engines.ollaya_client import MODELS, base_url, systemone


class BaseUrlTests(unittest.TestCase):
    def test_default_is_local_systemone(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("OLLAYA_BASE_URL", None)
            self.assertEqual(base_url(),
                             "http://127.0.0.1:11435/v1/systemone")

    def test_host_port_override_appends_the_systemone_path(self):
        with patch.dict("os.environ",
                        {"OLLAYA_BASE_URL": "http://localhost:11499/"}):
            self.assertEqual(base_url(),
                             "http://localhost:11499/v1/systemone")

    def test_full_url_override_wins_as_is(self):
        url = "http://example.test:1/api/decide/v1/systemone"
        with patch.dict("os.environ", {"OLLAYA_BASE_URL": url}):
            self.assertEqual(base_url(), url)


class ClientTests(unittest.TestCase):
    def test_client_targets_local_server_without_any_key(self):
        with patch.dict("os.environ", {}, clear=False):
            import os

            os.environ.pop("OLLAYA_BASE_URL", None)
            client = systemone("nli")
        self.assertEqual(client.model, "nli")
        self.assertEqual(client.base_url, base_url())
        # a local server never sees an authorization header
        self.assertEqual(client.api_key, "")


class RegistryTests(unittest.TestCase):
    def test_tags_are_nonempty_and_unique(self):
        self.assertEqual(len(set(MODELS.values())), len(MODELS))
        for name, tag in MODELS.items():
            self.assertTrue(tag, name)
            self.assertIn("(Ollaya)", name)

    def test_every_ollaya_system_is_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for name in MODELS:
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)


if __name__ == "__main__":
    unittest.main()
