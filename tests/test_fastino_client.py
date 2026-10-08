"""Offline regression tests for engines/fastino_client.py's hosted GLiNER
twins (the GLiDE path rides engines/jev_client.py and is covered by the
app.derived_cost tests).

No network: response parsing runs on recorded chat-completions shapes,
retries run against a stubbed urlopen with time.sleep patched out, and
registry checks pin the bench name maps against the client's selector
map (a name drift would silently drop a hosted twin from a benchmark).
"""

import email.message
import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from engines import fastino_client
from engines.fastino_client import (CHAT_COMPLETIONS_URL, GLINER_MODELS,
                                    INPUT_USD_PER_MTOK, HostedGliner,
                                    glide, gliner)

CHAT_URL = "https://api.fastino.ai/v1/chat/completions"


def _chat_payload(content: dict, prompt_tokens: int = 15,
                  total_tokens: int = 38) -> dict:
    return {"id": "chatcmpl-test", "object": "chat.completion",
            "model": "fastino/gliner2.5-base-v1",
            "choices": [{"index": 0,
                         "message": {"role": "assistant",
                                     "content": json.dumps(content)},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt_tokens,
                      "completion_tokens": total_tokens - prompt_tokens,
                      "total_tokens": total_tokens}}


class _FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._body


def _http_error(code: int, headers: str = "") -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    for line in headers.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            hdrs[key] = value.strip()
    return urllib.error.HTTPError(CHAT_URL, code, "err", hdrs,
                                  io.BytesIO(b""))


class RegistryTests(unittest.TestCase):
    def test_selectors_map_to_the_catalogs_canonical_ids(self):
        # GET /v1/base-models and docs.fastino.ai/concepts/models, both
        # checked 2026-10-05
        self.assertEqual(GLINER_MODELS, {
            "gliner2.5-base": "fastino/gliner2.5-base-v1",
            "gliner2.5-multi": "fastino/gliner2.5-multi-v1",
            "decide": "fastino/GLiNER-2.5-Decide",
        })

    def test_small_has_no_hosted_selector(self):
        # fastino/gliner2.5-small-v1 exists on Hugging Face but Fastino
        # does not host it (absent from the catalog; the id 404s on the
        # chat-completions endpoint) - a selector here would silently
        # invent an API
        self.assertNotIn("gliner2.5-small", GLINER_MODELS)
        self.assertNotIn("gliner2.5-small", INPUT_USD_PER_MTOK)

    def test_every_selector_has_a_published_input_price(self):
        # app.derived_cost prices a hosted system via
        # INPUT_USD_PER_MTOK[FASTINO_MODELS[system]] - a missing entry
        # would quietly put a paid system on a $0 chart
        self.assertIn("glide", INPUT_USD_PER_MTOK)
        for selector in GLINER_MODELS:
            self.assertIn(selector, INPUT_USD_PER_MTOK)

    def test_every_hosted_twin_is_registered_in_both_benchmarks(self):
        import bench_multilingual
        import bench_spectrum

        for name in bench_spectrum.FASTINO_EXTRACTORS:
            self.assertIn(name, bench_spectrum.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.ALL_SYSTEMS)
            self.assertIn(name, bench_multilingual.FASTINO_EXTRACTORS)
        # GLiDE keeps its batched-systemone registration in both
        self.assertIn("GLiDE (Fastino)", bench_spectrum.ALL_SYSTEMS)
        self.assertIn("GLiDE (Fastino)", bench_multilingual.ALL_SYSTEMS)


class BuilderTests(unittest.TestCase):
    def test_builder_selects_the_catalog_model(self):
        with patch.object(fastino_client, "load_api_key",
                          return_value="k"):
            client = gliner("decide")
        self.assertIsInstance(client, HostedGliner)
        self.assertEqual(client.model, "fastino/GLiNER-2.5-Decide")
        self.assertEqual(client.timeout, 300)  # docs: >=300 s read timeout

    def test_glide_still_builds_the_jev_systemone_client(self):
        with patch.object(fastino_client, "load_api_key",
                          return_value="k"):
            client = glide()
        self.assertEqual(client.model, "fastino/GLiDE")
        self.assertEqual(client.base_url, fastino_client.SYSTEM_ONE_URL)

    def test_unknown_selector_is_rejected_before_any_request(self):
        with patch.object(fastino_client, "load_api_key",
                          return_value="k"):
            with self.assertRaisesRegex(RuntimeError, "gliner2.5-small"):
                gliner("gliner2.5-small")

    def test_missing_key_raises_with_setup_hint(self):
        with patch.object(fastino_client, "load_api_key", return_value=""):
            with self.assertRaisesRegex(RuntimeError, "FASTINO_API_KEY"):
                gliner("gliner2.5-base")
            with self.assertRaisesRegex(RuntimeError, "FASTINO_API_KEY"):
                glide()


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.sink_rows = []
        self.client = HostedGliner(
            "fastino/gliner2.5-base-v1", usage_sink=self.sink_rows.append)

    def _serve(self, payload: dict):
        body = json.dumps(payload).encode()

        def fake_urlopen(request, timeout=None):
            self.assertEqual(request.full_url, CHAT_URL)
            self.assertEqual(request.headers["Authorization"], "Bearer k")
            return _FakeResponse(body)

        return fake_urlopen

    def test_classification_reads_the_observed_task_keyed_shape(self):
        # recorded 2026-10-05 from fastino/GLiNER-2.5-Decide
        content = {"sentiment": {"label": "positive",
                                 "confidence": 0.9823635816574097}}
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      self._serve(_chat_payload(content))):
            out = self.client.classify_text(
                "text", {"sentiment": ["positive", "negative", "neutral"]})
        self.assertEqual(out, {"sentiment": "positive"})

    def test_classification_tolerates_the_documented_wrapper(self):
        content = {"classifications": {
            "sentiment": {"label": "neutral", "confidence": 0.5}}}
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      self._serve(_chat_payload(content))):
            out = self.client.classify_text(
                "text", {"sentiment": ["positive", "negative", "neutral"]})
        self.assertEqual(out, {"sentiment": "neutral"})

    def test_extraction_reads_the_observed_wrapped_shape(self):
        content = {"entities": {
            "person": [{"text": "Marie Curie", "confidence": 1.0,
                        "start": 0, "end": 11}],
            "location": [{"text": "Warsaw", "confidence": 1.0,
                          "start": 24, "end": 30}]}}
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      self._serve(_chat_payload(content))):
            out = self.client.extract_entities(
                "Marie Curie was born in Warsaw.", ["person", "location"])
        self.assertEqual(out, {"entities": {
            "person": [{"start": 0, "end": 11, "text": "Marie Curie"}],
            "location": [{"start": 24, "end": 30, "text": "Warsaw"}]}})

    def test_extraction_tolerates_the_unwrapped_single_label_shape(self):
        # observed on a single-label schema: no "entities" wrapper
        content = {"city": [{"text": "Madrid", "confidence": 1.0,
                             "start": 50, "end": 56}]}
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      self._serve(_chat_payload(content))):
            out = self.client.extract_entities("... Madrid ...", ["city"])
        self.assertEqual(out, {"entities": {
            "city": [{"start": 50, "end": 56, "text": "Madrid"}]}})

    def test_request_carries_model_messages_schema_and_default_threshold(self):
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data.decode())
            captured["timeout"] = timeout
            return _FakeResponse(json.dumps(
                _chat_payload({"sentiment": {"label": "positive"}})).encode())

        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      fake_urlopen):
            self.client.classify_text(
                "some text", {"task": ["positive", "negative"]})
        body = captured["body"]
        self.assertEqual(body["model"], "fastino/gliner2.5-base-v1")
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": "some text"}])
        self.assertEqual(body["schema"], {"classifications": [
            {"task": "task", "labels": ["positive", "negative"],
             "multi_label": False, "top_k": 1}]})
        self.assertEqual(body["threshold"], 0.5)
        self.assertGreaterEqual(captured["timeout"], 300)

    def test_usage_sink_receives_prompt_tokens_as_input(self):
        rows = []
        client = HostedGliner("fastino/gliner2.5-base-v1",
                              usage_sink=rows.append)
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      self._serve(_chat_payload(
                          {"sentiment": {"label": "positive"}},
                          prompt_tokens=15, total_tokens=38))):
            client.classify_text("text", {"sentiment": ["a", "b"]})
        self.assertEqual(rows, [{"input_tokens": 15, "total_tokens": 38}])


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.client = HostedGliner("fastino/gliner2.5-base-v1")
        self.payload = json.dumps(
            _chat_payload({"sentiment": {"label": "positive"}})).encode()

    def test_warming_model_is_retried_then_served(self):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise _http_error(425)
            return _FakeResponse(self.payload)

        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      fake_urlopen), \
                patch("engines.fastino_client.time.sleep") as slept:
            out = self.client.classify_text("text", {"sentiment": ["a", "b"]})
        self.assertEqual(out, {"sentiment": "positive"})
        self.assertEqual(len(calls), 2)
        # first ladder step of CHAT_WAITS
        slept.assert_any_call(fastino_client.CHAT_WAITS[0])

    def test_retry_after_header_lifts_the_ladder_step(self):
        calls = []

        def fake_urlopen(request, timeout=None):
            calls.append(1)
            if len(calls) == 1:
                raise _http_error(425, "Retry-After: 45")
            return _FakeResponse(self.payload)

        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      fake_urlopen), \
                patch("engines.fastino_client.time.sleep") as slept:
            self.client.classify_text("text", {"sentiment": ["a", "b"]})
        slept.assert_any_call(45.0)

    def test_non_retryable_errors_fail_fast(self):
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      lambda *a, **kw: (_ for _ in ()).throw(
                          _http_error(404))), \
                patch("engines.fastino_client.time.sleep") as slept:
            with self.assertRaisesRegex(RuntimeError, "Fastino HTTP 404"):
                self.client.classify_text("text", {"sentiment": ["a", "b"]})
        slept.assert_not_called()

    def test_the_ladder_is_bounded(self):
        # exhausting the waits raises instead of looping forever
        with patch.object(fastino_client, "load_api_key", return_value="k"), \
                patch("engines.fastino_client.urllib.request.urlopen",
                      lambda *a, **kw: (_ for _ in ()).throw(
                          _http_error(503))), \
                patch("engines.fastino_client.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "Fastino HTTP 503"):
                self.client.classify_text("text", {"sentiment": ["a", "b"]})


if __name__ == "__main__":
    unittest.main()
