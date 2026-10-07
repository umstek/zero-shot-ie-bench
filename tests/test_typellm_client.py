"""Offline regression tests for engines/typellm_client.py.

No network, no key: the response gate (a malformed result must never
reach callers or usage accounting - the demo would KeyError and the
benches would score a dropped answer as a wrong prediction) and the
429 Retry-After clock (either RFC 9110 form, clamped, with a fallback
to the caller's own backoff).
"""

import email.message
import email.utils
import io
import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
from urllib.error import HTTPError

from engines import typellm_client as tlc

GOOD_PAYLOAD = {"id": "r1", "model": "typellm-latest",
                "result": {"q": "positive"},
                "usage": {"input_tokens": 10, "thinking_tokens": 0}}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code, headers=None, body=b""):
    message = email.message.Message()
    for name, value in (headers or {}).items():
        message[name] = value
    return HTTPError(tlc.GENERATE_URL, code, "err", message,
                     io.BytesIO(body))


def _gmt(delta_s):
    """An RFC 9110 HTTP-date `delta_s` seconds from now."""
    when = datetime.now(timezone.utc) + timedelta(seconds=delta_s)
    return email.utils.formatdate(when.timestamp(), usegmt=True)


def _generate(payloads, questions=None, **kwargs):
    """Run generate() against canned responses (exception objects are
    raised as-is); returns (payload-or-RuntimeError, usage-sink calls,
    sleeps)."""
    usage, sleeps = [], []
    questions = questions if questions is not None else {
        "q": tlc.enum_question("sentiment", ["positive", "negative"])}
    side = [p if isinstance(p, BaseException) else _FakeResponse(p)
            for p in payloads]
    with patch.dict("os.environ", {"TYPELLM_API_KEY": "test-key"}), \
            patch("urllib.request.urlopen", side_effect=side), \
            patch("time.sleep", side_effect=lambda s: sleeps.append(s)), \
            patch("time.perf_counter", side_effect=[0.0, 1.0] * 8):
        try:
            out = tlc.generate("ctx", questions, usage_sink=usage.append,
                               **kwargs)
        except RuntimeError as exc:
            out = exc
    return out, usage, sleeps


class ResponseGateTests(unittest.TestCase):
    def test_good_round_trip_forwards_usage_and_adds_latency(self):
        out, usage, _ = _generate([GOOD_PAYLOAD])
        self.assertEqual(out["result"], {"q": "positive"})
        self.assertEqual(out["_latency_s"], 1.0)
        self.assertEqual(usage, [{"input_tokens": 10,
                                  "thinking_tokens": 0}])

    def test_null_result_raises_and_never_reaches_usage(self):
        out, usage, _ = _generate([{"id": "r1", "result": None,
                                    "usage": {"input_tokens": 10}}])
        self.assertIsInstance(out, RuntimeError)
        self.assertIn("result is not an object", str(out))
        self.assertEqual(usage, [])

    def test_non_object_payload_raises(self):
        out, usage, _ = _generate([["not", "a", "response"]])
        self.assertIsInstance(out, RuntimeError)
        self.assertIn("response is not an object", str(out))
        self.assertEqual(usage, [])

    def test_missing_answer_names_the_question(self):
        questions = {"t0": tlc.enum_question("a", ["x", "y"]),
                     "t1": tlc.enum_question("b", ["x", "y"])}
        payload = {"result": {"t0": "x"}, "usage": {"input_tokens": 3}}
        out, usage, _ = _generate([payload], questions=questions)
        self.assertIsInstance(out, RuntimeError)
        self.assertIn("['t1']", str(out))
        self.assertEqual(usage, [])

    def test_malformed_body_fails_without_retrying(self):
        # a 200 with a malformed body must fail the call, not burn
        # retries into a fresh billable request
        out, usage, sleeps = _generate(
            [{"id": "r1", "result": None, "usage": {}}])
        self.assertIsInstance(out, RuntimeError)
        self.assertEqual(usage, [])
        self.assertEqual(sleeps, [])


class RetryAfterTests(unittest.TestCase):
    def test_delay_seconds_pass_through(self):
        self.assertEqual(tlc._retry_after_seconds("2.5"), 2.5)

    def test_delay_seconds_clamped_to_the_cap(self):
        self.assertEqual(tlc._retry_after_seconds("3600"),
                         tlc.MAX_RETRY_AFTER_S)

    def test_negative_delay_clamps_to_zero(self):
        self.assertEqual(tlc._retry_after_seconds("-3"), 0.0)

    def test_http_date_soon_parses_to_its_delay(self):
        self.assertAlmostEqual(tlc._retry_after_seconds(_gmt(40)),
                               40.0, delta=2.0)

    def test_http_date_past_and_far_future_clamp(self):
        self.assertEqual(tlc._retry_after_seconds(_gmt(-3600)), 0.0)
        self.assertEqual(tlc._retry_after_seconds(_gmt(86400)),
                         tlc.MAX_RETRY_AFTER_S)

    def test_invalid_header_falls_back_to_none(self):
        self.assertIsNone(tlc._retry_after_seconds("soon"))
        self.assertIsNone(tlc._retry_after_seconds(""))

    def test_429_with_http_date_is_honored_then_succeeds(self):
        out, usage, sleeps = _generate(
            [_http_error(429, {"Retry-After": _gmt(5)}), GOOD_PAYLOAD])
        self.assertEqual(out["result"], {"q": "positive"})
        self.assertEqual(len(sleeps), 1)
        self.assertLessEqual(sleeps[0], tlc.MAX_RETRY_AFTER_S)
        self.assertEqual(usage, [GOOD_PAYLOAD["usage"]])

    def test_429_with_invalid_header_uses_exponential_backoff(self):
        out, _, sleeps = _generate(
            [_http_error(429, {"Retry-After": "soon"}), GOOD_PAYLOAD])
        self.assertEqual(out["result"], {"q": "positive"})
        self.assertEqual(sleeps, [1.0])  # 2 ** 0, not a Retry-After value


if __name__ == "__main__":
    unittest.main()
