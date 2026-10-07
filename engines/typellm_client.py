"""Minimal client for TypeLLM's hosted type-safe generation API.

Wire format follows the REST reference (https://typellm.ai/docs/api):
POST {context, questions, images?, model?, options?, timeout?} to
https://api.typellm.ai/v1/generate with a Bearer key. Questions are JSON
Schema fields with TypeLLM keys - enum choices answer from a one-token
constrained pick:

    {"type": "string", "enum": [label, ...], "instructions": str,
     "thinking": true, "thinking_budget": 1024, ...}

At most 64 questions and 8 base64 data:image/... URIs per call; enums
hold at most 24 values. The response arrives flat: {id, model,
result: {name: value}, thinking: {name: str}, usage: {input_tokens,
thinking_tokens}, elapsed} - usage reports tokens but no $, so cost
charts derive TypeLLM's cost as measured tokens x the published prices
(app.derived_cost): input $0.05/M, thinking $0.50/M, answer tokens free.

TypeLLM (github.com/TypeLLM/TypeLLM, Apache 2.0) is a type-safe
generation harness over autoregressive LLMs - inspired by TypeSafe AI's
Jev, schema-guaranteed outputs via SGLang constrained decoding - and the
hosted API answers as typellm-latest (their Qwen3.8-27B deployment;
vision-capable, hence the images field). GET /v1/models lists the public
models a key can use.

The API's own Python client (pip `typellm`) retries a 429, a 5xx other
than 504 and a connection error up to twice; generate() here does the
same so one transient blip cannot kill a 54-request bench run. The
`timeout` request field maxes at 90 s.

API key: set the TYPELLM_API_KEY environment variable, or put
`TYPELLM_API_KEY=...` in a `.env` file in the repo root (gitignored).
Paid API; each generate() call is a request.

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import email.utils
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

BASE_URL = "https://api.typellm.ai"
GENERATE_URL = BASE_URL + "/v1/generate"
MODELS_URL = BASE_URL + "/v1/models"
DEFAULT_MODEL = "typellm-latest"
# TypeLLM's published prices, $ per 1M tokens: input and thinking bill,
# answer tokens are free, so these two are the whole cost
INPUT_USD_PER_MTOK = 0.05
THINKING_USD_PER_MTOK = 0.50
# the API's own ceilings (docs/api): the `timeout` request field maxes at
# 90 s and one call carries at most 64 questions
MAX_TIMEOUT_S = 90
MAX_QUESTIONS = 64
# Retry-After is advisory: a far-future or clock-skewed date must not
# stall a bench run, so honor it only up to this many seconds
MAX_RETRY_AFTER_S = 60.0


def load_api_key() -> str:
    """TYPELLM_API_KEY from the environment, else from a local .env file."""
    env = os.environ.get("TYPELLM_API_KEY")
    if env:
        return env.strip()
    # .env lives in the repo root, one level above engines/
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("TYPELLM_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def models(timeout: int = 30) -> dict:
    """GET /v1/models - the public models this key can use."""
    request = urllib.request.Request(
        MODELS_URL, method="GET",
        headers={"authorization": f"Bearer {_require_key()}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def enum_question(instructions: str, values: list, thinking: bool = False,
                  thinking_budget: int = 1024,
                  return_probabilities: bool = False) -> dict:
    """A choice question: one constrained pick among `values` (at most
    24). thinking=True spends reasoning tokens first ($0.50/M; budget
    caps them, 4096 max). return_probabilities=True answers with
    {"value", "probabilities"} instead of the bare value (demos; the
    benches take the bare pick like every other system)."""
    if not 2 <= len(values) <= 24:
        raise ValueError(f"enum needs 2-24 values, got {len(values)}")
    question = {"type": "string", "enum": list(values),
                "instructions": instructions}
    if thinking:
        question["thinking"] = True
        question["thinking_budget"] = thinking_budget
    if return_probabilities:
        question["return_probabilities"] = True
    return question


def generate(context: str, questions: dict, images: list[str] | None = None,
             model: str = DEFAULT_MODEL, timeout: int = 60,
             max_retries: int = 2, usage_sink=None) -> dict:
    """One POST /v1/generate; returns the flat response payload plus
    _latency_s. The response must carry a result object answering every
    asked question - anything less raises before the payload reaches
    callers or usage accounting. Retries a 429 (honoring Retry-After in
    either RFC 9110 form), a 5xx other than 504 and a connection error
    up to `max_retries` times - the same policy the pip `typellm`
    client documents."""
    if not questions or len(questions) > MAX_QUESTIONS:
        raise ValueError(f"1-{MAX_QUESTIONS} questions per call, got "
                         f"{len(questions)}")
    _require_key()
    request_body = {"context": context, "questions": questions,
                    "model": model, "timeout": timeout}
    if images:
        request_body["images"] = images
    body = json.dumps(request_body).encode("utf-8")
    last_error = None
    for attempt in range(max_retries + 1):
        request = urllib.request.Request(
            GENERATE_URL, data=body, method="POST",
            headers={"content-type": "application/json",
                     "authorization": f"Bearer {load_api_key()}"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout + 30
                                        ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            _validate_response(payload, questions)
            if usage_sink:
                usage_sink(payload.get("usage"))
            payload["_latency_s"] = round(time.perf_counter() - t0, 3)
            return payload
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            retryable = exc.code == 429 or (500 <= exc.code < 600
                                            and exc.code != 504)
            if not retryable or attempt == max_retries:
                raise RuntimeError(
                    f"TypeLLM HTTP {exc.code}: {detail}") from exc
            last_error = RuntimeError(
                f"TypeLLM HTTP {exc.code}: {detail}")
            if exc.code == 429:
                delay = _retry_after_seconds(
                    exc.headers.get("Retry-After", ""))
                if delay is not None:
                    time.sleep(delay)
                    continue
        except urllib.error.URLError as exc:
            # connection error (DNS, reset, timeout) - retryable
            last_error = RuntimeError(f"TypeLLM connection error: {exc}")
            if attempt == max_retries:
                raise last_error from exc
        time.sleep(2 ** attempt)
    raise last_error  # unreachable; the loop returns or raises


def _validate_response(payload, questions: dict) -> None:
    """Reject anything but the documented flat response: a result object
    holding an answer for every asked question. A null result or a
    dropped answer would otherwise surface as a KeyError in the demo or
    silently score as a wrong prediction in the benches."""
    if not isinstance(payload, dict):
        raise RuntimeError(f"TypeLLM response is not an object: "
                           f"{payload!r:.300}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError(f"TypeLLM response result is not an object: "
                           f"{payload!r:.300}")
    missing = sorted(name for name in questions if name not in result)
    if missing:
        raise RuntimeError(f"TypeLLM response has no answers for "
                           f"{missing}: {payload!r:.300}")


def _retry_after_seconds(value: str):
    """Retry-After as a sleep in seconds, or None to use the caller's
    own backoff. RFC 9110 allows delay-seconds or an HTTP-date; both are
    accepted, and either form is clamped to [0, MAX_RETRY_AFTER_S] so an
    absurd value cannot stall a bench run."""
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER_S)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    delay = (when - datetime.now(timezone.utc)).total_seconds()
    return min(max(delay, 0.0), MAX_RETRY_AFTER_S)


def _require_key() -> str:
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "No TYPELLM_API_KEY found. Set the environment variable or "
            "create a .env file in the repo root with TYPELLM_API_KEY=... "
            "(see README)."
        )
    return api_key
