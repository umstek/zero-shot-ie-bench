"""Minimal client for Fastino's hosted GLiDE and GLiNER models.

Two contracts, both behind one Bearer key (docs.fastino.ai/inference):

- GLiDE rides TypeSafe's System One wire format (the one
  engines/jev_client.py implements) behind Fastino's REST: POST
  {model, state, questions} to https://api.fastino.ai/v1/systemone, and
  the payload arrives flat ({model, answers, usage}) - no envelope to
  unwrap. Questions and answers are Jev-compatible; choice criteria
  descriptions may be null.
- The hosted GLiNER models - the extraction checkpoints AND the
  decision-tuned GLiNER-2.5-Decide - ride the OpenAI-compatible
  chat-completions endpoint instead: POST {model, messages:
  [{role: "user", content: <text>}], schema, ...} to
  https://api.fastino.ai/v1/chat/completions, where the top-level
  ``schema`` picks the tasks (entities / classifications / structures /
  relations) and the result is a JSON string in
  choices[0].message.content (docs.fastino.ai/inference/
  chat-completions; the native POST /v1/gliner-2 that the OpenAPI spec
  still lists serves one fixed GLiNER2-base deployment - it ignores any
  model field, verified 2026-10-05 against the returned inference
  records, so it is not usable for the hosted twins).

GLiDE (Generalized Lightweight Decision Engine, released Sep 2026) is a
"thinking decision model": one fast pass, then extra reasoning tokens when
the leading option is uncertain. usage.output_tokens counts those thinking
tokens, but they are priced at $0 - only input bills, at $0.15/M (Fastino's
pricing page; GET /v1/base-models agrees). 40k context, served on one B200,
and the model catalog flags it ZDR. The response usage block reports tokens
but no $, so cost charts derive GLiDE's cost as measured input tokens x
that price (app.derived_cost) - the same rule the README's cost table
applies to the Clef family and Jev.

The hosted GLiNER models are the same open-weight checkpoints this repo
benches locally through the gliner2 package (fastino/gliner2.5-*-v1,
fastino/GLiNER-2.5-Decide - see bench.py and bench_spectrum.py), so their
client here mirrors the local AutoExtractor surface (classify_text /
extract_entities) one request per call. All three catalog ids bill input
tokens at $0.03/M with output at $0 (GET /v1/base-models and
docs.fastino.ai/concepts/models agree; same token-x-price derivation as
GLiDE). fastino/gliner2.5-small-v1 is deliberately absent: the small
checkpoint exists on Hugging Face but Fastino does not host it (absent
from GET /v1/base-models and the docs' model table; POST /v1/chat/
completions with that id answers 404 "not a recognised model id").

Decide does NOT ride /v1/systemone despite being a decision model: it is
a GLiNER encoder addressed through chat completions with a
``classifications`` schema (docs.fastino.ai/concepts/gliner-2-5-decide);
only GLiDE speaks System One.

API key: set the FASTINO_API_KEY environment variable, or put
`FASTINO_API_KEY=...` in a `.env` file in the repo root (gitignored).
Paid API; each ask() call is a request.

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

# body "model" selector -> endpoint model id (lowercase fastino/glide is
# accepted too; the catalog's canonical id is used)
MODELS = {"glide": "fastino/GLiDE"}
SYSTEM_ONE_URL = "https://api.fastino.ai/v1/systemone"
# the hosted GLiNER models' endpoint (extraction AND Decide)
CHAT_COMPLETIONS_URL = "https://api.fastino.ai/v1/chat/completions"
# body "model" selectors for the hosted GLiNER twins; the local trio's
# small checkpoint is not hosted, so no selector for it (see module
# docstring)
GLINER_MODELS = {
    "gliner2.5-base": "fastino/gliner2.5-base-v1",
    "gliner2.5-multi": "fastino/gliner2.5-multi-v1",
    "decide": "fastino/GLiNER-2.5-Decide",
}
# Fastino's published input prices, $ per 1M input tokens (GET
# /v1/base-models; docs.fastino.ai/concepts/models agrees). Output tokens
# are $0 for every model here (GLiDE's thinking tokens included), so input
# pricing is the whole cost
INPUT_USD_PER_MTOK = {"glide": 0.15,
                      "gliner2.5-base": 0.03,
                      "gliner2.5-multi": 0.03,
                      "decide": 0.03}

# cold starts: bounded backoff for 425 (warming) / 429 / 503, Retry-After
# honored when present (docs.fastino.ai/inference "Cold starts and
# retries" also asks for a >=300 s read timeout)
CHAT_WAITS = (20, 40, 60, 60, 60)


def load_api_key() -> str:
    """FASTINO_API_KEY from the environment, else from a local .env file."""
    env = os.environ.get("FASTINO_API_KEY")
    if env:
        return env.strip()
    # .env lives in the repo root, one level above engines/
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("FASTINO_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def glide(usage_sink=None):
    """A JevClient pointed at Fastino's hosted GLiDE."""
    from engines.jev_client import JevClient

    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "No FASTINO_API_KEY found. Set the environment variable or "
            "create a .env file in the repo root with FASTINO_API_KEY=... "
            "(see README)."
        )
    return JevClient(api_key=api_key, model=MODELS["glide"],
                     base_url=SYSTEM_ONE_URL, usage_sink=usage_sink)


class HostedGliner:
    """A hosted GLiNER twin speaking the local AutoExtractor surface.

    classify_text(text, {task: [labels]}) -> {task: label} and
    extract_entities(text, labels, ...) -> {"entities": {label:
    [{start, end, text}]}}: the same shapes engines/bench call sites
    already read off gliner2's AutoExtractor, so the bench drivers run
    the hosted twins through the same code paths as the local
    checkpoints (one request per call - the hosted endpoint takes one
    conversation per request and the local bench is one call per text
    anyway). Output-token and confidence fields are parsed but dropped;
    callers that stored include_confidence=False locally get the same
    lean dicts.
    """

    def __init__(self, model: str, usage_sink=None, timeout: int = 300):
        self.model = model
        self.usage_sink = usage_sink
        self.timeout = timeout

    # -- plumbing ------------------------------------------------------

    def _post_chat(self, body: dict) -> dict:
        headers = {"content-type": "application/json",
                   "authorization": "Bearer " + load_api_key()}
        for attempt in range(len(CHAT_WAITS) + 1):
            request = urllib.request.Request(
                CHAT_COMPLETIONS_URL, data=json.dumps(body).encode("utf-8"),
                method="POST", headers=headers)
            try:
                with urllib.request.urlopen(request,
                                            timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (425, 429, 503) and attempt < len(CHAT_WAITS):
                    # a Retry-After header lifts the fixed ladder step,
                    # never past a 2-minute cap
                    try:
                        retry_after = float(exc.headers.get("Retry-After", 0))
                    except (TypeError, ValueError):
                        retry_after = 0.0
                    time.sleep(min(max(retry_after, CHAT_WAITS[attempt]), 120))
                    continue
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise RuntimeError(
                    f"Fastino HTTP {exc.code}: {detail}") from exc
        raise RuntimeError("Fastino retry loop exhausted")  # unreachable

    def _ask(self, text: str, schema: dict) -> dict:
        payload = self._post_chat({
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "schema": schema,
            "threshold": 0.5,  # the local bench runs AutoExtractor defaults
        })
        usage = payload.get("usage") or {}
        if self.usage_sink:
            # chat-completions shape -> the sink's input/total row; output
            # tokens are $0 for every model here, so input is the whole cost
            row = {}
            if usage.get("prompt_tokens") is not None:
                row["input_tokens"] = usage["prompt_tokens"]
            if usage.get("total_tokens") is not None:
                row["total_tokens"] = usage["total_tokens"]
            self.usage_sink(row)
        try:
            content = payload["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Fastino chat-completions response without a JSON "
                f"content string: {payload!r:.300}") from exc

    # -- the local AutoExtractor surface --------------------------------

    def classify_text(self, text: str, tasks: dict) -> dict:
        """One classification task per request (the hosted schema carries
        several, but the benches ask one question at a time). tasks maps
        task name -> label list; returns task name -> winning label
        (absent when the model scores nothing above threshold)."""
        out = {}
        for task, labels in tasks.items():
            schema = {"classifications": [
                {"task": task, "labels": list(labels),
                 "multi_label": False, "top_k": 1}]}
            content = self._ask(text, schema)
            # observed shape: {"<task>": {"label": ..., "confidence": ...}};
            # tolerate the docs' wrapped {"classifications": {...}} form
            node = content.get("classifications", content) \
                if isinstance(content, dict) else {}
            answer = node.get(task) if isinstance(node, dict) else None
            out[task] = answer.get("label") if isinstance(answer, dict) \
                else answer
        return out

    def extract_entities(self, text: str, labels,
                         include_spans: bool = True,
                         include_confidence: bool = False) -> dict:
        """NER over the given labels; returns the local
        {"entities": {label: [{start, end, text}]}} shape (confidence
        parsed but dropped, matching the local include_confidence=False
        calls; the include_spans flag is accepted for signature parity -
        the hosted endpoint always reports offsets)."""
        schema = {"entities": [{"name": label, "description": label}
                               for label in labels]}
        content = self._ask(text, schema)
        # observed shape: {"entities": {label: [{text, confidence, start,
        # end}]}}; tolerate the unwrapped {label: [...]} form seen on
        # single-label schemas
        ents = content.get("entities", content) if isinstance(content, dict) \
            else {}
        out = {}
        for label, items in (ents or {}).items():
            rows = []
            for item in items or []:
                if not isinstance(item, dict):
                    continue
                row = {"start": item.get("start"), "end": item.get("end"),
                       "text": item.get("text")}
                if include_confidence and item.get("confidence") is not None:
                    row["confidence"] = item["confidence"]
                rows.append(row)
            out[label] = rows
        return {"entities": out}


def gliner(selector: str, usage_sink=None) -> HostedGliner:
    """A hosted GLiNER twin for one of the GLINER_MODELS selectors."""
    if selector not in GLINER_MODELS:
        raise RuntimeError(f"unknown Fastino hosted GLiNER selector "
                           f"{selector!r}; known: {sorted(GLINER_MODELS)}")
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError(
            "No FASTINO_API_KEY found. Set the environment variable or "
            "create a .env file in the repo root with FASTINO_API_KEY=... "
            "(see README)."
        )
    return HostedGliner(GLINER_MODELS[selector], usage_sink=usage_sink)
