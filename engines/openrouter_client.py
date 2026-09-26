"""Minimal Python client for OpenRouter's decision endpoints.

Two contracts, both behind one Bearer key:

- ``/api/v1/systemone`` speaks the TypeSafe System One wire format (the same
  one engines/jev_client.py implements), with the ``model`` field selecting
  the serving checkpoint (jaredpalmer/kev-4b, respan/span-01, ...). Kev and
  Jev answer ``choice`` questions; Span-01 is a behavior scorer, so it is
  addressed with one ``noul`` (yes-probability) question per label and
  rejects an explicit ``criteria: null`` - omit the field instead.
- ``/api/v1/rerank`` is the Cohere-style {model, query, documents} contract
  returning relevance scores, used here as a decision engine the same way
  the local cross-encoder rerankers are: score one (instruction, label)
  pair per label, pick the highest.

API key: set the OPENROUTER_API_KEY environment variable, or put
`OPENROUTER_API_KEY=...` in a `.env` file in the repo root (gitignored).
Paid API; each call is a request. Account privacy settings that enforce
ZDR providers exclude every rerank endpoint and Span-01 (only Kev's
SiliconFlow endpoint passes).

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
SYSTEMONE_URL = f"{OPENROUTER_BASE}/systemone"
RERANK_URL = f"{OPENROUTER_BASE}/rerank"


def load_openrouter_key() -> str:
    """OPENROUTER_API_KEY from the environment, else from a local .env file."""
    env = os.environ.get("OPENROUTER_API_KEY")
    if env:
        return env.strip()
    # .env lives in the repo root, one level above engines/
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def systemone(model: str):
    """A JevClient pointed at OpenRouter's /systemone router."""
    from engines.jev_client import JevClient

    api_key = load_openrouter_key()
    if not api_key:
        raise RuntimeError(
            "No OPENROUTER_API_KEY found. Set the environment variable "
            "or create a .env file in the repo root with "
            "OPENROUTER_API_KEY=... (see README)."
        )
    return JevClient(api_key=api_key, model=model, base_url=SYSTEMONE_URL)


def _post(url: str, body: dict, timeout: int = 120) -> dict:
    headers = {"content-type": "application/json",
               "authorization": f"Bearer {load_openrouter_key()}"}
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     method="POST", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc


def rerank(model: str, query: str, documents: list[str],
           top_n: int | None = None) -> list[dict]:
    """Relevance-score every document against the query.

    Returns the endpoint's results rows ({"index": i, "relevance_score": s})
    in the order the endpoint sends them; callers rank by index themselves.
    """
    body = {"model": model, "query": query, "documents": documents}
    if top_n is not None:
        body["top_n"] = top_n
    payload = _post(RERANK_URL, body)
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"rerank response missing results: {payload!r:.300}")
    return results


def rerank_classify(model: str, instruction: str,
                    labels: list[str]) -> str | None:
    """Pick the label whose (instruction, label) pair scores highest."""
    rows = rerank(model, instruction, labels)
    scores = {row["index"]: row["relevance_score"] for row in rows}
    if not scores:
        return None
    return labels[max(scores, key=scores.get)]


def noul_classify(model: str, text: str, question: str,
                  labels: list[str]) -> str | None:
    """Span-01 style behavior scoring as classification: one noul
    (yes-probability) question per label in a single request, argmax wins.
    The criteria key must stay absent - Span rejects an explicit null."""
    client = systemone(model)
    questions = {f"l{i}": {"type": "noul", "instructions": question.format(label=label)}
                 for i, label in enumerate(labels)}
    out = client.ask({"text": text}, questions)
    probs = {i: out["answers"][f"l{i}"].get("noul", 0.0)
             for i in range(len(labels))}
    return labels[max(probs, key=probs.get)]
