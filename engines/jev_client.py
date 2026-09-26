"""Minimal Python client for Jev (TypeSafe AI "System One" classifier).

Wire format follows @typesafe-ai/sdk 0.6.x: POST {model, state, questions}
to https://api.typesafe.ai/v1/systemone with a Bearer key. Questions:

    {"type": "choice", "instructions": str|obj, "criteria": {label: desc|None}}
    {"type": "score",  "instructions": ...,      "criteria": [desc, ...]}
    {"type": "noul",   "instructions": ...,      "criteria": optional}

Answers come back as {"answers": {name: {...}}, "usage": {...}}.

API key: set the TYPESAFE_API_KEY environment variable, or put
`TYPESAFE_API_KEY=...` in a `.env` file in the repo root (gitignored).
Jev is a paid API; each `ask()` call is a request.

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
REQUEST_LIMIT_TOKENS = 32_000  # jev-latest request ceiling


def load_api_key() -> str:
    """TYPESAFE_API_KEY from the environment, else from a local .env file."""
    env = os.environ.get("TYPESAFE_API_KEY")
    if env:
        return env.strip()
    # .env lives in the repo root, one level above engines/
    env_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env")
    try:
        with open(env_path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("TYPESAFE_API_KEY="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return ""


def choice(instructions, criteria: dict[str, str | None]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, criteria: list[str]) -> dict:
    return {"type": "score", "instructions": instructions, "criteria": criteria}


def noul(instructions, criteria: dict | None = None) -> dict:
    return {"type": "noul", "instructions": instructions, "criteria": criteria}


class JevClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = SYSTEM_ONE_URL, usage_sink=None):
        self.base_url = base_url
        self.model = model
        # optional callable fed each response's usage row (cost/tokens),
        # so benchmark drivers can account paid usage without re-parsing
        self.usage_sink = usage_sink
        # any other System One endpoint (e.g. a local kev server) needs no key
        if base_url != SYSTEM_ONE_URL:
            self.api_key = api_key or ""
        else:
            self.api_key = api_key or load_api_key()
            if not self.api_key:
                raise RuntimeError(
                    "No TYPESAFE_API_KEY found. Set the environment variable "
                    "or create a .env file in the repo root with "
                    "TYPESAFE_API_KEY=... (see README)."
                )

    def ask(self, state, questions: dict, timeout: int = 120) -> dict:
        body = json.dumps(
            {"model": self.model, "state": state, "questions": questions}
        ).encode("utf-8")
        headers = {"content-type": "application/json"}
        if self.api_key:
            headers["authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url,
            data=body,
            method="POST",
            headers=headers,
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Jev HTTP {exc.code}: {detail}") from exc
        if "answers" not in payload:
            raise RuntimeError(f"Jev response missing answers: {payload!r:.300}")
        if self.usage_sink:
            self.usage_sink(payload.get("usage"))
        payload["_latency_s"] = round(time.perf_counter() - t0, 3)
        return payload

    def classify(
        self,
        texts: list[str],
        labels: dict[str, str | None],
        task: str = "classification",
    ) -> list[str]:
        """Classify many texts in ONE request; returns the chosen label per text."""
        questions = {
            f"t{i}": choice(
                {"task": f"{task} of this text", "text": text}, labels
            )
            for i, text in enumerate(texts)
        }
        payload = self.ask({"task": task, "labels": labels}, questions)
        return [
            payload["answers"][f"t{i}"].get("choice") for i in range(len(texts))
        ]
