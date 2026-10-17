"""Minimal Python client for Jev (TypeSafe AI "System One" classifier).

Wire format follows @typesafe-ai/sdk 0.6.x: POST {model, state, questions}
to https://api.typesafe.ai/v1/systemone with a Bearer key. Questions:

    {"type": "choice", "instructions": str|obj, "criteria": {label: desc|None}}
    {"type": "score",  "instructions": ...,      "criteria": [desc, ...]}
    {"type": "noul",   "instructions": ...,      "criteria": optional}
    {"type": "box",    "instructions": "the coffee mug in `photo`"}

Answers come back as {"model", "answers": {name: {...}}, "usage": {...}}.
`model` repeats the versioned id that answered (jev-latest moves with
releases, so the benches pin versioned ids and record this field).

Jev 1.14 (2026-10-14, blog/jev-1-14) added the danger-zone features:
images (1.13 and 1.14) and Box questions (1.14 only) behind the
TypeSafe-Danger-Zone-Images: true header, and 262k context behind
TypeSafe-Danger-Zone-Long-Context: true. Images ride a top-level
`assets` map — each entry {"type": "image", "format": "png"|"jpg",
"data_base64": ...} keyed by a reference string that must appear (as a
whole value or inside a longer string) somewhere in the state,
instructions or criteria; the model receives the image in its place.
Answer: {"type": "box", "box": {top, bottom, left, right} (fractions of
the image, 0-1), "confidence": 0-1} — always a box, even when the
target is absent, so pair it with a noul. Images bill as input tokens.
docs.typesafe.ai/images and docs.typesafe.ai/primitives/box.

API key: set the TYPESAFE_API_KEY environment variable, or put
`TYPESAFE_API_KEY=...` in a `.env` file in the repo root (gitignored).
Jev is a paid API; each `ask()` call is a request.

No third-party deps on purpose: urllib only.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
# TypeSafe's context ceilings (docs.typesafe.ai/models): the state plus
# all questions combined must fit PER_REQUEST_LIMIT_TOKENS, the state
# plus any one question PER_QUESTION_LIMIT_TOKENS. The
# TypeSafe-Danger-Zone-Long-Context header raises both to 262k — slower
# and less accurate, so it stays off in the benches
PER_REQUEST_LIMIT_TOKENS = 64_000
PER_QUESTION_LIMIT_TOKENS = 32_000
# Jev 1.14's danger-zone headers: images/box questions on one, 262k
# context on the other (DANGER_HEADERS turns both on for image demos,
# where box questions ride along)
DANGER_IMAGES_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
DANGER_LONG_CONTEXT_HEADERS = {"TypeSafe-Danger-Zone-Long-Context": "true"}
DANGER_HEADERS = {**DANGER_IMAGES_HEADERS, **DANGER_LONG_CONTEXT_HEADERS}
# TypeSafe's published input price, $ per 1M input tokens. The response
# usage block reports tokens but no $, so cost charts derive Jev's cost
# as measured input tokens × this price (app.derived_cost); System One
# answers in one forward pass — no output tokens — so input pricing is
# the whole cost (the same rule the README's cost table applies to the
# Clef family)
INPUT_USD_PER_MTOK = 0.042


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


def box(instructions) -> dict:
    """box row (Jev 1.14 only, danger zone): where the named target is
    in the request's image. No criteria — the instructions name the
    target and the image ("the coffee mug in `photo`"); the answer is
    always four edges, so pair with a noul when the target may be
    absent."""
    return {"type": "box", "instructions": instructions}


def image_asset(path) -> dict:
    """One assets-map entry: the image file base64-encoded (PNG/JPEG,
    280-1568 px per side — the API rejects rather than resizes). The
    MIME type comes from the platform resolver and anything besides
    PNG/JPEG raises here — mislabeling, say, a WebP as jpg would only
    fail later, after the upload. The caller picks the reference string
    the entry is keyed under and writes it into the state, instructions
    or criteria."""
    import mimetypes

    path = str(path)
    mime = mimetypes.guess_type(path)[0]
    if mime == "image/png":
        fmt = "png"
    elif mime == "image/jpeg":
        fmt = "jpg"
    else:
        raise ValueError(
            f"image_asset: {path!r} is {mime or 'an unrecognized type'} — "
            "Jev's assets accept PNG and JPEG still images only"
        )
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return {"type": "image", "format": fmt, "data_base64": data}


class JevClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 base_url: str = SYSTEM_ONE_URL, usage_sink=None,
                 extra_headers: dict | None = None):
        self.base_url = base_url
        self.model = model
        # optional callable fed each response's usage row (cost/tokens),
        # so benchmark drivers can account paid usage without re-parsing
        self.usage_sink = usage_sink
        # extra request headers — the danger-zone switches live here
        # (DANGER_IMAGES_HEADERS / DANGER_LONG_CONTEXT_HEADERS)
        self.extra_headers = dict(extra_headers or {})
        # the versioned model id from the latest response (TypeSafe's
        # aliases move with releases; this records what actually answered)
        self.last_model = None
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

    def ask(self, state, questions: dict, timeout: int = 120,
            assets: dict | None = None) -> dict:
        body_dict = {"model": self.model, "state": state,
                     "questions": questions}
        if assets:
            body_dict["assets"] = assets
        body = json.dumps(body_dict).encode("utf-8")
        headers = {"content-type": "application/json", **self.extra_headers}
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
        # Cloudflare /ai/run wraps payloads in a v4 envelope
        # ({"success": true, "result": {model, answers, usage}}); unwrap
        # so every System One caller sees the same flat shape
        if not isinstance(payload, dict):
            raise RuntimeError(f"Jev response is not an object: {payload!r:.300}")
        if "answers" not in payload and isinstance(payload.get("result"), dict):
            payload = payload["result"]
        if "answers" not in payload:
            raise RuntimeError(f"Jev response missing answers: {payload!r:.300}")
        self.last_model = payload.get("model")
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
